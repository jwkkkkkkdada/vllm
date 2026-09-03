# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import asyncio
import time

import pytest

from vllm.entrypoints.openai.responses.store.base import MemoryEvictionStatus
from vllm.entrypoints.openai.responses.store.cleanup import (
    PeriodicCleanupConfig,
    PeriodicSessionStoreCleanup,
    TierCleanupConfig,
)
from vllm.entrypoints.openai.responses.store.disk import SQLiteSessionStore
from vllm.entrypoints.openai.responses.store.eviction import (
    CapacityWaterMarks,
    EvictionSelectionBudget,
)
from vllm.entrypoints.openai.responses.store.memory import MemorySessionStore
from vllm.entrypoints.openai.responses.store.tiered import TieredSessionStore


async def _wait_until_complete(store: SQLiteSessionStore, session_id: str) -> None:
    for _ in range(200):
        if await store.is_complete(session_id):
            return
        await asyncio.sleep(0.001)
    pytest.fail(f"disk write did not complete for {session_id}")


@pytest.mark.asyncio
async def test_memory_only_store_skips_disk_operations() -> None:
    store = TieredSessionStore(
        MemorySessionStore(max_capacity_bytes=1024 * 1024),
        disk_store=None,
    )

    await store.save("session-1", "response-1", [1, 2])
    await store.save("session-1", "response-2", [3])

    assert not store.disk_enabled
    assert store.disk_store is None
    assert await store.get("session-1") == [1, 2, 3]
    assert await store.exists("session-1")
    assert await store.disk_used_bytes() == 0
    assert not await store.is_disk_complete("session-1")
    assert [state.session_id for state in await store.list()] == ["session-1"]
    assert await store.delete("session-1")
    assert await store.get("session-1") is None


@pytest.mark.asyncio
async def test_memory_only_expiry_requires_rerender() -> None:
    store = TieredSessionStore(
        MemorySessionStore(
            max_capacity_bytes=1024 * 1024,
            mem_idle_ttl_seconds=1,
        ),
        disk_store=None,
    )
    await store.save("session-1", "response-1", [1, 2])
    state = await store.memory_store.get_state_for_eviction("session-1")
    assert state is not None

    result = await store.evict_memory_candidate(
        session_id=state.session_id,
        expected_response_id=state.response_id,
        expected_updated_at=state.updated_at,
        now=int(time.time()) + 2,
    )

    assert result.status is MemoryEvictionStatus.EVICTED_REQUIRES_RERENDER
    assert await store.get("session-1") is None


@pytest.mark.asyncio
async def test_memory_only_pressure_can_force_eviction() -> None:
    store = TieredSessionStore(
        MemorySessionStore(max_capacity_bytes=1024 * 1024),
        disk_store=None,
    )
    await store.save("session-1", "response-1", [1, 2])
    state = await store.memory_store.get_state_for_eviction("session-1")
    assert state is not None

    result = await store.evict_memory_candidate(
        session_id=state.session_id,
        expected_response_id=state.response_id,
        expected_updated_at=state.updated_at,
        now=int(time.time()),
        force=True,
    )

    assert result.status is MemoryEvictionStatus.FORCE_EVICTED
    assert await store.get("session-1") is None


@pytest.mark.asyncio
async def test_pending_disk_data_is_replaced_by_complete_snapshot() -> None:
    disk = SQLiteSessionStore(":memory:", write_interval_seconds=0.05)
    store = TieredSessionStore(
        MemorySessionStore(max_capacity_bytes=1024 * 1024),
        disk,
    )

    try:
        await store.save("session-1", "response-1", [1, 2])
        assert await disk.get_state("session-1") is None

        await store.save("session-1", "response-2", [3])
        await _wait_until_complete(disk, "session-1")
        assert await store.memory_store.delete("session-1")
        assert await store.get("session-1") == [1, 2, 3]
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_cleanup_scans_metadata_without_materializing_tokens(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    disk = SQLiteSessionStore(":memory:", write_interval_seconds=0.001)
    store = TieredSessionStore(
        MemorySessionStore(max_capacity_bytes=1024 * 1024),
        disk,
    )
    watermarks = CapacityWaterMarks(
        max_bytes=1024 * 1024,
        low_watermark_bytes=512 * 1024,
        high_watermark_bytes=768 * 1024,
    )
    tier_config = TierCleanupConfig(
        watermarks=watermarks,
        budget=EvictionSelectionBudget(
            max_candidates=16,
            max_bytes=1024 * 1024,
        ),
    )
    cleanup = PeriodicSessionStoreCleanup(
        store,
        PeriodicCleanupConfig(
            interval_seconds=1,
            memory=tier_config,
            disk=tier_config,
        ),
    )

    async def fail_full_list() -> list[object]:
        raise AssertionError("cleanup must not request full session states")

    def fail_decrypt(*_args: object) -> list[int]:
        raise AssertionError("cleanup must not decrypt token data")

    try:
        await store.save("session-1", "response-1", [1, 2, 3])
        await _wait_until_complete(disk, "session-1")
        monkeypatch.setattr(store, "list", fail_full_list)
        monkeypatch.setattr(store.memory_store, "list", fail_full_list)
        monkeypatch.setattr(disk, "list", fail_full_list)
        monkeypatch.setattr(disk, "_decrypt_token_ids", fail_decrypt)

        result = await cleanup.run_once()

        assert result.scanned_session_count == 1
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_failed_disk_batch_is_not_retried() -> None:
    disk = SQLiteSessionStore(":memory:", write_interval_seconds=0.001)
    store = TieredSessionStore(
        MemorySessionStore(max_capacity_bytes=1024 * 1024),
        disk,
    )
    attempts = 0

    def fail_write(_batch: object) -> None:
        nonlocal attempts
        attempts += 1
        raise OSError("simulated disk failure")

    disk._write_batch_sync = fail_write  # type: ignore[method-assign]

    try:
        await store.save("session-1", "response-1", [1, 2])
        for _ in range(200):
            if attempts:
                break
            await asyncio.sleep(0.001)

        assert attempts == 1
        await asyncio.sleep(0.01)
        assert attempts == 1
        assert await store.get("session-1") == [1, 2]
        assert await disk.needs_snapshot("session-1")
    finally:
        await store.close()
