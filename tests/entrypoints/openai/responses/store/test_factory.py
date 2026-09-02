# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from argparse import Namespace

import pytest

from vllm.entrypoints.openai.responses.store.factory import (
    ResponsesStoreConfig,
    create_responses_store,
)


def _store_args(disk_path: str) -> Namespace:
    return Namespace(
        responses_store_disk_path=disk_path,
        responses_store_memory_capacity_mb=10,
        responses_store_disk_capacity_mb=100,
        responses_store_memory_low_watermark=0.5,
        responses_store_memory_high_watermark=0.8,
        responses_store_disk_low_watermark=0.6,
        responses_store_disk_high_watermark=0.9,
        responses_store_memory_ttl_seconds=0,
        responses_store_disk_ttl_seconds=120,
        responses_store_cleanup_interval_seconds=5.0,
        responses_store_cleanup_max_candidates=20,
        responses_store_cleanup_max_bytes_mb=8,
    )


def test_responses_store_config_converts_cli_units(tmp_path) -> None:
    disk_path = str(tmp_path / "responses.sqlite3")
    config = ResponsesStoreConfig.from_args(_store_args(disk_path))

    assert config.disk_path == disk_path
    assert config.memory_capacity_bytes == 10 * 1024 * 1024
    assert config.disk_capacity_bytes == 100 * 1024 * 1024
    assert config.memory_low_watermark_bytes == 5 * 1024 * 1024
    assert config.memory_high_watermark_bytes == 8 * 1024 * 1024
    assert config.memory_ttl_seconds is None
    assert config.disk_ttl_seconds == 120

    cleanup_config = config.build_cleanup_config()
    assert cleanup_config.memory.watermarks.max_bytes == 10 * 1024 * 1024
    assert cleanup_config.disk.watermarks.max_bytes == 100 * 1024 * 1024
    assert cleanup_config.memory.budget.max_candidates == 20
    assert cleanup_config.memory.budget.max_bytes == 8 * 1024 * 1024


@pytest.mark.asyncio
async def test_create_responses_store_builds_both_tiers(tmp_path) -> None:
    store, cleanup = create_responses_store(
        _store_args(str(tmp_path / "responses.sqlite3"))
    )
    try:
        assert store.memory_store.max_capacity_bytes == 10 * 1024 * 1024
        assert not cleanup.is_running
    finally:
        await store.close()
