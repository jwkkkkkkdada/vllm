# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from argparse import Namespace
from pathlib import Path
from unittest.mock import AsyncMock, Mock

import pytest
from fastapi import FastAPI

from vllm.entrypoints.openai.responses.store.service import ResponsesStoreService
from vllm.entrypoints.serve.utils.server_utils import lifespan


@pytest.mark.asyncio
async def test_lifespan_starts_and_closes_responses_store(monkeypatch) -> None:
    service = Mock()
    service.close = AsyncMock()
    create_service = Mock(return_value=service)
    monkeypatch.setattr(ResponsesStoreService, "from_cli_args", create_service)
    monkeypatch.setattr(
        "vllm.entrypoints.serve.utils.server_utils.freeze_gc_heap",
        Mock(),
    )

    args = Namespace()
    app = FastAPI()
    app.state.args = args
    app.state.log_stats = False
    app.state.responses_store_enabled = True
    app.state.responses_store_service = None

    async with lifespan(app):
        create_service.assert_called_once_with(args)
        service.start.assert_called_once_with()
        assert app.state.responses_store_service is service

    service.close.assert_awaited_once_with()


@pytest.mark.asyncio
async def test_lifespan_leaves_responses_store_disabled(monkeypatch) -> None:
    create_service = Mock()
    monkeypatch.setattr(ResponsesStoreService, "from_cli_args", create_service)
    monkeypatch.setattr(
        "vllm.entrypoints.serve.utils.server_utils.freeze_gc_heap",
        Mock(),
    )

    app = FastAPI()
    app.state.args = Namespace()
    app.state.log_stats = False
    app.state.responses_store_enabled = False
    app.state.responses_store_service = None

    async with lifespan(app):
        create_service.assert_not_called()


@pytest.mark.asyncio
async def test_responses_store_service_starts_without_business_wiring(
    tmp_path: Path,
) -> None:
    args = Namespace(
        enable_responses_store=True,
        responses_store_disk_enabled=True,
        responses_store_disk_path=str(tmp_path / "responses.sqlite3"),
        responses_store_key_file=None,
        responses_store_memory_capacity_mb=1,
        responses_store_disk_capacity_mb=2,
        responses_store_memory_low_watermark=0.5,
        responses_store_memory_high_watermark=0.8,
        responses_store_disk_low_watermark=0.6,
        responses_store_disk_high_watermark=0.9,
        responses_store_memory_ttl_seconds=30,
        responses_store_disk_ttl_seconds=60,
        responses_store_cleanup_interval_seconds=60,
        responses_store_cleanup_max_candidates=8,
        responses_store_cleanup_max_bytes_mb=1,
        responses_store_num_shards=4,
        responses_store_disk_write_interval_seconds=0.01,
    )

    service = ResponsesStoreService.from_cli_args(args)
    service.start()
    assert service.is_running
    assert service.store.disk_enabled

    await service.close()
    assert not service.is_running
