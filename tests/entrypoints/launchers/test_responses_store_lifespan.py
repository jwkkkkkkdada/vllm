# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from unittest.mock import AsyncMock, Mock

import pytest
from fastapi import FastAPI

from vllm.entrypoints.launchers.utils.server_utils import lifespan


@pytest.mark.asyncio
async def test_lifespan_starts_cleanup_and_closes_store(monkeypatch) -> None:
    monkeypatch.setattr(
        "vllm.entrypoints.launchers.utils.server_utils.freeze_gc_heap",
        Mock(),
    )
    cleanup = Mock()
    cleanup.stop = AsyncMock()
    store = Mock()
    store.close = AsyncMock()

    app = FastAPI()
    app.state.log_stats = False
    app.state.responses_store_cleanup = cleanup
    app.state.responses_session_store = store

    async with lifespan(app):
        cleanup.start.assert_called_once_with()

    cleanup.stop.assert_awaited_once_with()
    store.close.assert_awaited_once_with()
