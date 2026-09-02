# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, call

import pytest

from vllm.entrypoints.openai.responses.serving import OpenAIServingResponses


class _EngineClient:
    async def generate(self, *args, **kwargs):
        yield SimpleNamespace(
            outputs=[SimpleNamespace(token_ids=(11, 12))],
        )
        yield SimpleNamespace(
            outputs=[SimpleNamespace(token_ids=(13,))],
        )


@pytest.mark.asyncio
async def test_generation_persists_token_deltas_by_session() -> None:
    session_store = SimpleNamespace(save=AsyncMock())
    serving = SimpleNamespace(
        model_config=SimpleNamespace(max_model_len=1024),
        engine_client=_EngineClient(),
        session_store=session_store,
        _log_inputs=Mock(),
    )
    context = SimpleNamespace(
        append_output=Mock(),
        need_builtin_tool_call=Mock(return_value=False),
    )

    results = [
        result
        async for result in OpenAIServingResponses._generate_with_builtin_tools(
            serving,
            request_id="resp-1",
            engine_input={},
            sampling_params=SimpleNamespace(),
            context=context,
            session_id="session-1",
            persist_token_ids=True,
        )
    ]

    assert results == [context, context]
    assert session_store.save.await_args_list == [
        call("session-1", "resp-1", [11, 12]),
        call("session-1", "resp-1", [13]),
    ]
