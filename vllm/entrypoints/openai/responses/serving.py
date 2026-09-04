# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import asyncio
import time
from collections import deque
from collections.abc import AsyncGenerator, AsyncIterator, Callable, Mapping, Sequence
from contextlib import AsyncExitStack
from dataclasses import dataclass, field
from http import HTTPStatus
from typing import Any, Final

# 修改后
from store.service import ResponsesStoreService
from store.tiered import TieredSessionStore

from fastapi import Request
from openai.types.responses import (
    ResponseFunctionToolCall,
    ResponseOutputItem,
    ResponseOutputMessage,
    ResponseOutputText,
    ResponseStatus,
    response_text_delta_event,
)
from openai.types.responses.response_output_text import Logprob, LogprobTopLogprob
from openai.types.responses.tool import Mcp, Tool
from openai_harmony import Message as OpenAIHarmonyMessage
from pydantic import TypeAdapter

from vllm import envs
from vllm.config.utils import replace
from vllm.engine.protocol import EngineClient
from vllm.entrypoints.chat_utils import ChatTemplateContentFormatOption

from vllm.entrypoints.generate.base.serving import (
    GenerateBaseServing,
    GenerationError,
)
from vllm.entrypoints.mcp.tool_server import ToolServer
from vllm.entrypoints.openai.engine.protocol import (
    DeltaMessage,
    ErrorResponse,
    RequestResponseMetadata,
)
from vllm.entrypoints.openai.models.serving import OpenAIServingModels
from vllm.entrypoints.openai.parser.harmony_utils import (
    build_harmony_preamble,
    extract_instructions_from_messages,
    get_user_message,
    has_custom_tools,
    render_for_completion,
)
from vllm.entrypoints.openai.responses.context import (
    ConversationContext,
    HarmonyContext,
    ParsableContext,
    SimpleContext,
)
from vllm.entrypoints.openai.responses.harmony import (
    construct_harmony_previous_input_messages,
    harmony_to_response_output,
    response_input_to_harmony,
)
from vllm.entrypoints.openai.responses.protocol import (
    InputTokensDetails,
    OutputTokensDetails,
    ResponseCompletedEvent,
    ResponseCreatedEvent,
    ResponseInProgressEvent,
    ResponseInputOutputItem,
    ResponseInputOutputMessage,
    ResponsesRequest,
    ResponsesResponse,
    ResponseUsage,
    StreamingResponsesResponse,
)
from vllm.entrypoints.openai.responses.store.base import SessionStore
from vllm.entrypoints.openai.responses.streaming_events import (
    SimpleStreamingEventProcessor,
    StreamingState,
    _StateType,
    emit_content_delta_events,
    emit_previous_item_done_events,
    emit_tool_action_events,
    split_delta,
)
from vllm.entrypoints.openai.responses.utils import (
    build_response_output_items,
    construct_input_messages,
    construct_tool_dicts,
    extract_function_tool_names,
    extract_tool_types,
)
from vllm.entrypoints.serve.utils.api_utils import get_max_tokens
from vllm.entrypoints.serve.utils.request_logger import RequestLogger
from vllm.exceptions import VLLMValidationError
from vllm.inputs import EngineInput, tokens_input
from vllm.logger import init_logger
from vllm.logprobs import Logprob as SampleLogprob
from vllm.logprobs import SampleLogprobs
from vllm.lora.request import LoRARequest
from vllm.outputs import CompletionOutput
from vllm.parser import Parser, ParserManager
from vllm.renderers.online_renderer import OnlineRenderer
from vllm.sampling_params import SamplingParams, StructuredOutputsParams
from vllm.tokenizers import TokenizerLike
from vllm.utils import random_uuid
from vllm.utils.collection_utils import as_list

logger = init_logger(__name__)

@dataclass(slots=True)
class _SessionTokenState:
    """单个 Responses 请求内部维护的增量 token 状态。"""

    store: SessionStore
    session_id: str
    should_save: bool

    # SessionStore.get() 返回的历史 token 副本。
    reused_prompt_token_ids: list[int] | None = None

    # 本次请求开始前的历史长度，用于最终切出 delta。
    history_length: int = 0

    # 当前模型子请求使用的完整 prompt。
    prompt_token_ids: list[int] | None = None

    # 当前模型子请求生成的输出。
    output_token_ids: list[int] = field(default_factory=list)

    def merge_prompt(
        self,
        engine_input: EngineInput,
        bos_token_id: int | None,
    ) -> None:
        """将当前用户输入的 token 追加到跨请求历史 token 后。"""

        prompt_token_ids = engine_input.get("prompt_token_ids")
        if prompt_token_ids is None:
            raise ValueError(
                "Incremental token reuse requires tokenized text input."
            )

        reused_prompt_token_ids = self.reused_prompt_token_ids
        if reused_prompt_token_ids is None:
            return

        # 增量片段不应再次引入 BOS。
        start = int(
            bos_token_id is not None
            and bool(prompt_token_ids)
            and prompt_token_ids[0] == bos_token_id
        )

        reused_prompt_token_ids.extend(prompt_token_ids[start:])
        engine_input["prompt_token_ids"] = reused_prompt_token_ids

    def begin_turn(self, engine_input: EngineInput) -> None:
        """记录当前模型子请求的完整 prompt。"""

        prompt_token_ids = engine_input.get("prompt_token_ids")
        if prompt_token_ids is None:
            raise ValueError(
                "Session token storage requires tokenized text input."
            )

        self.prompt_token_ids = prompt_token_ids
        self.output_token_ids.clear()

    def append_output(self, output: Any) -> None:
        """收集当前模型子请求生成的输出 token。"""

        self.output_token_ids.extend(output.outputs[0].token_ids)

    def merge_tool_delta(
        self,
        engine_input: EngineInput,
        bos_token_id: int | None,
        eos_token_id: int | None,
    ) -> None:
        """拼接模型工具调用输出以及新工具结果。"""

        tool_delta_token_ids = engine_input.get("prompt_token_ids")
        if tool_delta_token_ids is None:
            raise ValueError(
                "Incremental tool tokenization requires tokenized input."
            )

        prompt_token_ids = self.prompt_token_ids
        if prompt_token_ids is None:
            raise ValueError("No active prompt is available.")

        # 提交刚刚由模型生成的工具调用。
        prompt_token_ids.extend(self.output_token_ids)
        self.output_token_ids.clear()

        # 模型输出通常不包含触发停止的 EOS。
        if eos_token_id is not None and (
            not prompt_token_ids
            or prompt_token_ids[-1] != eos_token_id
        ):
            prompt_token_ids.append(eos_token_id)

        start = int(
            bos_token_id is not None
            and bool(tool_delta_token_ids)
            and tool_delta_token_ids[0] == bos_token_id
        )
        prompt_token_ids.extend(tool_delta_token_ids[start:])

        engine_input["prompt_token_ids"] = prompt_token_ids
        self.prompt_token_ids = prompt_token_ids

    def build_delta(self, eos_token_id: int | None) -> list[int]:
        """生成本次请求需要传给 SessionStore.save() 的 token。"""

        assert self.prompt_token_ids is not None

        # 切片会生成独立列表，不会修改完整 prompt。
        delta_token_ids = self.prompt_token_ids[self.history_length :]
        delta_token_ids.extend(self.output_token_ids)

        if eos_token_id is not None and (
            not delta_token_ids
            or delta_token_ids[-1] != eos_token_id
        ):
            delta_token_ids.append(eos_token_id)

        return delta_token_ids


def _extract_allowed_tools_from_mcp_requests(
    tools: list[Tool],
) -> dict[str, list[str] | None]:
    """
    Extract allowed_tools mapping from MCP tool requests.

    Returns a dictionary mapping server_label to allowed_tools list.
    Handles both list format and McpAllowedToolsMcpToolFilter object format.

    Special handling:
    - If allowed_tools is None, returns None (allows all tools)
    - If allowed_tools contains "*", returns None (allows all tools)
    - Otherwise, returns the list of specific tool names

    This function can be reused for both harmony and non-harmony MCP calls.
    """
    allowed_tools_map: dict[str, list[str] | None] = {}
    for tool in tools:
        if not isinstance(tool, Mcp):
            continue

        # allowed_tools can be a list or an object with tool_names
        # Extract the actual list of tool names
        allowed_tools_val = None
        if tool.allowed_tools is not None:
            if isinstance(tool.allowed_tools, list):
                allowed_tools_val = tool.allowed_tools
            elif hasattr(tool.allowed_tools, "tool_names"):
                # It's an McpAllowedToolsMcpToolFilter object
                allowed_tools_val = tool.allowed_tools.tool_names

        # Normalize "*" to None (both mean "allow all tools")
        if allowed_tools_val is not None and "*" in allowed_tools_val:
            allowed_tools_val = None

        allowed_tools_map[tool.server_label] = allowed_tools_val
    return allowed_tools_map


class OpenAIServingResponses(GenerateBaseServing):
    def __init__(
        self,
        engine_client: EngineClient,
        models: OpenAIServingModels,
        online_renderer: OnlineRenderer,
        *,
        request_logger: RequestLogger | None,
        chat_template: str | None,
        chat_template_content_format: ChatTemplateContentFormatOption,
        return_tokens_as_token_ids: bool = False,
        reasoning_parser: str = "",
        enable_auto_tools: bool = False,
        tool_parser: str | None = None,
        tool_server: ToolServer | None = None,
        enable_prompt_tokens_details: bool = False,
        enable_force_include_usage: bool = False,
        enable_log_outputs: bool = False,
        default_chat_template_kwargs: dict[str, Any] | None = None,
        # 增加一个可选参数
        responses_store_service: ResponsesStoreService | None = None,
    ) -> None:
        super().__init__(
            engine_client=engine_client,
            models=models,
            request_logger=request_logger,
            return_tokens_as_token_ids=return_tokens_as_token_ids,
        )

        self.online_renderer = online_renderer
        self.chat_template = chat_template
        self.chat_template_content_format: Final = chat_template_content_format
        self.chat_template_kwargs = default_chat_template_kwargs or {}
        self.enable_log_outputs = enable_log_outputs

        # Set up the unified parser - either a unified parser or fall back to
        # separate parsers accessed through the parser interface
        self.parser = ParserManager.get_parser(
            tool_parser_name=tool_parser,
            reasoning_parser_name=reasoning_parser,
            enable_auto_tools=enable_auto_tools,
            model_name=self.model_config.model,
            is_harmony=self.model_config.hf_config.model_type == "gpt_oss",
        )
        self.enable_prompt_tokens_details = enable_prompt_tokens_details
        self.enable_force_include_usage = enable_force_include_usage

        self.default_sampling_params = self.model_config.get_diff_sampling_param()
        mc = self.model_config
        self.override_max_tokens = (
            self.default_sampling_params.get("max_tokens")
            if mc.generation_config not in ("auto", "vllm")
            else getattr(mc, "override_generation_config", {}).get("max_new_tokens")
        )

        # If False (default), the "store" option is (silently) ignored and the
        # response is not stored. If True, the response is stored in memory.
        # NOTE(woosuk): This may not be intuitive for users, as the default
        # behavior in OpenAI's Responses API is to store the response, but
        # vLLM's default behavior is not.
        self.enable_store = envs.VLLM_ENABLE_RESPONSES_API_STORE
        if self.enable_store:
            logger.warning_once(
                "`VLLM_ENABLE_RESPONSES_API_STORE` is enabled. This may "
                "cause a memory leak since we never remove responses from "
                "the store."
            )

        self.use_harmony = self.model_config.hf_config.model_type == "gpt_oss"
        if self.use_harmony:
            logger.warning(
                "For gpt-oss, we ignore --enable-auto-tool-choice "
                "and always enable tool use."
            )
        self.enable_auto_tools = enable_auto_tools
        # HACK(woosuk): This is a hack. We should use a better store.
        # FIXME: If enable_store=True, this may cause a memory leak since we
        # never remove responses from the store.
        self.response_store: dict[str, ResponsesResponse] = {}
        self.response_store_lock = asyncio.Lock()

        # HACK(wuhang): This is a hack. We should use a better store.
        # FIXME: If enable_store=True, this may cause a memory leak since we
        # never remove events from the store.
        self.event_store: dict[
            str, tuple[deque[StreamingResponsesResponse], asyncio.Event]
        ] = {}

        self.background_tasks: dict[str, asyncio.Task] = {}

        self.tool_server = tool_server

        self.ResponsesStoreService = responses_store_service

    def _effective_chat_template_kwargs(
        self, request: ResponsesRequest
    ) -> dict[str, Any]:
        return (
            request.build_chat_params(
                self.chat_template,
                self.chat_template_content_format,
            )
            .with_defaults(self.chat_template_kwargs)
            .chat_template_kwargs
        )

    def _make_response_parser(
        self,
        request: ResponsesRequest,
        tokenizer: TokenizerLike,
        chat_template_kwargs: dict[str, Any],
    ) -> Parser | None:
        if self.parser is None:
            return None
        return self.parser(
            tokenizer,
            request.tools,
            chat_template_kwargs=chat_template_kwargs,
            model_config=self.model_config,
        )

    def _validate_generator_input(
        self,
        engine_input: EngineInput,
    ) -> ErrorResponse | None:
        """Add validations to the input to the generator here."""
        prompt_len = self._extract_prompt_len(engine_input)
        max_model_len = self.model_config.max_model_len

        if prompt_len >= max_model_len:
            error_message = (
                f"The engine prompt length {prompt_len} "
                f"exceeds the max_model_len {max_model_len}. "
                "Please reduce prompt."
            )
            return self.create_error_response(
                err_type="invalid_request_error",
                message=error_message,
                status_code=HTTPStatus.BAD_REQUEST,
                param="input",
            )

        return None

    def _validate_create_responses_input(
        self, request: ResponsesRequest
    ) -> ErrorResponse | None:
        if self.use_harmony and request.is_include_output_logprobs():
            return self.create_error_response(
                err_type="invalid_request_error",
                message="logprobs are not supported with gpt-oss models",
                status_code=HTTPStatus.BAD_REQUEST,
                param="logprobs",
            )
        if request.store and not self.enable_store and request.background:
            return self.create_error_response(
                err_type="invalid_request_error",
                message=(
                    "This vLLM engine does not support `store=True` and "
                    "therefore does not support the background mode. To "
                    "enable these features, set the environment variable "
                    "`VLLM_ENABLE_RESPONSES_API_STORE=1` when launching "
                    "the vLLM server."
                ),
                status_code=HTTPStatus.BAD_REQUEST,
                param="background",
            )
        return None

    def _make_incremental_context_miss_error(
        self,
        session_id: str,
    ) -> ErrorResponse:
        return self.create_error_response(
            err_type="incremental_context_miss",
            message=(
                "Incremental context is no longer available. "
                "Resend the request with full context."
            ),
            status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
            param=session_id,
        )

    @staticmethod
    def _resolve_session_store(
        raw_request: Request | None,
    ) -> SessionStore | None:
        if raw_request is None:
            return None

        service = getattr(
            raw_request.app.state,
            "responses_store_service",
            None,
        )
        return None if service is None else service.store

    async def _make_session_token_state(
        self,
        request: ResponsesRequest,
        raw_request: Request | None,
    ) -> _SessionTokenState | ErrorResponse | None:
        use_incremental_token = request.use_incremental_token
        use_store = request.use_store

        if not use_incremental_token and not use_store:
            return None

        # 本次实现没有覆盖 Harmony 的跨请求增量渲染语义。
        if use_incremental_token and self.use_harmony:
            return self.create_error_response(
                err_type="invalid_request_error",
                message=(
                    "Incremental token reuse is not supported for "
                    "Harmony models."
                ),
                status_code=HTTPStatus.BAD_REQUEST,
                param="use_incremental_token",
            )

        if raw_request is None:
            return self.create_error_response(
                err_type="invalid_request_error",
                message="The x-session-id header is required.",
                status_code=HTTPStatus.BAD_REQUEST,
                param="x-session-id",
            )

        session_id = raw_request.headers.get("x-session-id")
        if not session_id:
            return self.create_error_response(
                err_type="invalid_request_error",
                message="The x-session-id header is required.",
                status_code=HTTPStatus.BAD_REQUEST,
                param="x-session-id",
            )

        session_store = self._resolve_session_store(raw_request)
        if session_store is None:
            return self.create_error_response(
                err_type="session_store_unavailable",
                message="SessionStore is not configured.",
                status_code=HTTPStatus.SERVICE_UNAVAILABLE,
                param=session_id,
            )

        reused_prompt_token_ids = None
        if use_incremental_token:
            # 每个请求仅在这里调用一次 get()。
            reused_prompt_token_ids = await session_store.get(session_id)
            if reused_prompt_token_ids is None:
                return self._make_incremental_context_miss_error(session_id)

        return _SessionTokenState(
            store=session_store,
            session_id=session_id,
            should_save=use_store,
            reused_prompt_token_ids=reused_prompt_token_ids,
            history_length=(
                len(reused_prompt_token_ids)
                if reused_prompt_token_ids is not None
                else 0
            ),
        )

    async def create_responses(
        self,
        request: ResponsesRequest,
        raw_request: Request | None = None,
    ) -> (
        AsyncGenerator[StreamingResponsesResponse, None]
        | ResponsesResponse
        | ErrorResponse
    ):
        return await self._with_kv_transfer_rejection_cleanup(
            self._create_responses(request, raw_request), request, raw_request
        )

    async def _create_responses(
        self, request: ResponsesRequest, raw_request: Request | None = None
    ) -> (
        AsyncGenerator[StreamingResponsesResponse, None]
        | ResponsesResponse
        | ErrorResponse
    ):
        error_check_ret = await self._check_model(request)
        if error_check_ret is not None:
            logger.error("Error with model %s", error_check_ret)
            return error_check_ret
        maybe_validation_error = self._validate_create_responses_input(request)
        if maybe_validation_error is not None:
            return maybe_validation_error

        # If the engine is dead, raise the engine's DEAD_ERROR.
        # This is required for the streaming case, where we return a
        # success status before we actually start generating text :).
        if self.engine_client.errored:
            raise self.engine_client.dead_error

        if request.store and not self.enable_store:
            # Disable the store option.
            # NOTE(woosuk): Although returning an error is possible, we opted
            # to implicitly disable store and process the request anyway, as
            # we assume most users do not intend to actually store the response
            # (i.e., their request's `store=True` just because it's the default
            # value).
            request.store = False

        session_token_state = await self._make_session_token_state(
            request,
            raw_request,
        )
        if isinstance(session_token_state, ErrorResponse):
            return session_token_state

        lora_request = self._maybe_get_adapters(request)
        model_name = self.models.model_name(lora_request)

        if self.use_harmony:
            messages, engine_inputs = self._make_request_with_harmony(request)
        else:
            messages, engine_inputs = await self._make_request(request, session_token_state)

        request_metadata = RequestResponseMetadata(request_id=request.request_id)
        if raw_request:
            raw_request.state.request_metadata = request_metadata

        # Schedule the request and get the result generator.
        max_model_len = self.model_config.max_model_len
        generators: list[AsyncGenerator[ConversationContext, None]] = []

        # Only include builtin tools that the request actually asked for.
        # Without this filter, tools registered on the server (e.g. via
        # --tool-server demo) would be available for execution even when
        # the request didn't enable them.
        requested_tool_types = extract_tool_types(request.tools)
        builtin_tool_list: list[str] = []
        if self.tool_server is not None:
            if (
                self.tool_server.has_tool("browser")
                and "web_search_preview" in requested_tool_types
            ):
                builtin_tool_list.append("browser")
            if (
                self.tool_server.has_tool("python")
                and "code_interpreter" in requested_tool_types
            ):
                builtin_tool_list.append("python")
            if (
                self.tool_server.has_tool("container")
                and "container" in requested_tool_types
            ):
                builtin_tool_list.append("container")

        if self.tool_server is not None:
            available_tools = builtin_tool_list
        else:
            assert len(builtin_tool_list) == 0
            available_tools = []
        tokenizer = self.renderer.get_tokenizer()

        for engine_input in engine_inputs:
            maybe_error = self._validate_generator_input(engine_input)
            if maybe_error is not None:
                return maybe_error

            default_max_tokens = get_max_tokens(
                max_model_len,
                request.max_output_tokens,
                self._extract_prompt_len(engine_input),
                self.default_sampling_params,
                self.override_max_tokens,
                truncate_prompt_tokens=(
                    -1 if request.truncation != "disabled" else None
                ),
            )

            sampling_params = request.to_sampling_params(
                default_max_tokens, self.default_sampling_params
            )

            trace_headers = (
                None
                if raw_request is None
                else await self._get_trace_headers(raw_request.headers)
            )

            chat_template_kwargs = self._effective_chat_template_kwargs(request)
            response_parser = self._make_response_parser(
                request, tokenizer, chat_template_kwargs
            )

            context: ConversationContext
            function_tool_names = extract_function_tool_names(request.tools)
            if self.use_harmony:
                context = HarmonyContext(
                    messages,
                    available_tools,
                    function_tool_names,
                    response_parser=response_parser,
                )
            else:
                if envs.VLLM_USE_EXPERIMENTAL_PARSER_CONTEXT:
                    # This is a feature in development for parsing
                    # tokens during generation instead of at the end
                    context = ParsableContext(
                        response_messages=messages,
                        tokenizer=tokenizer,
                        parser_cls=self.parser,
                        request=request,
                        response_parser=response_parser,
                        available_tools=available_tools,
                        chat_template=self.chat_template,
                        chat_template_content_format=self.chat_template_content_format,
                        enable_auto_tools=self.enable_auto_tools,
                    )
                else:
                    context = SimpleContext(
                        response_parser=response_parser,
                    )

            if (
                context.response_parser is not None
                and context.response_parser.reasoning_parser is not None
            ):
                reasoning_parser_kwargs = {
                    "chat_template_kwargs": chat_template_kwargs,
                }
                if (
                    isinstance(
                        struct_out := sampling_params.structured_outputs,
                        StructuredOutputsParams,
                    )
                    and struct_out.all_non_structural_tag_constraints_none()
                ):
                    sampling_params.structured_outputs = replace(
                        struct_out,
                        structural_tag=(
                            context.response_parser.reasoning_parser.prepare_structured_tag(
                                struct_out.structural_tag, self.tool_server
                            )
                        ),
                    )
            generator = self._generate_with_builtin_tools(
                request_id=request.request_id,
                engine_input=engine_input,
                sampling_params=sampling_params,
                context=context,
                lora_request=lora_request,
                priority=request.priority,
                trace_headers=trace_headers,
                reasoning_parser_kwargs=reasoning_parser_kwargs
                if self.parser and self.parser.reasoning_parser_cls is not None
                else None,
                session_token_state=session_token_state,
            )
            generators.append(generator)

        assert len(generators) == 1
        (result_generator,) = generators

        if request.background:
            created_time = int(time.time())
            response = ResponsesResponse.from_request(
                request,
                sampling_params,
                model_name=model_name,
                created_time=created_time,
                output=[],
                status="queued",
                usage=None,
            )
            async with self.response_store_lock:
                self.response_store[response.id] = response

            # Run the request in the background.
            if request.stream:
                task = asyncio.create_task(
                    self._run_background_request_stream(
                        request,
                        sampling_params,
                        result_generator,
                        context,
                        model_name,
                        tokenizer,
                        request_metadata,
                        created_time,
                        session_token_state,
                    ),
                    name=f"create_{request.request_id}",
                )
            else:
                task = asyncio.create_task(
                    self._run_background_request(
                        request,
                        sampling_params,
                        result_generator,
                        context,
                        model_name,
                        tokenizer,
                        request_metadata,
                        created_time,
                        session_token_state,
                    ),
                    name=f"create_{response.id}",
                )

            # For cleanup.
            response_id = response.id
            self.background_tasks[response_id] = task
            task.add_done_callback(
                lambda _: self.background_tasks.pop(response_id, None)
            )

            if request.stream:
                return self.responses_background_stream_generator(request.request_id)
            return response

        if request.stream:
            return self.responses_stream_generator(
                request,
                sampling_params,
                result_generator,
                context,
                model_name,
                tokenizer,
                request_metadata,
                session_token_state=session_token_state,
            )

        return await self.responses_full_generator(
            request,
            sampling_params,
            result_generator,
            context,
            model_name,
            tokenizer,
            request_metadata,
            session_token_state=session_token_state,
        )

    async def _make_request(
        self,
        request: ResponsesRequest,
        session_token_state: _SessionTokenState | None,
    ):
        is_incremental = request.use_incremental_token
        if is_incremental:
            assert session_token_state is not None

        # tools 和 instructions 已经位于存储的历史 token 中。
        tool_dicts = (
            None
            if is_incremental
            else construct_tool_dicts(request.tools, request.tool_choice)
        )
        # Construct the input messages.
        messages = construct_input_messages(
            request_instructions=(
                None if is_incremental else request.instructions
            ),
            request_input=request.input,
        )
        chat_template_kwargs = self._effective_chat_template_kwargs(request)
        if is_incremental:
            chat_template_kwargs["use_incremental_token"] = True
        _, engine_inputs = await self.online_renderer.preprocess_chat(
            request,
            messages,
            default_template=self.chat_template,
            default_template_content_format=self.chat_template_content_format,
            default_template_kwargs=chat_template_kwargs,
            tool_dicts=tool_dicts,
            parser=self.parser,
        )
        if is_incremental:
            assert session_token_state is not None
            (engine_input,) = engine_inputs
            session_token_state.merge_prompt(
                engine_input,
                self.renderer.get_bos_token_id(),
            )
        return messages, engine_inputs

    async def _render_next_turn(
        self,
        request: ResponsesRequest,
        messages: list[ResponseInputOutputItem],
        tool_dicts: list[dict[str, Any]] | None,
        parser: type[Parser] | None,
        chat_template: str | None,
        chat_template_content_format: ChatTemplateContentFormatOption,
        session_token_state: _SessionTokenState | None,
        new_item_count: int,
    ):
        incremental_tool_turn = request.use_incremental_token

        if incremental_tool_turn:
            assert session_token_state is not None
            assert new_item_count > 0
            split_index = len(messages) - new_item_count

            context_messages = construct_input_messages(
                request_input=messages[:split_index],
            )
            new_messages = construct_input_messages(
                request_input=messages[split_index:],
            )
        else:
            context_messages = None
            new_messages = construct_input_messages(
                request_input=messages,
            )

        chat_template_kwargs = self._effective_chat_template_kwargs(request)
        if incremental_tool_turn:
            chat_template_kwargs["incremental_context"] = context_messages
        _, engine_inputs = await self.online_renderer.preprocess_chat(
            request,
            new_messages,
            default_template=chat_template,
            default_template_content_format=chat_template_content_format,
            default_template_kwargs=chat_template_kwargs,
            tool_dicts=None if incremental_tool_turn else tool_dicts,
            parser=parser,
        )
        return engine_inputs

    async def _generate_with_builtin_tools(
        self,
        request_id: str,
        engine_input: EngineInput,
        sampling_params: SamplingParams,
        context: ConversationContext,
        lora_request: LoRARequest | None = None,
        priority: int = 0,
        trace_headers: Mapping[str, str] | None = None,
        reasoning_parser_kwargs: dict[str, Any] | None = None,
        session_token_state: _SessionTokenState | None = None,
    ):
        max_model_len = self.model_config.max_model_len

        orig_priority = priority
        sub_request = 0
        while True:
            # Ensure that each sub-request has a unique request id.
            sub_request_id = f"{request_id}_{sub_request}"

            self._log_inputs(
                sub_request_id,
                engine_input,
                params=sampling_params,
                lora_request=lora_request,
            )

            if session_token_state is not None:
                session_token_state.begin_turn(engine_input)

            generator = self.engine_client.generate(
                engine_input,
                sampling_params,
                sub_request_id,
                lora_request=lora_request,
                trace_headers=trace_headers,
                priority=priority,
                reasoning_parser_kwargs=reasoning_parser_kwargs,
            )

            async for res in generator:
                if session_token_state is not None:
                    session_token_state.append_output(res)
                context.append_output(res)
                # NOTE(woosuk): The stop condition is handled by the engine.
                yield context

            if not context.need_builtin_tool_call():
                # The model did not ask for a tool call, so we're done.
                break

            # Call the tool and update the context with the result.
            tool_output = await context.call_tool()
            context.append_tool_output(tool_output)

            # TODO: uncomment this and enable tool output streaming
            # yield context

            # Create inputs for the next turn.
            # Render the next prompt token ids and update sampling_params.
            if isinstance(context, HarmonyContext):
                token_ids = context.render_for_completion()
                engine_input = tokens_input(token_ids)

                sampling_params.max_tokens = max_model_len - len(token_ids)
            elif isinstance(context, ParsableContext):
                (engine_input,) = await self._render_next_turn(
                    context.request,
                    context.response_messages,
                    context.tool_dicts,
                    context.parser_cls,
                    context.chat_template,
                    context.chat_template_content_format,
                    session_token_state,
                    new_item_count=len(tool_output),
                )

                if context.request.use_incremental_token:
                    assert session_token_state is not None
                    session_token_state.merge_tool_delta(
                        engine_input,
                        self.renderer.get_bos_token_id(),
                        self.renderer.get_eos_token_id(),
                    )

                sampling_params.max_tokens = get_max_tokens(
                    max_model_len,
                    context.request.max_output_tokens,
                    self._extract_prompt_len(engine_input),
                    self.default_sampling_params,  # type: ignore
                    self.override_max_tokens,  # type: ignore
                    truncate_prompt_tokens=(
                        -1 if context.request.truncation != "disabled" else None
                    ),
                )

            # OPTIMIZATION
            priority = orig_priority - 1
            sub_request += 1

    def _make_request_with_harmony(
        self,
        request: ResponsesRequest,
    ):
        if request.tool_choice not in ("auto", "none"):
            raise NotImplementedError(
                "Only 'auto' or 'none' tool_choice is supported "
                "in response API with Harmony"
            )

        arrival_time = time.time()
        messages = self._construct_input_messages_with_harmony(request)
        prompt_token_ids = render_for_completion(messages)
        engine_input = tokens_input(prompt_token_ids, cache_salt=request.cache_salt)
        engine_input["arrival_time"] = arrival_time

        return messages, [engine_input]

    async def _initialize_tool_sessions(
        self,
        request: ResponsesRequest,
        context: ConversationContext,
        exit_stack: AsyncExitStack,
    ):
        # we should only initialize the tool session if the request needs tools
        if len(request.tools) == 0:
            return
        mcp_tools = {
            tool.server_label: tool for tool in request.tools if tool.type == "mcp"
        }
        await context.init_tool_sessions(
            self.tool_server, exit_stack, request.request_id, mcp_tools
        )

    async def responses_full_generator(
        self,
        request: ResponsesRequest,
        sampling_params: SamplingParams,
        result_generator: AsyncIterator[ConversationContext],
        context: ConversationContext,
        model_name: str,
        tokenizer: TokenizerLike,
        request_metadata: RequestResponseMetadata,
        created_time: int | None = None,
        session_token_state: _SessionTokenState | None = None,
    ) -> ErrorResponse | ResponsesResponse:
        if created_time is None:
            created_time = int(time.time())

        async with AsyncExitStack() as exit_stack:
            try:
                await self._initialize_tool_sessions(request, context, exit_stack)
                async for _ in result_generator:
                    pass
            except asyncio.CancelledError:
                return self.create_error_response("Client disconnected")

        # NOTE: Implementation of status is still WIP, but for now
        # we guarantee that if the status is not "completed", it is accurate.
        # "completed" is implemented as the "catch-all" for now.
        status: ResponseStatus = "completed"

        input_messages: ResponseInputOutputMessage | None = None
        output_messages: ResponseInputOutputMessage | None = None
        if self.use_harmony:
            assert isinstance(context, HarmonyContext)
            output = []
            harmony_msgs = context.messages[context.num_init_messages :]
            if harmony_msgs:
                fn_names = context.function_tool_names
                for msg in harmony_msgs[:-1]:
                    output.extend(harmony_to_response_output(msg, fn_names))
                output.extend(
                    harmony_to_response_output(
                        harmony_msgs[-1],
                        fn_names,
                        incomplete=context.last_append_flush_status,
                    )
                )

            if request.enable_response_messages:
                input_messages = context.messages[: context.num_init_messages]
                output_messages = context.messages[context.num_init_messages :]
            num_tool_output_tokens = context.num_tool_output_tokens
            if len(output) > 0:
                if context.finish_reason == "length":
                    status = "incomplete"
                elif context.finish_reason == "abort":
                    status = "cancelled"
                else:
                    self._raise_if_error(context.finish_reason, request.request_id)
            else:
                status = "incomplete"
        elif isinstance(context, ParsableContext):
            output = context.make_response_output_items()

            if request.enable_response_messages:
                input_messages = context.input_messages
                output_messages = context.output_messages

            # TODO: Calculate usage.
            # assert final_res.prompt_token_ids is not None
            num_tool_output_tokens = 0

            # Check finish reason from the parser
            if context.finish_reason == "length":
                status = "incomplete"
        else:
            assert isinstance(context, SimpleContext)
            # Use final_output which has accumulated text/token_ids/logprobs
            final_res = context.final_output
            assert final_res is not None
            assert len(final_res.outputs) == 1
            final_output = final_res.outputs[0]

            # finish_reason='error' indicates retryable internal error
            self._raise_if_error(final_output.finish_reason, request.request_id)

            # Check if generation was stopped due to max_tokens
            if final_output.finish_reason == "length":
                status = "incomplete"

            output = self._make_response_output_items(
                request,
                final_output,
                tokenizer,
                parser=context.response_parser,
            )

            if request.enable_response_messages:
                input_messages = context.input_messages
                output_messages = context.output_messages

            # Calculate usage.
            assert final_res.prompt_token_ids is not None
            num_tool_output_tokens = 0

        assert isinstance(context, (SimpleContext, HarmonyContext, ParsableContext))
        num_prompt_tokens = context.num_prompt_tokens
        num_generated_tokens = context.num_output_tokens
        num_cached_tokens = context.num_cached_tokens
        num_reasoning_tokens = context.num_reasoning_tokens
        # For text-based reasoning parsers (e.g., <think>...</think>),
        # HarmonyContext already counts reasoning tokens via channels.
        # For Simple/Parsable contexts, derive reasoning_tokens from
        # accumulated output token IDs using the parser if not already set.
        if (
            num_reasoning_tokens == 0
            and isinstance(context, (SimpleContext, ParsableContext))
            and context.response_parser is not None
            and context.response_parser.reasoning_parser is not None
        ):
            accumulated = getattr(context, "_accumulated_token_ids", []) or []
            num_reasoning_tokens = (
                context.response_parser.reasoning_parser.count_reasoning_tokens(
                    accumulated
                )
            )

        usage = ResponseUsage(
            input_tokens=num_prompt_tokens,
            output_tokens=num_generated_tokens,
            total_tokens=num_prompt_tokens + num_generated_tokens,
            input_tokens_details=InputTokensDetails(
                cached_tokens=num_cached_tokens,
                input_tokens_per_turn=[
                    turn.input_tokens for turn in context.all_turn_metrics
                ],
                cached_tokens_per_turn=[
                    turn.cached_input_tokens for turn in context.all_turn_metrics
                ],
            ),
            output_tokens_details=OutputTokensDetails(
                reasoning_tokens=num_reasoning_tokens,
                tool_output_tokens=num_tool_output_tokens,
                output_tokens_per_turn=[
                    turn.output_tokens for turn in context.all_turn_metrics
                ],
                tool_output_tokens_per_turn=[
                    turn.tool_output_tokens for turn in context.all_turn_metrics
                ],
            ),
        )
        response = ResponsesResponse.from_request(
            request,
            sampling_params,
            input_messages=input_messages,
            output_messages=output_messages,
            model_name=model_name,
            created_time=created_time,
            output=output,
            status=status,
            usage=usage,
            kv_transfer_params=context.kv_transfer_params,
        )

        if request.store:
            async with self.response_store_lock:
                stored_response = self.response_store.get(response.id)
                # If the response is already cancelled, don't update it.
                if stored_response is None or stored_response.status != "cancelled":
                    self.response_store[response.id] = response

        if session_token_state is not None and session_token_state.should_save:
            delta_token_ids = session_token_state.build_delta(
                self.renderer.get_eos_token_id()
            )

            # 每个完整请求只在此处调用一次 save()。
            await session_token_state.store.save(
                session_token_state.session_id,
                response.id,
                delta_token_ids,
            )
        return response

    def _topk_logprobs(
        self,
        logprobs: dict[int, SampleLogprob],
        top_logprobs: int,
        tokenizer: TokenizerLike,
    ) -> list[LogprobTopLogprob]:
        """Returns the top-k logprobs from the logprobs dictionary."""
        out = []
        for i, (token_id, _logprob) in enumerate(logprobs.items()):
            if i >= top_logprobs:
                break
            text = self._get_decoded_token(
                logprob=_logprob,
                token_id=token_id,
                tokenizer=tokenizer,
                return_as_token_id=self.return_tokens_as_token_ids,
            )
            out.append(
                LogprobTopLogprob(
                    token=text,
                    logprob=max(_logprob.logprob, -9999.0),
                    bytes=list(text.encode("utf-8", errors="replace")),
                )
            )
        return out

    def _create_response_logprobs(
        self,
        token_ids: Sequence[int],
        logprobs: SampleLogprobs | None,
        tokenizer: TokenizerLike,
        top_logprobs: int | None = None,
    ) -> list[Logprob]:
        assert logprobs is not None, "logprobs must be provided"
        assert len(token_ids) == len(logprobs), (
            "token_ids and logprobs.token_ids must have the same length"
        )
        out = []
        for i, token_id in enumerate(token_ids):
            logprob = logprobs[i]
            token_logprob = logprob[token_id]
            text = self._get_decoded_token(
                logprob=token_logprob,
                token_id=token_id,
                tokenizer=tokenizer,
                return_as_token_id=self.return_tokens_as_token_ids,
            )
            out.append(
                Logprob(
                    token=text,
                    logprob=max(token_logprob.logprob, -9999.0),
                    bytes=list(text.encode("utf-8", errors="replace")),
                    top_logprobs=(
                        self._topk_logprobs(
                            logprob, top_logprobs=top_logprobs, tokenizer=tokenizer
                        )
                        if top_logprobs
                        else []
                    ),
                )
            )
        return out

    def _create_stream_response_logprobs(
        self,
        token_ids: Sequence[int],
        logprobs: SampleLogprobs | None,
        tokenizer: TokenizerLike,
        top_logprobs: int | None = None,
    ) -> list[response_text_delta_event.Logprob]:
        lgs = self._create_response_logprobs(
            token_ids=token_ids,
            logprobs=logprobs,
            tokenizer=tokenizer,
            top_logprobs=top_logprobs,
        )
        return [
            response_text_delta_event.Logprob(
                token=lg.token,
                logprob=lg.logprob,
                top_logprobs=[
                    response_text_delta_event.LogprobTopLogprob(
                        token=tl.token, logprob=tl.logprob
                    )
                    for tl in lg.top_logprobs
                ],
            )
            for lg in lgs
        ]

    def _make_response_output_items(
        self,
        request: ResponsesRequest,
        final_output: CompletionOutput,
        tokenizer: TokenizerLike,
        parser: Parser | None = None,
    ) -> list[ResponseOutputItem]:
        # Log complete response if output logging is enabled
        if self.enable_log_outputs and self.request_logger:
            self.request_logger.log_outputs(
                request_id=request.request_id,
                outputs=final_output.text,
                output_token_ids=final_output.token_ids,
                finish_reason=final_output.finish_reason,
                is_streaming=False,
                delta=False,
            )

        # Compute logprobs if requested
        logprobs = None
        if request.is_include_output_logprobs() and final_output.logprobs:
            logprobs = self._create_response_logprobs(
                token_ids=final_output.token_ids,
                logprobs=final_output.logprobs,
                tokenizer=tokenizer,
                top_logprobs=request.top_logprobs,
            )

        # Use parser to extract reasoning, content, and tool calls
        if parser:
            reasoning, content, tool_calls = parser.parse(
                final_output.text,
                request,
                enable_auto_tools=self.enable_auto_tools,
                model_output_token_ids=final_output.token_ids,
            )
            return build_response_output_items(
                reasoning=reasoning,
                content=content,
                tool_calls=tool_calls,
                logprobs=logprobs,
                tools=request.tools,
            )

        # Fallback when no parser is configured
        return [
            ResponseOutputMessage(
                id=f"msg_{random_uuid()}",
                content=[
                    ResponseOutputText(
                        text=final_output.text,
                        annotations=[],
                        type="output_text",
                        logprobs=logprobs,
                    )
                ]
                if final_output.text
                else [],
                role="assistant",
                status="completed",
                type="message",
            )
        ]

    def _get_harmony_builtin_tool_descriptions(
        self, request: ResponsesRequest, tool_types: set[str]
    ) -> dict[str, str | None]:
        # Extract allowed_tools from MCP tool requests
        allowed_tools_map = _extract_allowed_tools_from_mcp_requests(request.tools)

        # Get filtered tool descriptions first.
        # If get_tool_description returns None (due to filtering), the tool is disabled.
        browser_description = (
            self.tool_server.get_tool_description(
                "browser", allowed_tools_map.get("web_search_preview")
            )
            if "web_search_preview" in tool_types
            and self.tool_server is not None
            and self.tool_server.has_tool("browser")
            else None
        )
        python_description = (
            self.tool_server.get_tool_description(
                "python", allowed_tools_map.get("code_interpreter")
            )
            if "code_interpreter" in tool_types
            and self.tool_server is not None
            and self.tool_server.has_tool("python")
            else None
        )
        container_description = (
            self.tool_server.get_tool_description(
                "container", allowed_tools_map.get("container")
            )
            if "container" in tool_types
            and self.tool_server is not None
            and self.tool_server.has_tool("container")
            else None
        )
        return {
            "browser_description": browser_description,
            "python_description": python_description,
            "container_description": container_description,
        }

    def _construct_input_messages_with_harmony(
        self,
        request: ResponsesRequest,
    ) -> list[OpenAIHarmonyMessage]:
        messages: list[OpenAIHarmonyMessage] = []
        request_input = request.input

        tool_types = extract_tool_types(request.tools)
        with_custom_tools = has_custom_tools(tool_types)
        instructions = request.instructions
        if instructions is None and isinstance(request_input, list):
            instructions, request_input = extract_instructions_from_messages(
                request_input
             )
        tool_descriptions = self._get_harmony_builtin_tool_descriptions(
            request, tool_types
        )
        tools = request.tools if with_custom_tools else None
        messages.extend(
            build_harmony_preamble(
                instructions=instructions,
                tools=tools,
                reasoning_effort=(
                    request.reasoning.effort if request.reasoning else None
                ),
                with_custom_tools=with_custom_tools,
                **tool_descriptions,
            )
        )
        messages += construct_harmony_previous_input_messages(request)

        # Append the new input.
        # Responses API supports simple text inputs without chat format.
        if isinstance(request_input, str):
            # Skip empty string input when previous_input_messages supplies
            # the full conversation history --- an empty trailing user message
            # confuses the model into thinking nothing was sent.
            if request_input or not request.previous_input_messages:
                messages.append(get_user_message(request_input))
        else:
            if prev_response is not None:
                prev_outputs = copy(prev_response.output)
            else:
                prev_outputs = []
            for response_msg in request_input:
                new_msg = response_input_to_harmony(response_msg, prev_outputs)
                if new_msg is not None:
                    messages.append(new_msg)

                # User passes in a tool call request and its output. We need
                # to add the tool call request to prev_outputs so that
                # response_input_to_harmony can find the tool call request when
                # parsing the tool call output.
                if isinstance(response_msg, ResponseFunctionToolCall):
                    prev_outputs.append(response_msg)
        return messages

    async def _run_background_request_stream(
        self,
        request: ResponsesRequest,
        *args,
        **kwargs,
    ):
        event_deque: deque[StreamingResponsesResponse] = deque()
        new_event_signal = asyncio.Event()
        self.event_store[request.request_id] = (event_deque, new_event_signal)
        generator = self.responses_stream_generator(request, *args, **kwargs)
        try:
            async for event in generator:
                event_deque.append(event)
                new_event_signal.set()  # Signal new event available
        finally:
            new_event_signal.set()

    async def _run_background_request(
        self,
        request: ResponsesRequest,
        *args,
        **kwargs,
    ):
        response = await self.responses_full_generator(request, *args, **kwargs)

        if isinstance(response, ErrorResponse):
            # If the request has failed, update the status to "failed".
            response_id = request.request_id
            async with self.response_store_lock:
                stored_response = self.response_store.get(response_id)
                assert stored_response is not None
                if stored_response.status not in ("completed", "cancelled"):
                    stored_response.status = "failed"

    async def responses_background_stream_generator(
        self,
        response_id: str,
        starting_after: int | None = None,
    ) -> AsyncGenerator[StreamingResponsesResponse, None]:
        if response_id not in self.event_store:
            raise VLLMValidationError(
                f"Unknown response_id: {response_id}",
                parameter="response_id",
                value=response_id,
            )

        event_deque, new_event_signal = self.event_store[response_id]
        start_index = 0 if starting_after is None else starting_after + 1
        current_index = start_index

        while True:
            new_event_signal.clear()

            # Yield existing events from start_index
            while current_index < len(event_deque):
                event = event_deque[current_index]
                yield event
                if getattr(event, "type", "unknown") == "response.completed":
                    return
                current_index += 1

            await new_event_signal.wait()

    async def retrieve_responses(
        self,
        response_id: str,
        starting_after: int | None,
        stream: bool | None,
    ) -> (
        ErrorResponse
        | ResponsesResponse
        | AsyncGenerator[StreamingResponsesResponse, None]
    ):
        async with self.response_store_lock:
            response = self.response_store.get(response_id)

        if response is None:
            return self._make_not_found_error(response_id)

        if stream:
            return self.responses_background_stream_generator(
                response_id,
                starting_after,
            )
        return response

    async def cancel_responses(
        self,
        response_id: str,
    ) -> ErrorResponse | ResponsesResponse:
        async with self.response_store_lock:
            response = self.response_store.get(response_id)
            if response is None:
                return self._make_not_found_error(response_id)

            prev_status = response.status
            if prev_status not in ("queued", "in_progress"):
                return self.create_error_response(
                    err_type="invalid_request_error",
                    message="Cannot cancel a synchronous response.",
                    param="response_id",
                )

            # Update the status to "cancelled".
            response.status = "cancelled"

        # Abort the request.
        if task := self.background_tasks.get(response_id):
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                logger.exception("Background task for %s was cancelled", response_id)
        return response

    def _make_not_found_error(self, response_id: str) -> ErrorResponse:
        return self.create_error_response(
            err_type="invalid_request_error",
            message=f"Response with id '{response_id}' not found.",
            status_code=HTTPStatus.NOT_FOUND,
            param="response_id",
        )

    async def _process_simple_streaming_events(
        self,
        request: ResponsesRequest,
        sampling_params: SamplingParams,
        result_generator: AsyncIterator[ConversationContext | None],
        context: ConversationContext,
        model_name: str,
        tokenizer: TokenizerLike,
        request_metadata: RequestResponseMetadata,
        created_time: int,
        _increment_sequence_number_and_return: Callable[
            [StreamingResponsesResponse], StreamingResponsesResponse
        ],
    ) -> AsyncGenerator[StreamingResponsesResponse, None]:
        processor = SimpleStreamingEventProcessor(tools=request.tools)

        def _get_logprobs(
            output: CompletionOutput,
        ) -> list[response_text_delta_event.Logprob]:
            if not request.is_include_output_logprobs():
                return []
            return self._create_stream_response_logprobs(
                token_ids=output.token_ids,
                logprobs=output.logprobs,
                tokenizer=tokenizer,
                top_logprobs=request.top_logprobs,
            )

        async for ctx in result_generator:
            assert isinstance(ctx, SimpleContext)
            if ctx.last_output is None or not ctx.last_output.outputs:
                continue

            output = ctx.last_output.outputs[0]
            self._raise_if_error(output.finish_reason, request.request_id)
            delta_text = output.text
            delta_token_ids = as_list(output.token_ids)

            if ctx.response_parser:
                delta_message = ctx.response_parser.parse_delta(
                    delta_text=delta_text,
                    delta_token_ids=delta_token_ids,
                    request=request,
                    prompt_token_ids=ctx.last_output.prompt_token_ids,
                    finished=output.finish_reason is not None,
                )
            else:
                delta_message = DeltaMessage(content=output.text)

            if not delta_message:
                continue

            for dm in split_delta(delta_message):
                target_state, tool_call = processor.resolve_target_state(dm)
                if target_state == _StateType.NONE:
                    continue

                if processor.needs_transition(target_state, tool_call):
                    for event in processor.close_current():
                        yield _increment_sequence_number_and_return(event)
                    for event in processor.open(target_state, tool_call):
                        yield _increment_sequence_number_and_return(event)

                for event in processor.emit_delta(dm, output, _get_logprobs):
                    yield _increment_sequence_number_and_return(event)

        for event in processor.close_current():
            yield _increment_sequence_number_and_return(event)

    async def _process_harmony_streaming_events(
        self,
        request: ResponsesRequest,
        sampling_params: SamplingParams,
        result_generator: AsyncIterator[ConversationContext | None],
        context: ConversationContext,
        model_name: str,
        tokenizer: TokenizerLike,
        request_metadata: RequestResponseMetadata,
        created_time: int,
        _increment_sequence_number_and_return: Callable[
            [StreamingResponsesResponse], StreamingResponsesResponse
        ],
    ) -> AsyncGenerator[StreamingResponsesResponse, None]:
        state = StreamingState()

        async for ctx in result_generator:
            assert isinstance(ctx, HarmonyContext)

            # finish_reason='error' indicates a retryable error
            self._raise_if_error(ctx.finish_reason, request.request_id)

            for segment in ctx.last_append_segments:
                if segment.delta:
                    for event in emit_content_delta_events(
                        segment, state, ctx.function_tool_names
                    ):
                        yield _increment_sequence_number_and_return(event)

                elif completed_message := segment.completed_message:
                    # TODO: Fix browser emitted as MCP calls
                    for event in emit_previous_item_done_events(
                        completed_message, state, ctx.function_tool_names
                    ):
                        yield _increment_sequence_number_and_return(event)

                    for event in emit_tool_action_events(
                        completed_message, state, self.tool_server
                    ):
                        yield _increment_sequence_number_and_return(event)
                    state.reset_for_new_item()

    async def responses_stream_generator(
        self,
        request: ResponsesRequest,
        sampling_params: SamplingParams,
        result_generator: AsyncIterator[ConversationContext | None],
        context: ConversationContext,
        model_name: str,
        tokenizer: TokenizerLike,
        request_metadata: RequestResponseMetadata,
        created_time: int | None = None,
        session_token_state: _SessionTokenState | None = None,
    ) -> AsyncGenerator[StreamingResponsesResponse, None]:
        # TODO:
        # 1. Handle disconnect

        created_time = created_time or int(time.time())

        sequence_number = 0

        def _increment_sequence_number_and_return(
            event: StreamingResponsesResponse,
        ) -> StreamingResponsesResponse:
            nonlocal sequence_number
            # Set sequence_number if the event has this attribute
            if hasattr(event, "sequence_number"):
                event.sequence_number = sequence_number
            sequence_number += 1
            return event

        async with AsyncExitStack() as exit_stack:
            if self.use_harmony:
                # TODO: in streaming, we noticed this bug:
                # https://github.com/vllm-project/vllm/issues/25697
                await self._initialize_tool_sessions(request, context, exit_stack)
                processor = self._process_harmony_streaming_events
            else:
                processor = self._process_simple_streaming_events
            # TODO Hanchen make sampling params to include the structural tag

            initial_response = ResponsesResponse.from_request(
                request,
                sampling_params,
                model_name=model_name,
                created_time=created_time,
                output=[],
                status="in_progress",
                usage=None,
            ).model_dump(mode="json", by_alias=True)
            yield _increment_sequence_number_and_return(
                ResponseCreatedEvent(
                    type="response.created",
                    sequence_number=-1,
                    response=initial_response,
                )
            )
            yield _increment_sequence_number_and_return(
                ResponseInProgressEvent(
                    type="response.in_progress",
                    sequence_number=-1,
                    response=initial_response,
                )
            )

            try:
                async for event_data in processor(
                    request,
                    sampling_params,
                    result_generator,
                    context,
                    model_name,
                    tokenizer,
                    request_metadata,
                    created_time,
                    _increment_sequence_number_and_return,
                ):
                    yield event_data
            except GenerationError as e:
                error_json = self._convert_generation_error_to_streaming_response(e)
                yield _increment_sequence_number_and_return(
                    TypeAdapter(StreamingResponsesResponse).validate_json(error_json)
                )
                return

            async def empty_async_generator():
                # A hack to trick Python to think this is a generator but
                # in fact it immediately returns.
                if False:
                    yield

            final_response = await self.responses_full_generator(
                request,
                sampling_params,
                empty_async_generator(),
                context,
                model_name,
                tokenizer,
                request_metadata,
                created_time=created_time,
                session_token_state=session_token_state,
            )
            yield _increment_sequence_number_and_return(
                ResponseCompletedEvent(
                    type="response.completed",
                    sequence_number=-1,
                    response=final_response,
                )
            )
    async def delete_response_session(self, session_id: str) -> bool:
        if self.store_service is None:
            return False
        store: TieredSessionStore = self.store_service.store
        return await store.delete(session_id)

    async def get_response_session(self, session_id: str) -> bool:
        if self.store_service is None:
            return False
        store: TieredSessionStore = self.store_service.store
        return await store.exists(session_id)
