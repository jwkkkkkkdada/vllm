# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
from collections.abc import Iterable
from typing import Any

from openai.types.chat import (
    ChatCompletionAssistantMessageParam,
    ChatCompletionMessageToolCallParam,
    ChatCompletionToolMessageParam,
)
from openai.types.chat.chat_completion_message_tool_call_param import (
    Function as FunctionCallTool,
)
from openai.types.responses import (
    ResponseCustomToolCall,
    ResponseCustomToolCallOutputItem,
    ResponseFunctionToolCall,
    ResponseOutputItem,
    ResponseOutputMessage,
    ResponseOutputText,
    ResponseReasoningItem,
)
from openai.types.responses.response import ToolChoice
from openai.types.responses.response_function_tool_call_output_item import (
    ResponseFunctionToolCallOutputItem,
)
from openai.types.responses.response_output_text import Logprob
from openai.types.responses.response_reasoning_item import (
    Content as ResponseReasoningTextContent,
)
from openai.types.responses.tool import Tool

from vllm import envs
from vllm.entrypoints.chat_utils import make_tool_call_id
from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionMessageParam,
)
from vllm.entrypoints.openai.engine.protocol import FunctionCall
from vllm.entrypoints.openai.responses.protocol import ResponseInputOutputItem
from vllm.logger import init_logger
from vllm.tool_parsers.utils import (
    build_responses_tool_call_name_map,
    flat_namespace_tool_name,
    iter_response_function_tool_dicts,
    resolve_responses_tool_call_name,
)
from vllm.utils import random_uuid

logger = init_logger(__name__)


def build_response_output_items(
    reasoning: str | None,
    content: str | None,
    tool_calls: list[FunctionCall] | None,
    logprobs: list[Logprob] | None = None,
    tools: list[Tool] | None = None,
) -> list[ResponseOutputItem]:
    outputs: list[ResponseOutputItem] = []
    tool_call_name_map = build_responses_tool_call_name_map(tools)
    custom_tool_names = extract_custom_tool_names(tools)

    if reasoning:
        outputs.append(
            ResponseReasoningItem(
                id=f"rs_{random_uuid()}",
                summary=[],
                type="reasoning",
                content=[
                    ResponseReasoningTextContent(text=reasoning, type="reasoning_text")
                ],
                status=None,
            )
        )

    if content:
        outputs.append(
            ResponseOutputMessage(
                id=f"msg_{random_uuid()}",
                content=[
                    ResponseOutputText(
                        text=content,
                        annotations=[],
                        type="output_text",
                        logprobs=logprobs,
                    )
                ],
                role="assistant",
                status="completed",
                type="message",
            )
        )

    if tool_calls:
        for idx, tool_call in enumerate(tool_calls):
            call_name = resolve_responses_tool_call_name(
                tool_call.name, tool_call_name_map=tool_call_name_map
            )
            if tool_call.name in custom_tool_names:
                # 该工具原声明为 custom（自由文本）。输入侧把自由文本封装为
                # {"input": <original>} (JSON)，此处对称解包还原为原生 custom 形态。
                try:
                    raw_args = json.loads(tool_call.arguments)
                    input_raw = (
                        raw_args.get("input", tool_call.arguments)
                        if isinstance(raw_args, dict)
                        else raw_args
                    )
                except (ValueError, TypeError):
                    input_raw = tool_call.arguments
                if not isinstance(input_raw, str):
                    input_raw = json.dumps(input_raw)
                outputs.append(
                    ResponseCustomToolCall(
                        id=f"ct_{random_uuid()}",
                        call_id=tool_call.id
                        or make_tool_call_id(func_name=tool_call.name, idx=idx),
                        type="custom_tool_call",
                        name=call_name.name,
                        namespace=call_name.namespace,
                        input=input_raw,
                    )
                )
            else:
                outputs.append(
                    ResponseFunctionToolCall(
                        id=f"fc_{random_uuid()}",
                        call_id=tool_call.id
                        or make_tool_call_id(func_name=tool_call.name, idx=idx),
                        type="function_call",
                        status="completed",
                        name=call_name.name,
                        namespace=call_name.namespace,
                        arguments=tool_call.arguments,
                    )
                )

    return outputs


def should_continue_final_message(
    request_input: str | list[ResponseInputOutputItem],
) -> bool:
    """
    Determine if the last input message is a partial assistant message
    that should be continued rather than starting a new generation.

    This enables partial message completion similar to Anthropic's Messages API,
    where users can provide an incomplete assistant message and have the model
    continue from where it left off.

    A message is considered partial if:
    1. It's a ResponseOutputMessage or ResponseReasoningItem
    2. Its status is "in_progress" or "incomplete"

    Args:
        request_input: The input to the Responses API request

    Returns:
        True if the final message should be continued, False otherwise
    """
    if isinstance(request_input, str):
        # Simple string input is always a user message
        return False

    if not request_input:
        return False

    last_item = request_input[-1]

    # Check if the last item is a partial assistant message
    if isinstance(last_item, ResponseOutputMessage):
        return last_item.status in ("in_progress", "incomplete")

    # Check if the last item is a partial reasoning item
    if isinstance(last_item, ResponseReasoningItem):
        return last_item.status in ("in_progress", "incomplete")

    if isinstance(last_item, dict):
        # only support partial completion for messages for now
        if last_item.get("type", "message") not in ("message", "reasoning"):
            return False
        return last_item.get("status") in ("in_progress", "incomplete")

    return False


def construct_input_messages(
    *,
    request_instructions: str | None = None,
    request_input: str | list[ResponseInputOutputItem],
    prev_msg: list[ChatCompletionMessageParam] | None = None,
    prev_response_output: list[ResponseOutputItem] | None = None,
):
    messages: list[ChatCompletionMessageParam] = []
    if request_instructions:
        messages.append(
            {
                "role": "system",
                "content": request_instructions,
            }
        )

    # Prepend the conversation history.
    if prev_msg is not None:
        # Filter out system messages from previous conversation -- per the
        # OpenAI spec, instructions should NOT carry over across responses.
        # The current request's instructions (if any) were already added above.
        messages.extend(m for m in prev_msg if m.get("role") != "system")
    if prev_response_output is not None:
        # Add the previous output.
        for output_item in prev_response_output:
            # NOTE: We skip the reasoning output.
            if isinstance(output_item, ResponseOutputMessage):
                for content in output_item.content:
                    messages.append(
                        {
                            "role": "assistant",
                            "content": content.text,
                        }
                    )

    # Append the new input.
    # Responses API supports simple text inputs without chat format.
    if isinstance(request_input, str):
        messages.append({"role": "user", "content": request_input})
    else:
        input_messages = construct_chat_messages_with_tool_call(request_input)
        messages.extend(input_messages)
    return messages


def construct_chat_messages_with_tool_call(
    input_messages: list[ResponseInputOutputItem],
) -> list[ChatCompletionMessageParam]:
    """Build chat messages from response items.

    Some chat messages span multiple response items (e.g., reasoning + tool calls).
    """
    messages: list[ChatCompletionMessageParam] = []
    for item in input_messages:
        message = _construct_message_from_response_item(
            item, prev_msg=messages[-1] if messages else None
        )
        if message is not None:
            messages.append(message)

    return messages


def _construct_message_from_response_item(
    item: ResponseInputOutputItem,
    prev_msg: ChatCompletionMessageParam | None = None,
) -> ChatCompletionMessageParam | None:
    """
    Returns a new message or None. If `None`, `prev_msg` might be updated.
    If `prev_msg` is `None`, a new message is always returned.
    """
    prev_assistant_msg = (
        prev_msg if prev_msg and prev_msg.get("role") == "assistant" else None
    )

    tool_call: (
        ChatCompletionMessageToolCallParam
        | None
    ) = None
    if isinstance(item, ResponseFunctionToolCall):
        tool_name = item.name
        if item.namespace:
            tool_name = flat_namespace_tool_name(item.namespace, item.name)
        tool_call = ChatCompletionMessageToolCallParam(
            id=item.call_id,
            function=FunctionCallTool(
                name=tool_name,
                arguments=item.arguments,
            ),
            type="function",
        )
    elif isinstance(item, ResponseCustomToolCall) or (
        isinstance(item, dict) and item.get("type") == "custom_tool_call"
    ):
        # 兼容 Pydantic 模型（ResponseCustomToolCall）与 dict（...Param）两种形态。
        call_id = (
            item.call_id
            if isinstance(item, ResponseCustomToolCall)
            else item.get("call_id")
        )
        func_name = (
            item.name
            if isinstance(item, ResponseCustomToolCall)
            else item.get("name")
        )
        namespace = (
            item.namespace
            if isinstance(item, ResponseCustomToolCall)
            else item.get("namespace")
        )
        if namespace:
            func_name = flat_namespace_tool_name(namespace, func_name)
        func_input = (
            item.input
            if isinstance(item, ResponseCustomToolCall)
            else item.get("input")
        )
        # 下游 chat completions 管线仅支持 type="function" 的 assistant tool_call。
        # 把自由文本封装为 {"input": ...} 合法 JSON，保证下一轮 JSON 解析干净。
        tool_call = ChatCompletionMessageToolCallParam(
            id=call_id,
            function=FunctionCallTool(
                name=func_name,
                arguments=json.dumps({"input": func_input}),
            ),
            type="function",
        )

    # function 与 custom 共用合并入口：尝试并入前一条 assistant 消息。
    if tool_call is not None:
        if prev_assistant_msg:
            tool_calls = prev_assistant_msg.get("tool_calls")
            if tool_calls is None:
                prev_assistant_msg["tool_calls"] = [tool_call]
                return None
            if isinstance(tool_calls, list):
                tool_calls.append(tool_call)
                return None
            if isinstance(tool_calls, Iterable) and not isinstance(
                tool_calls, (dict, str)
            ):
                tool_calls = list(tool_calls)
                tool_calls.append(tool_call)
                prev_assistant_msg["tool_calls"] = tool_calls
                return None
            logger.warning(
                "Previous assistant message has unknown tool_calls format. "
                "Tool call merging is skipped and a new assistant message is created. "
                "Item %s",
                getattr(item, "id", None),
            )
        return ChatCompletionAssistantMessageParam(
            role="assistant",
            tool_calls=[tool_call],
        )
    if isinstance(item, ResponseReasoningItem):
        reasoning = ""
        if item.encrypted_content:
            raise ValueError("Encrypted content is not supported.")
        elif item.content and len(item.content) >= 1:
            reasoning = item.content[0].text
        elif len(item.summary) >= 1:
            reasoning = item.summary[0].text
            logger.warning(
                "Using summary text as reasoning content for item %s. "
                "Please use content instead of summary for "
                "reasoning items.",
                item.id,
            )

        if prev_assistant_msg:
            previous_reasoning = prev_assistant_msg.get("reasoning")
            if previous_reasoning is None:
                prev_assistant_msg["reasoning"] = reasoning
                return None
        return {
            "role": "assistant",
            "reasoning": reasoning,
        }
    elif isinstance(item, ResponseOutputMessage):
        output_text = item.content[0].text
        if prev_assistant_msg:
            previous_content = prev_assistant_msg.get("content")
            if previous_content is None:
                prev_assistant_msg["content"] = output_text
                return None
        return {
            "role": "assistant",
            "content": output_text,
        }
    elif isinstance(
        item,
        (ResponseFunctionToolCallOutputItem, ResponseCustomToolCallOutputItem),
    ):
        # custom_tool_call_output 也是 tool 消息，统一走工具结果分支。
        return ChatCompletionToolMessageParam(
            role="tool",
            content=item.output,
            tool_call_id=item.call_id,
        )
    elif isinstance(item, dict) and item.get("type") in (
        "function_call_output",
        "custom_tool_call_output",
    ):
        # dict/TypedDict 形态的工具输出：Pydantic 可能解析为
        # ResponseXxxToolCallOutputParam (dict) 而非对应 model。
        return ChatCompletionToolMessageParam(
            role="tool",
            content=item.get("output"),
            tool_call_id=item.get("call_id"),
        )
    elif isinstance(item, dict) and item.get("role") == "assistant":
        content = item.get("content")
        text: str | None = None
        if isinstance(content, str):
            text = content
        elif isinstance(content, list) and content:
            text = content[0].get("text")
        if text is not None:
            if prev_assistant_msg:
                previous_content = prev_assistant_msg.get("content")
                if previous_content is None:
                    prev_assistant_msg["content"] = text
                    return None
            return {"role": "assistant", "content": text}
    return item  # type: ignore[arg-type]


def extract_function_tool_names(tools: list[Tool]) -> frozenset[str]:
    names = []
    for tool in tools:
        if tool.type == "function":
            names.append(tool.name)
        elif tool.type == "namespace":
            names.extend(
                flat_namespace_tool_name(tool.name, namespaced_tool.name)
                for namespaced_tool in tool.tools
                if namespaced_tool.type == "function"
            )
    return frozenset(names)


def extract_custom_tool_names(tools: list[Tool] | None) -> frozenset[str]:
    """
    收集 custom（自由文本、无 JSON Schema）工具名（含 namespace 内嵌套的 custom）。
    这些工具的 Responses 输出项必须还原为 ResponseCustomToolCall，而非 function call。
    """
    names = []
    if tools:
        for tool in tools:
            if tool.type == "custom":
                names.append(tool.name)
            elif tool.type == "namespace":
                names.extend(
                    flat_namespace_tool_name(tool.name, namespaced_tool.name)
                    for namespaced_tool in tool.tools
                    if namespaced_tool.type == "custom"
                )
    return frozenset(names)


def extract_tool_types(tools: list[Tool]) -> set[str]:
    """
    Extracts the tool types from the given tools.
    """
    tool_types: set[str] = set()
    for tool in tools:
        if tool.type == "mcp":
            # Allow the MCP Tool type to enable built in tools if the
            # server_label is allowlisted in
            # envs.VLLM_GPT_OSS_SYSTEM_TOOL_MCP_LABELS
            if tool.server_label in envs.VLLM_GPT_OSS_SYSTEM_TOOL_MCP_LABELS:
                tool_types.add(tool.server_label)
        else:
            tool_types.add(tool.type)
    return tool_types


def convert_tool_responses_to_completions_format(
    tool: dict,
) -> dict[str, Any]:
    """
    Convert a flat Responses tool schema:
        {"type": "function", "name": "...", "description": "...", "parameters": {...}}
    into a Chat Completions tool param for chat-template rendering.
    """
    return {
        "type": "function",
        "function": tool,
    }


def convert_custom_tool_responses_to_completions_format(
    tool: Tool,
    name: str | None = None,
) -> dict[str, Any]:
    """将 custom（自由文本）Responses 工具转换为 Chat Completions function 工具，
    使模型能够发起调用。

    自由文本输入对外暴露为单个 JSON ''input'' 字段，既让共享函数式工具解析器
    可解析模型输出，也让输出侧能解包还原成原生 custom 形态。
    """
    return {
        "type": "function",
        "function": {
            "name": name or tool.name,
            "description": getattr(tool, "description", None)
            or "Custom tool (freeform input).",
            "parameters": {
                "type": "object",
                "properties": {
                    "input": {
                        "type": "string",
                        "description": "Freeform input for the custom tool.",
                    }
                },
                "required": ["input"],
            },
        },
    }


def construct_tool_dicts(
    tools: list[Tool],
    tool_choice: ToolChoice,
    exclude_tools_when_tool_choice_none: bool = False,
) -> list[dict[str, Any]] | None:
    if not tools or (tool_choice == "none" and exclude_tools_when_tool_choice_none):
        return None
    tool_dicts = [
        convert_tool_responses_to_completions_format(tool)
        for tool in iter_response_function_tool_dicts(tools)
    ]
    for tool in tools:
        if tool.type == "custom":
            tool_dicts.append(
                convert_custom_tool_responses_to_completions_format(tool)
            )
        elif tool.type == "namespace":
            for namespaced_tool in tool.tools:
                if namespaced_tool.type != "custom":
                    continue
                tool_dicts.append(
                    convert_custom_tool_responses_to_completions_format(
                        namespaced_tool,
                        name=flat_namespace_tool_name(
                            tool.name, namespaced_tool.name
                        ),
                    )
                )
    return tool_dicts
