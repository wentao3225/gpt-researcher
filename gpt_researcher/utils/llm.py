"""GPT Researcher 的 LLM 工具。

此模块提供通过统一接口与各类 LLM 提供者交互的工具函数。
"""
from __future__ import annotations

import logging
import os
from typing import Any
import asyncio

from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate

from gpt_researcher.llm_provider.generic.base import (
    SUPPORT_REASONING_EFFORT_MODELS,
    ReasoningEfforts,
    supports_temperature,
)

from ..prompts import PromptFamily
from .costs import calculate_llm_cost
from .validators import Subtopics


def get_llm(llm_provider: str, **kwargs):
    """获取 LLM 提供者实例。

    参数：
        llm_provider: LLM 提供者名称（如 'openai'、'anthropic'）。
        **kwargs: 传递给提供者的额外关键字参数。

    返回：
        针对指定提供者配置的 GenericLLMProvider 实例。
    """
    from gpt_researcher.llm_provider import GenericLLMProvider
    return GenericLLMProvider.from_provider(llm_provider, **kwargs)


async def create_chat_completion(
        messages: list[dict[str, str]],
        model: str | None = None,
        temperature: float | None = 0.4,
        max_tokens: int | None = 4000,
        llm_provider: str | None = None,
        stream: bool = False,
        websocket: Any | None = None,
        llm_kwargs: dict[str, Any] | None = None,
        cost_callback: callable = None,
        reasoning_effort: str | None = ReasoningEfforts.Medium.value,
        **kwargs
) -> str:
    """使用 OpenAI API 创建聊天补全
    参数：
        messages (list[dict[str, str]]): 要发送给聊天补全的消息。
        model (str, 可选): 要使用的模型。默认为 None。
        temperature (float, 可选): 要使用的温度。默认为 0.4。
        max_tokens (int, 可选): 要使用的最大 token 数。默认为 4000。
        llm_provider (str, 可选): 要使用的 LLM 提供者。
        stream (bool): 是否流式返回响应。默认为 False。
        websocket (WebSocket): 当前请求使用的 websocket。
        llm_kwargs (dict[str, Any], 可选): 额外的 LLM 关键字参数。默认为 None。
        cost_callback: 用于更新成本的回调函数。
        reasoning_effort (str, 可选): OpenAI 推理模型的推理力度。默认为 'low'。
        **kwargs: 额外的关键字参数。
    返回：
        str: 聊天补全的响应。
    """
    # 校验输入
    if model is None:
        raise ValueError("Model cannot be None")
    # 对荒谬值（如环境变量拼写错误）做合理性防护。各模型实际的
    # 输出上限由上游提供者强制执行。
    if max_tokens is not None and max_tokens > 200_000:
        raise ValueError(
            f"max_tokens={max_tokens} exceeds the largest output limit of "
            "any currently available model (128k as of late 2025). "
            "Check your FAST_TOKEN_LIMIT / SMART_TOKEN_LIMIT / "
            "STRATEGIC_TOKEN_LIMIT env vars for typos."
        )

    # 从支持的提供者中获取提供者
    provider_kwargs = {'model': model}

    if model in SUPPORT_REASONING_EFFORT_MODELS:
        provider_kwargs['reasoning_effort'] = reasoning_effort

    if supports_temperature(model):
        provider_kwargs['temperature'] = temperature
    else:
        # 这些模型强制使用其默认温度，但输出上限仍然适用
        # （langchain-openai 将 max_tokens 映射到 API 的
        # max_completion_tokens）。注意对推理模型而言该上限
        # 也涵盖推理 token，因此预算需要额外余量。
        provider_kwargs['temperature'] = None
    provider_kwargs['max_tokens'] = max_tokens

    # 调用方/环境变量覆盖优先于上面计算的默认值。特意放在最后应用：在此之前，
    # 无条件的 ``provider_kwargs['temperature'] = ...`` 会覆盖通过 ``llm_kwargs``
    # 或 ``LLM_KWARGS``（文档中说明的逃生通道）传入的温度，导致只接受唯一一个
    # 温度值的提供者无法使用（Moonshot kimi-k3 对 0.35 会返回 400
    # "only 1 is allowed for this model"）。
    if llm_kwargs:
        provider_kwargs.update(llm_kwargs)
    elif os.environ.get("LLM_KWARGS"):
        import json
        try:
            provider_kwargs.update(json.loads(os.environ["LLM_KWARGS"]))
        except json.JSONDecodeError:
            pass

    if llm_provider == "openai":
        base_url = os.environ.get("OPENAI_BASE_URL", None)
        if base_url:
            provider_kwargs['openai_api_base'] = base_url

    provider = get_llm(llm_provider, **provider_kwargs)
    response = ""
    # 创建响应
    max_attempts = 1 if (stream and websocket is not None) else 10
    last_exception: Exception | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            response = await provider.get_chat_response(
                messages, stream, websocket, **kwargs
            )
        except Exception as exc:
            last_exception = exc
            logging.getLogger(__name__).warning(
                f"LLM request failed (attempt {attempt}/{max_attempts}): {exc}"
            )
            if attempt < max_attempts:
                await asyncio.sleep(min(2 ** (attempt - 1), 8))
                continue
            break

        # 提供者可能把预算花在推理上后返回空内容的已完成请求，因此每次调用都记账。
        if cost_callback:
            llm_costs = calculate_llm_cost(
                llm_provider=llm_provider,
                model=model,
                input_content=str(messages),
                output_content=response,
                response_metadata=provider.last_response_metadata,
                usage_metadata=provider.last_usage_metadata,
                request_options=provider_kwargs,
            )
            cost_callback(llm_costs)

        if not response:
            last_exception = RuntimeError("Empty response from LLM provider")
            logging.getLogger(__name__).warning(
                f"LLM returned empty response (attempt {attempt}/{max_attempts})"
            )
            if attempt < max_attempts:
                await asyncio.sleep(min(2 ** (attempt - 1), 8))
                continue
            break

        return response

    logging.error(f"Failed to get response from {llm_provider} API")
    raise RuntimeError(f"Failed to get response from {llm_provider} API") from last_exception


async def construct_subtopics(
    task: str,
    data: str,
    config,
    subtopics: list = [],
    prompt_family: type[PromptFamily] | PromptFamily = PromptFamily,
    **kwargs
) -> list:
    """
    根据给定的任务和数据构建子主题。

    参数：
        task (str): 主任务或主题。
        data (str): 用于上下文的附加数据。
        config: 配置项。
        subtopics (list, 可选): 已有的子主题。默认为 []。
        prompt_family (PromptFamily): 提示词家族
        **kwargs: 额外的关键字参数。

    返回：
        list: 构建出的子主题列表。
    """
    try:
        parser = PydanticOutputParser(pydantic_object=Subtopics)

        prompt = PromptTemplate(
            template=prompt_family.generate_subtopics_prompt(),
            input_variables=["task", "data", "subtopics", "max_subtopics"],
            partial_variables={
                "format_instructions": parser.get_format_instructions()},
        )

        provider_kwargs = {'model': config.smart_llm_model}

        if config.llm_kwargs:
            provider_kwargs.update(config.llm_kwargs)

        if config.smart_llm_model in SUPPORT_REASONING_EFFORT_MODELS:
            provider_kwargs['reasoning_effort'] = ReasoningEfforts.High.value
        else:
            provider_kwargs['temperature'] = config.temperature
        provider_kwargs['max_tokens'] = config.smart_token_limit

        provider = get_llm(config.smart_llm_provider, **provider_kwargs)

        model = provider.llm

        chain = prompt | model | parser

        output = await chain.ainvoke({
            "task": task,
            "data": data,
            "subtopics": subtopics,
            "max_subtopics": config.max_subtopics
        }, **kwargs)

        return output

    except Exception as e:
        logging.getLogger(__name__).error(
            "Exception in parsing subtopics: %s", e, exc_info=True
        )
        return subtopics
