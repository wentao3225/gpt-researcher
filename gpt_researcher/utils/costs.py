"""LLM API 使用的成本估算工具。"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any

import tiktoken

# 依据 OpenAI 定价页面：https://openai.com/api/pricing/
ENCODING_MODEL = "o200k_base"
INPUT_COST_PER_TOKEN = 0.000005
OUTPUT_COST_PER_TOKEN = 0.000015
IMAGE_INFERENCE_COST = 0.003825
EMBEDDING_COST = 0.02 / 1000000  # 假设使用新的 ada-3-small

# OpenAI 的自动提示词缓存对输入 token 中命中缓存的部分按标准输入
# 费率打折计费（目前 GPT-4o/4.1/5 系列统一为 50% 折扣）。由于 OpenAI
# 迄今对所有支持缓存的模型都采用统一的 50% 缓存折扣，这里保持为单一
# 常量；若未来模型引入不同比例，再重新审视。
# https://openai.com/api/pricing/
OPENAI_CACHED_INPUT_DISCOUNT = 0.5

logger = logging.getLogger(__name__)

# (模式, 输入 $/百万 token, 输出 $/百万 token) - 首个匹配生效，因此更
# 具体的模式（如 "-mini"）必须排在其基础模型之前。
# 价格截至 2026 年 11 月：https://developers.openai.com/api/docs/pricing
OPENAI_MODEL_PRICING = (
    (("gpt-6.1-sol",), 2.0, 10.0),
    (("gpt-6.1-terra",), 1.0, 5.0),
    (("gpt-6-astra",), 10.0, 50.0),
    (("gpt-6-sol",), 2.0, 10.0),
    (("gpt-6-luna",), 0.1, 0.5),
    (("gpt-5.6-sol",), 4.0, 20.0),
    (("gpt-5.6-terra",), 2.0, 12.0),
    (("gpt-5.6-luna",), 0.2, 1.2),
    (("gpt-5.5-pro",), 30.0, 180.0),
    (("gpt-5.5",), 5.0, 30.0),
    (("gpt-5.4-pro",), 30.0, 180.0),
    (("gpt-5.4-mini",), 0.75, 4.5),
    (("gpt-5.4-nano",), 0.2, 1.25),
    (("gpt-5.4",), 2.5, 15.0),
    (("gpt-5-mini",), 0.25, 2.0),
    (("gpt-5-nano",), 0.05, 0.4),
    (("gpt-5",), 1.25, 10.0),
    (("gpt-4.1-mini",), 0.4, 1.6),
    (("gpt-4.1-nano",), 0.1, 0.4),
    (("gpt-4.1",), 2.0, 8.0),
    (("gpt-4o-mini",), 0.15, 0.6),
    (("gpt-4o",), 2.5, 10.0),
    (("o4-mini",), 1.1, 4.4),
    (("o3-mini",), 1.1, 4.4),
    (("o3",), 2.0, 8.0),
)

ANTHROPIC_MODEL_PRICING = (
    (("claude-opus-4-7",), 5.0, 25.0),
    (("claude-opus-4-6",), 5.0, 25.0),
    (("claude-opus-4-5", "claude-4-opus"), 5.0, 25.0),
    (("claude-opus-4-1",), 15.0, 75.0),
    (("claude-opus-4",), 15.0, 75.0),
    (("claude-sonnet-4-6",), 3.0, 15.0),
    (("claude-sonnet-4-5", "claude-4-sonnet"), 3.0, 15.0),
    (("claude-sonnet-4",), 3.0, 15.0),
    (("claude-haiku-4-5",), 1.0, 5.0),
    (("claude-3-5-haiku",), 0.8, 4.0),
)

ANTHROPIC_US_INFERENCE_GEO_MODELS = (
    "claude-opus-4-7",
    "claude-opus-4-6",
    "claude-sonnet-4-6",
)


def estimate_llm_cost(input_content: str, output_content: str) -> float:
    """根据输入和输出内容估算 LLM API 调用的成本。

    成本估算基于 OpenAI 定价，其他模型可能有所不同。

    参数：
        input_content: 发送给 LLM 的输入文本。
        output_content: 从 LLM 接收的输出文本。

    返回：
        以美元计的估算成本。
    """
    # 回调和流式路径可能在内容可用之前传入 None。
    # tiktoken.encode(None) 会抛出 TypeError 并中断调用方。
    if input_content is None:
        input_content = ""
    if output_content is None:
        output_content = ""
    if not isinstance(input_content, str):
        input_content = str(input_content)
    if not isinstance(output_content, str):
        output_content = str(output_content)
    encoding = tiktoken.get_encoding(ENCODING_MODEL)
    input_tokens = encoding.encode(input_content)
    output_tokens = encoding.encode(output_content)
    input_costs = len(input_tokens) * INPUT_COST_PER_TOKEN
    output_costs = len(output_tokens) * OUTPUT_COST_PER_TOKEN
    return input_costs + output_costs


def _mapping_to_dict(value: Mapping[str, Any] | Any | None) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, Mapping):
        return dict(value)
    if hasattr(value, "model_dump"):
        return dict(value.model_dump())
    return {}


def _resolve_anthropic_model_name(
    model: str | None,
    response_metadata: Mapping[str, Any] | None = None,
) -> str:
    metadata = _mapping_to_dict(response_metadata)
    return str(
        metadata.get("model")
        or metadata.get("model_name")
        or model
        or ""
    ).lower()


def _coerce_token_count(value: Any) -> int:
    if value is None:
        return 0
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _extract_anthropic_usage(
    response_metadata: Mapping[str, Any] | None = None,
    usage_metadata: Mapping[str, Any] | Any | None = None,
) -> dict[str, int] | None:
    """提取 Anthropic 用量，包括提示词缓存 token 字段。

    Anthropic Messages 的用量可能包括：
    - input_tokens / output_tokens
    - cache_creation_input_tokens（按约 1.25 倍输入计费）
    - cache_read_input_tokens（按约 0.1 倍输入计费）
    """
    metadata = _mapping_to_dict(response_metadata)
    usage = _mapping_to_dict(metadata.get("usage"))

    def _from_usage(usage_dict: dict[str, Any]) -> dict[str, int] | None:
        input_tokens = usage_dict.get("input_tokens")
        output_tokens = usage_dict.get("output_tokens")
        if input_tokens is None or output_tokens is None:
            return None
        return {
            "input_tokens": int(input_tokens),
            "output_tokens": int(output_tokens),
            "cache_creation_input_tokens": _coerce_token_count(
                usage_dict.get("cache_creation_input_tokens")
            ),
            "cache_read_input_tokens": _coerce_token_count(
                usage_dict.get("cache_read_input_tokens")
            ),
        }

    if usage:
        parsed = _from_usage(usage)
        if parsed is not None:
            return parsed

    usage = _mapping_to_dict(usage_metadata)
    return _from_usage(usage)


def _get_anthropic_pricing(model_name: str) -> tuple[float, float] | None:
    normalized_model_name = model_name.lower()
    for patterns, input_price_per_mtok, output_price_per_mtok in ANTHROPIC_MODEL_PRICING:
        if any(pattern in normalized_model_name for pattern in patterns):
            return input_price_per_mtok, output_price_per_mtok
    return None


def _get_anthropic_pricing_multiplier(
    model_name: str,
    request_options: Mapping[str, Any] | None = None,
) -> float:
    if not request_options:
        return 1.0

    inference_geo = str(request_options.get("inference_geo", "")).lower()
    if inference_geo != "us":
        return 1.0

    if any(pattern in model_name for pattern in ANTHROPIC_US_INFERENCE_GEO_MODELS):
        return 1.1

    return 1.0


def calculate_anthropic_cost(
    model: str | None,
    response_metadata: Mapping[str, Any] | None = None,
    usage_metadata: Mapping[str, Any] | Any | None = None,
    request_options: Mapping[str, Any] | None = None,
) -> float | None:
    usage = _extract_anthropic_usage(response_metadata=response_metadata, usage_metadata=usage_metadata)
    if not usage:
        return None

    model_name = _resolve_anthropic_model_name(model=model, response_metadata=response_metadata)
    pricing = _get_anthropic_pricing(model_name)
    if pricing is None:
        logger.warning(
            "Missing Anthropic pricing rule for model '%s'; falling back to token estimator.",
            model_name or model,
        )
        return None

    input_price_per_mtok, output_price_per_mtok = pricing
    multiplier = _get_anthropic_pricing_multiplier(model_name, request_options=request_options)
    # Anthropic 提示词缓存：缓存写入约 1.25 倍输入，缓存读取约 0.1 倍输入。
    # input_tokens 仅指非缓存输入；缓存字段单独计费。
    cache_write_price_per_mtok = input_price_per_mtok * 1.25
    cache_read_price_per_mtok = input_price_per_mtok * 0.1
    input_cost = usage["input_tokens"] * input_price_per_mtok / 1_000_000
    cache_write_cost = (
        usage.get("cache_creation_input_tokens", 0) * cache_write_price_per_mtok / 1_000_000
    )
    cache_read_cost = (
        usage.get("cache_read_input_tokens", 0) * cache_read_price_per_mtok / 1_000_000
    )
    output_cost = usage["output_tokens"] * output_price_per_mtok / 1_000_000
    return (input_cost + cache_write_cost + cache_read_cost + output_cost) * multiplier


def _extract_usage_tokens(
    usage_metadata: Mapping[str, Any] | Any | None,
) -> tuple[int, int, int] | None:
    """返回 (input_tokens, output_tokens, cache_read_tokens)。

    cache_read_tokens 是 input_tokens 中命中提供者提示词缓存的部分
    （LangChain 标准化的 ``input_token_details.cache_read``），
    应按 OPENAI_CACHED_INPUT_DISCOUNT 而非标准输入费率计费。
    当响应未报告缓存明细时默认为 0，因此非缓存调用不受影响。
    """
    usage = _mapping_to_dict(usage_metadata)
    input_tokens = usage.get("input_tokens")
    output_tokens = usage.get("output_tokens")
    if input_tokens is None or output_tokens is None:
        return None

    input_token_details = _mapping_to_dict(usage.get("input_token_details"))
    cache_read_tokens = int(input_token_details.get("cache_read") or 0)

    return int(input_tokens), int(output_tokens), cache_read_tokens


def _get_openai_pricing(model: str | None) -> tuple[float, float] | None:
    normalized = (model or "").lower()
    for patterns, input_price_per_mtok, output_price_per_mtok in OPENAI_MODEL_PRICING:
        if any(pattern in normalized for pattern in patterns):
            return input_price_per_mtok, output_price_per_mtok
    return None


def calculate_llm_cost(
    llm_provider: str | None,
    model: str | None,
    input_content: str,
    output_content: str,
    response_metadata: Mapping[str, Any] | None = None,
    usage_metadata: Mapping[str, Any] | Any | None = None,
    request_options: Mapping[str, Any] | None = None,
) -> float:
    if llm_provider == "anthropic":
        anthropic_cost = calculate_anthropic_cost(
            model=model,
            response_metadata=response_metadata,
            usage_metadata=usage_metadata,
            request_options=request_options,
        )
        if anthropic_cost is not None:
            return anthropic_cost

    # 在序列化消息字典上，优先使用 API 报告的 token 用量而非 tiktoken 估算；
    # 后者会高估输入且完全遗漏推理 token。
    usage_tokens = _extract_usage_tokens(usage_metadata)
    if usage_tokens is not None:
        input_tokens, output_tokens, cache_read_tokens = usage_tokens
        # cache_read_tokens 是 input_tokens 的子集而非额外的 token ——
        # 将 input_tokens 拆分为非缓存和缓存两部分，使缓存部分按折扣
        # 费率而非完整输入费率计价。
        non_cached_input_tokens = input_tokens - cache_read_tokens
        pricing = _get_openai_pricing(model)
        if pricing is not None:
            input_price_per_mtok, output_price_per_mtok = pricing
            non_cached_cost = non_cached_input_tokens * input_price_per_mtok
            cached_cost = (
                cache_read_tokens
                * input_price_per_mtok
                * OPENAI_CACHED_INPUT_DISCOUNT
            )
            output_cost = output_tokens * output_price_per_mtok
            return (non_cached_cost + cached_cost + output_cost) / 1_000_000
        cached_cost = (
            cache_read_tokens * INPUT_COST_PER_TOKEN * OPENAI_CACHED_INPUT_DISCOUNT
        )
        non_cached_cost = non_cached_input_tokens * INPUT_COST_PER_TOKEN
        return (
            non_cached_cost
            + cached_cost
            + output_tokens * OUTPUT_COST_PER_TOKEN
        )

    return estimate_llm_cost(input_content, output_content)


def estimate_embedding_cost(model: str, docs: list) -> float:
    """估算文档嵌入的成本。

    参数：
        model: 嵌入模型名称。
        docs: 要嵌入的文档列表。

    返回：
        以美元计的估算嵌入成本。
    """
    if not docs:
        return 0.0

    if not model or not isinstance(model, str):
        encoding = tiktoken.get_encoding(ENCODING_MODEL)
    else:
        try:
            encoding = tiktoken.encoding_for_model(model)
        except (KeyError, ValueError):
            # tiktoken 只认识 OpenAI 模型名。非 OpenAI 嵌入提供者
            # （Ollama、Cohere、Nomic、HuggingFace 等）在此处抛出 KeyError，
            # 否则会中断研究过程中的成本跟踪。回退到默认的 OpenAI 编码
            # 以获得尽力而为的 token 估算。
            encoding = tiktoken.get_encoding(ENCODING_MODEL)

    if isinstance(docs, (str, bytes)) or not hasattr(docs, "__iter__"):
        docs = [docs]

    total_tokens = sum(len(encoding.encode(str(doc))) for doc in docs)
    return total_tokens * EMBEDDING_COST
