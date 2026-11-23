"""MiniMax 端点与模型元数据。"""

from __future__ import annotations

import os
from typing import Literal


MINIMAX_ENDPOINTS = {
    "global_en": {
        "openai": "https://api.minimax.io/v1",
        "anthropic": "https://api.minimax.io/anthropic",
    },
    "cn_zh": {
        "openai": "https://api.minimaxi.com/v1",
        "anthropic": "https://api.minimaxi.com/anthropic",
    },
}

MINIMAX_DEFAULT_MODEL = "MiniMax-M3.1-Flash"

# 仅推荐模型；显式的旧版和自定义模型 ID 仍然有效。
MINIMAX_MODEL_SPECS = {
    MINIMAX_DEFAULT_MODEL: {
        "context_window": 1_000_000,
        "max_output_tokens": 524_288,
        "thinking": ("adaptive",),
        "reasoning_effort": ("low", "medium", "high", "xhigh", "max"),
        "pricing_usd_per_million_tokens": {
            "input": 0.15,
            "output": 0.60,
            "cache_read": 0.015,
            "cache_write": None,
        },
    },
    "MiniMax-M3": {
        "context_window": 1_000_000,
        "input_modalities": ("text", "image", "video"),
        "thinking": ("adaptive", "disabled"),
        "pricing_usd_per_million_tokens": {
            "input": 0.3,
            "output": 1.2,
            "cache_read": 0.06,
            "cache_write": None,
        },
        "pricing_above_512k_usd_per_million_tokens": {
            "input": 0.6,
            "output": 2.4,
            "cache_read": 0.12,
            "cache_write": None,
        },
    },
}


def get_minimax_base_url(protocol: Literal["openai", "anthropic"]) -> str:
    """为请求的协议解析区域 MiniMax 端点。"""
    env_name = (
        "MINIMAX_BASE_URL"
        if protocol == "openai"
        else "MINIMAX_ANTHROPIC_BASE_URL"
    )
    region = os.getenv("MINIMAX_REGION", "global_en")
    base_url = os.getenv(env_name) or MINIMAX_ENDPOINTS.get(region, {}).get(protocol)
    if not base_url:
        supported_regions = ", ".join(sorted(MINIMAX_ENDPOINTS))
        raise ValueError(
            f"Unsupported MINIMAX_REGION={region!r}; choose one of {supported_regions}"
        )

    base_url = base_url.rstrip("/")
    required_suffix = "/v1" if protocol == "openai" else "/anthropic"
    if not base_url.endswith(required_suffix):
        raise ValueError(
            f"{env_name} must end with {required_suffix}; received {base_url!r}"
        )
    return base_url
