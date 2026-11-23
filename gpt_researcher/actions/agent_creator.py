"""GPT Researcher 的智能体创建与选择工具。

此模块提供根据查询类型自动选择和配置
合适研究智能体的函数。
"""

import json
import logging
import re

import json_repair

from ..prompts import PromptFamily
from ..utils.llm import create_chat_completion

logger = logging.getLogger(__name__)

async def choose_agent(
    query,
    cfg,
    parent_query=None,
    cost_callback: callable = None,
    headers=None,
    prompt_family: type[PromptFamily] | PromptFamily = PromptFamily,
    **kwargs
):
    """
    自动选择智能体
    参数：
        parent_query: 某些情况下研究是针对主查询的某个子主题进行的。
            父查询让智能体了解主上下文，以便更好地推理。
        query: 原始查询
        cfg: 配置
        cost_callback: 用于计算 LLM 成本的回调
        prompt_family: 提示词家族

    返回：
        agent: 智能体名称
        agent_role_prompt: 智能体角色提示词
    """
    query = f"{parent_query} - {query}" if parent_query else f"{query}"
    response = None  # 初始化 response，确保变量已定义

    try:
        response = await create_chat_completion(
            model=cfg.smart_llm_model,
            messages=[
                {"role": "system", "content": f"{prompt_family.auto_agent_instructions()}"},
                {"role": "user", "content": f"task: {query}"},
            ],
            temperature=0.15,
            llm_provider=cfg.smart_llm_provider,
            llm_kwargs=cfg.llm_kwargs,
            cost_callback=cost_callback,
            **kwargs
        )

        # 优先使用 json_repair，让带围栏/轻微损坏的 LLM JSON 在热路径上
        # 直接成功，而不是总是落入异常处理。
        try:
            agent_dict = json_repair.loads(response) if response is not None else None
        except Exception:
            agent_dict = None
        if not isinstance(agent_dict, dict):
            agent_dict = json.loads(response)
        if not isinstance(agent_dict, dict):
            raise ValueError("agent JSON was not an object")
        return agent_dict["server"], agent_dict["agent_role_prompt"]

    except Exception as e:
        return await handle_json_error(response)


async def handle_json_error(response: str | None):
    """处理 LLM 响应中的 JSON 解析错误。

    使用 json_repair 和正则提取作为回退手段，
    尝试从格式错误的 JSON 响应中恢复智能体信息。

    参数：
        response: 初始 JSON 解析失败的 LLM 响应字符串。

    返回：
        (agent_name, agent_role_prompt) 元组。所有解析尝试都失败时
        返回默认智能体。
    """
    try:
        agent_dict = json_repair.loads(response) if response is not None else None
        recovered = _agent_pair_from_payload(agent_dict)
        if recovered is not None:
            return recovered
    except Exception as e:
        error_type = type(e).__name__
        error_msg = str(e)
        logger.warning(
            f"Failed to parse agent JSON with json_repair: {error_type}: {error_msg}",
            exc_info=True
        )
        if response:
            logger.debug(f"LLM response that failed to parse: {response[:500]}...")

    json_string = extract_json_with_regex(response)
    if json_string:
        try:
            json_data = json.loads(json_string)
            recovered = _agent_pair_from_payload(json_data)
            if recovered is not None:
                return recovered
        except json.JSONDecodeError as e:
            logger.warning(
                f"Failed to decode JSON from regex extraction: {str(e)}",
                exc_info=True
            )

    logger.info("No valid JSON found in LLM response. Falling back to default agent.")
    return "Default Agent", (
        "You are an AI critical thinker research assistant. Your sole purpose is to write well written, "
        "critically acclaimed, objective and structured reports on given text."
    )


def _agent_pair_from_payload(payload):
    """仅对完整的字典载荷返回 (server, agent_role_prompt)。"""
    if not isinstance(payload, dict):
        return None
    server = payload.get("server")
    role = payload.get("agent_role_prompt")
    if server and role:
        return server, role
    return None


def extract_json_with_regex(response: str | None) -> str | None:
    """使用正则从字符串中提取 JSON 对象。

    尝试在响应字符串中找到第一个 JSON 对象模式。

    参数：
        response: 要搜索 JSON 内容的字符串。

    返回：
        找到时返回提取的 JSON 字符串，否则返回 None。
    """
    if not response:
        return None
    # 贪婪的 ``{.*}`` 使匹配从第一个 ``{`` 跨到响应中最后一个 ``}``，
    # 捕获整个对象。非贪婪的 ``{.*?}`` 会在第一个 ``}`` 处停止，把
    # 含多个键或字符串值中带 ``}`` 的对象（如提到 "{markets}" 的
    # agent_role_prompt）截断成无效 JSON。
    json_match = re.search(r"{.*}", response, re.DOTALL)
    if json_match:
        return json_match.group(0)
    return None
