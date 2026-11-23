import json_repair
import logging
from typing import Any, Dict, List

from gpt_researcher.llm_provider.generic.base import ReasoningEfforts

from ..config import Config
from ..prompts import PromptFamily
from ..utils.llm import create_chat_completion


def _normalize_sub_queries(parsed: Any, fallback_query: str) -> List[str]:
    """将解析出的 LLM 响应规范化为扁平的查询字符串列表。

    模型未返回干净 JSON 时，``json_repair.loads`` 可能返回列表、字典
    （如 ``{"queries": [...]}`` 或单个 ``{"query": "..."}``）、裸字符串
    或 ``None``。调用方期望 ``list[str]``，否则会在 ``.append`` 或迭代时
    崩溃，因此这里做防御性规范化。
    """
    if isinstance(parsed, dict):
        for key in ("queries", "sub_queries", "subQueries", "items"):
            value = parsed.get(key)
            if isinstance(value, list):
                parsed = value
                break
        else:
            # 单查询字典如 {"query": "..."} 或无法识别的形状。
            single = parsed.get("query")
            parsed = [single] if isinstance(single, str) else []

    if isinstance(parsed, str):
        parsed = [parsed] if parsed.strip() else []

    if not isinstance(parsed, list):
        parsed = []

    queries = [str(item).strip() for item in parsed if str(item).strip()]
    if not queries and fallback_query.strip():
        return [fallback_query.strip()]
    return queries

logger = logging.getLogger(__name__)

async def get_search_results(
    query: str,
    retriever: Any,
    query_domains: List[str] = None,
    researcher=None,
    max_results: int | None = None,
) -> List[Dict[str, Any]]:
    """
    为给定查询获取网络搜索结果。

    参数：
        query: 搜索查询
        retriever: 检索器实例
        query_domains: 可选的搜索域名列表
        researcher: 研究员实例（MCP 检索器需要）
        max_results: 可选的结果数量上限

    返回：
        搜索结果列表
    """
    import asyncio

    # 检查是否为 MCP 检索器并传入研究员实例
    if "mcpretriever" in retriever.__name__.lower():
        search_retriever = retriever(
            query, 
            query_domains=query_domains,
            researcher=researcher  # 为 MCP 检索器传入研究员实例
        )
    else:
        search_retriever = retriever(query, query_domains=query_domains)

    search_kwargs = {}
    if max_results is not None:
        search_kwargs["max_results"] = max_results

    # 检索器搜索是阻塞的 HTTP 调用；保持事件循环空闲
    return await asyncio.to_thread(search_retriever.search, **search_kwargs)

async def generate_sub_queries(
    query: str,
    parent_query: str,
    report_type: str,
    context: List[Dict[str, Any]],
    cfg: Config,
    cost_callback: callable = None,
    prompt_family: type[PromptFamily] | PromptFamily = PromptFamily,
    **kwargs
) -> List[str]:
    """
    使用指定的 LLM 模型生成子查询。

    参数：
        query: 原始查询
        parent_query: 父查询
        report_type: 报告类型
        max_iterations: 最大研究迭代次数
        context: 搜索结果上下文
        cfg: 配置对象
        cost_callback: 用于成本计算的回调
        prompt_family: 提示词家族

    返回：
        子查询列表
    """
    gen_queries_prompt = prompt_family.generate_search_queries_prompt(
        query,
        parent_query,
        report_type,
        max_iterations=cfg.max_iterations or 3,
        context=context,
    )

    try:
        response = await create_chat_completion(
            model=cfg.strategic_llm_model,
            messages=[{"role": "user", "content": gen_queries_prompt}],
            llm_provider=cfg.strategic_llm_provider,
            max_tokens=None,
            llm_kwargs=cfg.llm_kwargs,
            reasoning_effort=ReasoningEfforts.Medium.value,
            cost_callback=cost_callback,
            **kwargs
        )
    except Exception as e:
        logger.warning(f"Error with strategic LLM: {e}. Retrying with max_tokens={cfg.strategic_token_limit}.")
        logger.warning(f"See https://github.com/assafelovic/gpt-researcher/issues/1022")
        try:
            response = await create_chat_completion(
                model=cfg.strategic_llm_model,
                messages=[{"role": "user", "content": gen_queries_prompt}],
                max_tokens=cfg.strategic_token_limit,
                llm_provider=cfg.strategic_llm_provider,
                llm_kwargs=cfg.llm_kwargs,
                cost_callback=cost_callback,
                **kwargs
            )
            logger.warning(f"Retrying with max_tokens={cfg.strategic_token_limit} successful.")
        except Exception as e:
            logger.warning(f"Retrying with max_tokens={cfg.strategic_token_limit} failed.")
            logger.warning(f"Error with strategic LLM: {e}. Falling back to smart LLM.")
            response = await create_chat_completion(
                model=cfg.smart_llm_model,
                messages=[{"role": "user", "content": gen_queries_prompt}],
                temperature=cfg.temperature,
                max_tokens=cfg.smart_token_limit,
                llm_provider=cfg.smart_llm_provider,
                llm_kwargs=cfg.llm_kwargs,
                cost_callback=cost_callback,
                **kwargs
            )

    return _normalize_sub_queries(json_repair.loads(response), query)

async def plan_research_outline(
    query: str,
    search_results: List[Dict[str, Any]],
    agent_role_prompt: str,
    cfg: Config,
    parent_query: str,
    report_type: str,
    cost_callback: callable = None,
    retriever_names: List[str] = None,
    **kwargs
) -> List[str]:
    """
    通过生成子查询来规划研究大纲。

    参数：
        query: 原始查询
        search_results: 初始搜索结果
        agent_role_prompt: 智能体角色提示词
        cfg: 配置对象
        parent_query: 父查询
        report_type: 报告类型
        cost_callback: 用于成本计算的回调
        retriever_names: 正在使用的检索器名称

    返回：
        子查询列表
    """
    # 处理未提供 retriever_names 的情况
    if retriever_names is None:
        retriever_names = []
    
    # 对 MCP 检索器，可能需要跳过子查询生成
    # 检查 MCP 是唯一检索器还是多个检索器之一
    if retriever_names and ("mcp" in retriever_names or "MCPRetriever" in retriever_names):
        mcp_only = (len(retriever_names) == 1 and 
                   ("mcp" in retriever_names or "MCPRetriever" in retriever_names))
        
        if mcp_only:
            # 如果 MCP 是唯一检索器，跳过子查询生成
            logger.info("Using MCP retriever only - skipping sub-query generation")
            # 返回原始查询以避免额外的搜索迭代
            return [query]
        else:
            # 如果 MCP 是多个检索器之一，为其他检索器生成子查询
            logger.info("Using MCP with other retrievers - generating sub-queries for non-MCP retrievers")

    # 为研究大纲生成子查询
    sub_queries = await generate_sub_queries(
        query,
        parent_query,
        report_type,
        search_results,
        cfg,
        cost_callback,
        **kwargs
    )

    return sub_queries
