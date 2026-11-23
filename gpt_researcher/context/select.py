"""以统一的方式将文本页面转换为 LLM 上下文。

研究子查询（抓取的页面）和报告对话（已完成的报告）都调用
``select_context``。``CONTEXT_FILTER`` 决定分块的挑选方式：

  auto        设置了 TYPESAFE_API_KEY 时用 jev，否则用 keyword
  jev         TypeSafe 的 Jev 为每个分块的有用性评分（需要密钥）
  keyword     BM25 关键词排序：本地运行，无 API、无模型、无嵌入
  embeddings  嵌入相似度（需要嵌入提供者）
  none        不过滤：每个页面全文保留

任何失败都会降级为 ``keyword``，因此任何模式都不会让一次运行
没有上下文。基准测试：``evals/context_filter``。
"""

import logging
import os
from typing import Callable

from langchain_core.documents import Document

from ..prompts import PromptFamily
from .jev_filter import JevContextCompressor, JevError
from .lexical import LexicalContextCompressor
from .retriever import SearchAPIRetriever

logger = logging.getLogger(__name__)

CONTEXT_FILTERS = ("auto", "jev", "keyword", "embeddings", "none")

#: 关键词分块的分数至少要达到最佳分块的该比例，最多保留这么多。
#: 由 evals/context_filter 选定：该阈值是能胜过嵌入方案的设置；
#: 纯 top-10 BM25 会输给嵌入方案。
KEYWORD_RELATIVE_THRESHOLD = float(os.environ.get("KEYWORD_RELATIVE_THRESHOLD", "0.5"))
KEYWORD_MAX_RESULTS = int(os.environ.get("KEYWORD_MAX_RESULTS", "25"))


def resolve_context_filter(setting: str | None) -> str:
    """要使用的过滤器；``auto`` 表示设置了 TypeSafe 密钥时用 Jev，否则用 keyword。"""
    mode = (setting or "auto").strip().lower()
    if mode not in CONTEXT_FILTERS:
        logger.warning(f"Unknown CONTEXT_FILTER {setting!r}; expected one of {CONTEXT_FILTERS}. Using 'auto'.")
        mode = "auto"
    if mode == "auto":
        return "jev" if os.environ.get("TYPESAFE_API_KEY") else "keyword"
    return mode


def _as_documents(pages: list[dict]) -> list[Document]:
    return SearchAPIRetriever(pages=pages).invoke("")


async def select_context(
    query: str,
    pages: list[dict],
    cfg,
    *,
    max_results: int = 10,
    prompt_family: type[PromptFamily] | PromptFamily = PromptFamily,
    cost_callback: Callable[[float], None] | None = None,
    embeddings: Callable[[], object] | None = None,
) -> str:
    """从 ``pages``（含 ``raw_content``、``url``、``title`` 的字典）为 ``query`` 获取上下文。

    ``embeddings`` 是零参数工厂，仅在 embeddings 模式下调用，
    因此未选择该模式时不会构建任何嵌入模型。
    """
    mode = resolve_context_filter(getattr(cfg, "context_filter", None))
    prompt_family = prompt_family or PromptFamily

    if mode == "none":
        return prompt_family.pretty_print_docs(_as_documents(pages))

    # 内容很少时无需过滤，无论何种模式。
    total_chars = sum(len(str(p.get("raw_content") or "")) for p in pages)
    if total_chars < int(os.environ.get("COMPRESSION_THRESHOLD", "8000")) and len(pages) <= max_results:
        return prompt_family.pretty_print_docs(_as_documents(pages)[:max_results], max_results)

    if mode == "jev":
        try:
            return await JevContextCompressor(documents=pages, prompt_family=prompt_family).async_get_context(
                query=query, max_results=max_results, cost_callback=cost_callback)
        except JevError as e:
            logger.warning(f"Jev context filter unavailable ({e}); using keyword ranking")

    if mode == "embeddings":
        try:
            from .compression import ContextCompressor

            return await ContextCompressor(
                documents=pages,
                embeddings=embeddings() if embeddings else None,
                similarity_threshold=getattr(cfg, "similarity_threshold", None),
                prompt_family=prompt_family,
            ).async_get_context(query=query, max_results=max_results, cost_callback=cost_callback)
        except Exception as e:
            logger.warning(f"Embedding context filter unavailable ({e}); using keyword ranking")

    return await LexicalContextCompressor(
        documents=pages, relative_threshold=KEYWORD_RELATIVE_THRESHOLD, prompt_family=prompt_family,
    ).async_get_context(query, KEYWORD_MAX_RESULTS)
