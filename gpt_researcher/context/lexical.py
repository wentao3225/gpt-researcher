"""抓取内容的关键词（BM25）相关性过滤。

零依赖的上下文过滤器：无 API、无模型、无嵌入。分块按 BM25 排序
（大多数搜索引擎背后的词法评分），针对为单个子查询抓取的页面
分块计算。本地运行，毫秒级完成。
"""

import math
import re
from collections import Counter

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from ..prompts import PromptFamily
from .retriever import SearchAPIRetriever

_TOKEN = re.compile(r"\w+", re.UNICODE)
_STOPWORDS = frozenset("""
a about above after again against all am an and any are as at be because been before being below
between both but by can could did do does doing down during each few for from further had has have
having he her here hers herself him himself his how i if in into is it its itself just me more most
my myself no nor not now of off on once only or other our ours ourselves out over own same she should
so some such than that the their theirs them themselves then there these they this those through to
too under until up very was we were what when where which while who whom why will with would you your
yours yourself yourselves
""".split())


def _stem(token: str) -> str:
    """轻度去复数，使 "batteries" 匹配 "battery"、"costs" 匹配 "cost"。"""
    if len(token) <= 4 or token.endswith("ss"):
        return token
    if token.endswith("ies"):
        return token[:-3] + "y"
    if token.endswith(("sses", "xes", "zes", "ches", "shes")):
        return token[:-2]
    if token.endswith("s"):
        return token[:-1]
    return token


def tokenize(text: str) -> list[str]:
    return [_stem(t) for t in _TOKEN.findall(text.lower()) if len(t) > 1 and t not in _STOPWORDS]


def bm25_scores(query: str, passages: list[str], k1: float = 1.5, b: float = 0.75) -> list[float]:
    """每个段落对 ``query`` 的 BM25 分数，IDF 取自段落本身。"""
    docs = [Counter(tokenize(p)) for p in passages]
    if not docs:
        return []
    avg_len = sum(sum(d.values()) for d in docs) / len(docs) or 1.0
    query_terms = set(tokenize(query))
    df = {t: sum(1 for d in docs if t in d) for t in query_terms}
    idf = {t: math.log((len(docs) - n + 0.5) / (n + 0.5) + 1) for t, n in df.items()}
    scores = []
    for d in docs:
        length = sum(d.values())
        score = 0.0
        for t in query_terms:
            tf = d.get(t, 0)
            if tf:
                score += idf[t] * tf * (k1 + 1) / (tf + k1 * (1 - b + b * length / avg_len))
        scores.append(score)
    return scores


class LexicalContextCompressor:
    """保留对查询 BM25 分数最高的分块。

    ``relative_threshold`` 会丢弃分数低于最佳分块该比例的分块，
    这样一个只含一段相关文字的页面就不会为了填满预算而
    把它的废话也带进来。
    """

    def __init__(
        self,
        documents,
        relative_threshold: float = 0.0,
        chunk_size: int = 1000,
        prompt_family: type[PromptFamily] | PromptFamily = PromptFamily,
        **kwargs,
    ):
        self.documents = documents
        self.relative_threshold = relative_threshold
        self.chunk_size = chunk_size
        self.prompt_family = prompt_family

    def _chunks(self) -> list[Document]:
        pages = SearchAPIRetriever(pages=self.documents).invoke("")
        splitter = RecursiveCharacterTextSplitter(chunk_size=self.chunk_size, chunk_overlap=self.chunk_size // 10)
        return [c for c in splitter.split_documents(pages) if c.page_content.strip()]

    def select(self, query: str, max_results: int) -> list[Document]:
        chunks = self._chunks()
        scores = bm25_scores(query, [c.page_content for c in chunks])
        best = max(scores, default=0.0)
        if best <= 0:
            # 完全没有关键词重叠：回退到各页面的开头分块。
            return chunks[:max_results]
        ranked = sorted(range(len(chunks)), key=lambda i: (-scores[i], i))
        keep = [i for i in ranked if scores[i] >= self.relative_threshold * best][:max_results]
        return [chunks[i] for i in keep]

    async def async_get_context(self, query: str, max_results: int = 10, cost_callback=None) -> str:
        return self.prompt_family.pretty_print_docs(self.select(query, max_results), max_results)


def rank_written_sections(query: str, sections: list[dict], max_results: int = 10,
                          relative_threshold: float = 0.5) -> list[str]:
    """与 ``query`` 最相关的先前写好的报告章节。

    ``WrittenContentCompressor`` 的关键词对应版本，在没有嵌入模型
    可用时使用。``sections`` 是
    ``{"section_title": ..., "written_content": ...}`` 字典。
    """
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = [
        (section.get("section_title", ""), chunk)
        for section in sections
        for chunk in splitter.split_text(section.get("written_content") or "")
        if chunk.strip()
    ]
    scores = bm25_scores(query, [f"{title}\n{text}" for title, text in chunks])
    best = max(scores, default=0.0)
    if best <= 0:
        return []
    ranked = sorted(range(len(chunks)), key=lambda i: (-scores[i], i))
    return [
        f"Title: {chunks[i][0]}\nContent: {chunks[i][1]}\n"
        for i in ranked if scores[i] >= relative_threshold * best
    ][:max_results]
