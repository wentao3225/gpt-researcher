"""GPT Researcher 的上下文压缩工具。

此模块提供使用嵌入和相似度过滤来压缩和检索文档中
相关上下文的类。

压缩管线：
1. 将文档拆分为分块
2. 按与查询的嵌入相似度过滤分块
3. 返回最相关的分块作为上下文

类：
    VectorstoreCompressor: 从向量库检索上下文。
    ContextCompressor: 使用嵌入相似度压缩原始文档。
    WrittenContentCompressor: 压缩先前写好的内容章节。
"""

import asyncio
import os
from typing import Optional

from langchain_classic.retrievers import ContextualCompressionRetriever
from langchain_classic.retrievers.document_compressors import (
    DocumentCompressorPipeline,
    EmbeddingsFilter,
)
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from ..memory.embeddings import OPENAI_EMBEDDING_MODEL
from ..prompts import PromptFamily
from ..utils.costs import estimate_embedding_cost
from ..vector_store import VectorStoreWrapper
from .retriever import SearchAPIRetriever, SectionRetriever


class VectorstoreCompressor:
    """从向量库检索并压缩上下文。

    对现有向量库使用相似度搜索，为给定查询找到相关文档。

    属性：
        vector_store: 要搜索的向量库封装。
        max_results: 返回结果的最大数量。
        filter: 向量库查询的可选过滤器。
    """

    def __init__(
        self,
        vector_store: VectorStoreWrapper,
        max_results: int = 7,
        filter: Optional[dict] = None,
        prompt_family: type[PromptFamily] | PromptFamily = PromptFamily,
        **kwargs,
    ):
        """初始化 VectorstoreCompressor。

        参数：
            vector_store: 要搜索的向量库。
            max_results: 返回结果的最大数量。
            filter: 查询的可选过滤字典。
            prompt_family: 用于格式化输出的提示词家族。
            **kwargs: 额外的关键字参数。
        """
        self.vector_store = vector_store
        self.max_results = max_results
        self.filter = filter
        self.kwargs = kwargs
        self.prompt_family = prompt_family

    async def async_get_context(self, query: str, max_results: int = 5) -> str:
        """从向量库获取相关上下文。

        参数：
            query: 搜索查询。
            max_results: 返回结果的最大数量。

        返回：
            相关文档内容的格式化字符串。
        """
        results = await self.vector_store.asimilarity_search(query=query, k=max_results, filter=self.filter)
        return self.prompt_family.pretty_print_docs(results)


class ContextCompressor:
    """压缩原始文档以提取相关上下文。

    使用嵌入相似度过滤文档分块，只为给定查询返回
    最相关的内容。

    属性：
        documents: 要压缩的文档列表。
        embeddings: 用于相似度计算的嵌入模型。
        max_results: 返回结果的最大数量。
        similarity_threshold: 纳入结果的最低相似度分数。
    """

    def __init__(
        self,
        documents,
        embeddings,
        max_results: int = 5,
        similarity_threshold: float | None = None,
        prompt_family: type[PromptFamily] | PromptFamily = PromptFamily,
        **kwargs,
    ):
        """初始化 ContextCompressor。

        参数：
            documents: 要压缩的文档列表。
            embeddings: 嵌入模型实例。
            max_results: 返回结果的最大数量。
            similarity_threshold: 纳入结果的最低相似度分数。
                未提供时回退到 SIMILARITY_THRESHOLD 环境变量。
            prompt_family: 用于格式化输出的提示词家族。
            **kwargs: 额外的关键字参数。
        """
        self.max_results = max_results
        self.documents = documents
        self.kwargs = kwargs
        self.embeddings = embeddings
        if similarity_threshold is None:
            similarity_threshold = float(os.environ.get("SIMILARITY_THRESHOLD", 0.35))
        self.similarity_threshold = similarity_threshold
        self.prompt_family = prompt_family

    def __get_contextual_retriever(self):
        """构建上下文压缩检索器管线。

        返回：
            配置了文本拆分和基于嵌入过滤的
            ContextualCompressionRetriever。
        """
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        relevance_filter = EmbeddingsFilter(embeddings=self.embeddings,
                                            similarity_threshold=self.similarity_threshold)
        pipeline_compressor = DocumentCompressorPipeline(
            transformers=[splitter, relevance_filter]
        )
        base_retriever = SearchAPIRetriever(
            pages=self.documents
        )
        contextual_retriever = ContextualCompressionRetriever(
            base_compressor=pipeline_compressor, base_retriever=base_retriever
        )
        return contextual_retriever

    async def async_get_context(self, query: str, max_results: int = 5, cost_callback=None) -> str:
        """异步从文档获取相关上下文。

        优化：对小文档集跳过昂贵的压缩管线。
        当文档已经足够精简时，直接使用它们而不做基于嵌入的过滤。

        参数：
            query: 搜索查询。
            max_results: 返回结果的最大数量。
            cost_callback: 用于跟踪嵌入成本的可选回调。

        返回：
            相关文档内容的格式化字符串。
        """
        # 优化：计算总内容大小
        total_chars = sum(len(str(doc.get('raw_content', ''))) for doc in self.documents)
        chunk_threshold = int(os.environ.get("COMPRESSION_THRESHOLD", "8000"))

        # 如果总内容较小，跳过昂贵的压缩直接返回
        if total_chars < chunk_threshold and len(self.documents) <= max_results:
            # 快速路径：无需压缩
            # 将抓取器/检索器字典的键映射为 pretty_print_docs 期望的元数据。
            # 原始字典使用 `url`；SearchAPIRetriever / pretty_print 使用 `source`。
            direct_docs = [
                Document(
                    page_content=doc.get('raw_content', '') or '',
                    metadata={
                        "title": doc.get("title", "") or "",
                        "source": doc.get("source") or doc.get("url") or "",
                    },
                )
                for doc in self.documents[:max_results]
            ]
            return self.prompt_family.pretty_print_docs(direct_docs, max_results)

        # 标准路径：对大内容使用压缩
        compressed_docs = self.__get_contextual_retriever()
        if cost_callback:
            cost_callback(estimate_embedding_cost(model=OPENAI_EMBEDDING_MODEL, docs=self.documents))
        relevant_docs = await asyncio.to_thread(compressed_docs.invoke, query, **self.kwargs)
        return self.prompt_family.pretty_print_docs(relevant_docs, max_results)


class WrittenContentCompressor:
    """压缩先前写好的内容章节。

    专用压缩器，用于从先前写好的报告内容中查找相关章节，
    保留章节标题和结构。

    属性：
        documents: 已写内容章节列表。
        embeddings: 用于相似度计算的嵌入模型。
        similarity_threshold: 纳入结果的最低相似度分数。
    """

    def __init__(self, documents, embeddings, similarity_threshold: float, **kwargs):
        """初始化 WrittenContentCompressor。

        参数：
            documents: 已写内容章节列表。
            embeddings: 嵌入模型实例。
            similarity_threshold: 纳入结果的最低相似度分数。
            **kwargs: 额外的关键字参数。
        """
        self.documents = documents
        self.kwargs = kwargs
        self.embeddings = embeddings
        self.similarity_threshold = similarity_threshold

    def __get_contextual_retriever(self):
        """为章节构建上下文压缩检索器。

        返回：
            为章节检索配置的 ContextualCompressionRetriever。
        """
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        relevance_filter = EmbeddingsFilter(embeddings=self.embeddings,
                                            similarity_threshold=self.similarity_threshold)
        pipeline_compressor = DocumentCompressorPipeline(
            transformers=[splitter, relevance_filter]
        )
        base_retriever = SectionRetriever(
            sections=self.documents
        )
        contextual_retriever = ContextualCompressionRetriever(
            base_compressor=pipeline_compressor, base_retriever=base_retriever
        )
        return contextual_retriever

    def __pretty_docs_list(self, docs, top_n: int) -> list[str]:
        """将文档格式化为标题/内容字符串列表。

        参数：
            docs: 要格式化的文档列表。
            top_n: 包含文档的最大数量。

        返回：
            格式化文档字符串列表。
        """
        return [f"Title: {d.metadata.get('section_title')}\nContent: {d.page_content}\n" for i, d in enumerate(docs) if i < top_n]

    async def async_get_context(self, query: str, max_results: int = 5, cost_callback=None) -> list[str]:
        """异步获取相关的已写内容章节。

        参数：
            query: 搜索查询。
            max_results: 返回结果的最大数量。
            cost_callback: 用于跟踪嵌入成本的可选回调。

        返回：
            格式化章节字符串列表。
        """
        compressed_docs = self.__get_contextual_retriever()
        if cost_callback:
            cost_callback(estimate_embedding_cost(model=OPENAI_EMBEDDING_MODEL, docs=self.documents))
        relevant_docs = await asyncio.to_thread(compressed_docs.invoke, query, **self.kwargs)
        return self.__pretty_docs_list(relevant_docs, max_results)
