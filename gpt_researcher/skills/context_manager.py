"""GPT Researcher 的上下文管理器技能。

此模块提供 ContextManager 类，处理研究查询的上下文
检索、压缩和相似度匹配。
"""

import asyncio
import logging
from typing import Dict, List, Optional, Set

from ..actions.utils import stream_output
from ..context.compression import VectorstoreCompressor, WrittenContentCompressor
from ..context.lexical import rank_written_sections
from ..context.select import select_context

logger = logging.getLogger(__name__)


class ContextManager:
    """管理研究的上下文检索和压缩。

    此类处理基于查询查找相似内容、从各种来源
    管理上下文，以及压缩内容以实现高效处理。

    属性：
        researcher: 上级 GPTResearcher 实例。
    """

    def __init__(self, researcher):
        """初始化 ContextManager。

        参数：
            researcher: 拥有此管理器的 GPTResearcher 实例。
        """
        self.researcher = researcher

    async def get_similar_content_by_query(self, query: str, pages: list) -> str:
        """根据查询从页面获取相似内容。

        参数：
            query: 用于查找相似内容的搜索查询。
            pages: 要搜索的页面内容列表。

        返回：
            相关内容的压缩上下文字符串。
        """
        if self.researcher.verbose:
            await stream_output(
                "logs",
                "fetching_query_content",
                f"📚 Getting relevant content based on query: {query}...",
                self.researcher.websocket,
            )

        return await select_context(
            query,
            pages,
            self.researcher.cfg,
            max_results=10,
            prompt_family=self.researcher.prompt_family,
            cost_callback=self.researcher.add_costs,
            embeddings=lambda: self.researcher.memory.get_embeddings(),
        )

    async def get_similar_content_by_query_with_vectorstore(self, query: str, filter: dict | None) -> str:
        """根据查询从向量库获取相似内容。

        参数：
            query: 用于查找相似内容的搜索查询。
            filter: 向量库查询的可选过滤字典。

        返回：
            来自向量库的相关内容的压缩上下文字符串。
        """
        if self.researcher.verbose:
            await stream_output(
                "logs",
                "fetching_query_format",
                f" Getting relevant content based on query: {query}...",
                self.researcher.websocket,
                )
        vectorstore_compressor = VectorstoreCompressor(
            self.researcher.vector_store, filter=filter, prompt_family=self.researcher.prompt_family,
            **self.researcher.kwargs
        )
        return await vectorstore_compressor.async_get_context(query=query, max_results=8)

    async def get_similar_written_contents_by_draft_section_titles(
        self,
        current_subtopic: str,
        draft_section_titles: List[str],
        written_contents: List[Dict],
        max_results: int = 10
    ) -> List[str]:
        """根据草稿章节标题获取相似的已写内容。

        搜索与当前子主题和草稿章节标题匹配的
        相关先前已写内容。

        参数：
            current_subtopic: 当前正在撰写的子主题。
            draft_section_titles: 草稿章节标题字符串列表。
            written_contents: 先前已写内容的字典列表。
            max_results: 返回结果的最大数量。

        返回：
            相关已写内容字符串列表。
        """
        all_queries = [current_subtopic] + draft_section_titles

        async def process_query(query: str) -> Set[str]:
            return set(await self.__get_similar_written_contents_by_query(query, written_contents, **self.researcher.kwargs))

        results = await asyncio.gather(*[process_query(query) for query in all_queries])
        relevant_contents = set().union(*results)
        relevant_contents = list(relevant_contents)[:max_results]

        return relevant_contents

    async def __get_similar_written_contents_by_query(
        self,
        query: str,
        written_contents: List[Dict],
        similarity_threshold: float = 0.5,
        max_results: int = 10
    ) -> List[str]:
        """为单个查询获取相似的已写内容。

        参数：
            query: 用于查找相似内容的查询。
            written_contents: 已写内容字典列表。
            similarity_threshold: 最低相似度分数阈值。
            max_results: 返回结果的最大数量。

        返回：
            相似已写内容字符串列表。
        """
        if self.researcher.verbose:
            await stream_output(
                "logs",
                "fetching_relevant_written_content",
                f"🔎 Getting relevant written content based on query: {query}...",
                self.researcher.websocket,
            )

        try:
            embeddings = self.researcher.memory.get_embeddings()
        except Exception as e:
            # 没有可用的嵌入模型（未配置，或它的包或密钥缺失）：
            # 改为按关键词排序已写章节。
            logger.info(f"Embeddings unavailable ({e}); ranking written content by keywords")
            return rank_written_sections(query, written_contents, max_results=max_results)

        written_content_compressor = WrittenContentCompressor(
            documents=written_contents,
            embeddings=embeddings,
            similarity_threshold=similarity_threshold,
            **self.researcher.kwargs
        )
        return await written_content_compressor.async_get_context(
            query=query, max_results=max_results, cost_callback=self.researcher.add_costs
        )
