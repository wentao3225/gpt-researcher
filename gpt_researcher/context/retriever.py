import os
from enum import Enum
from typing import Any, Dict, List, Optional

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

# 每个文档用于嵌入的 raw_content 最大字符数。
# 大文档（如抓取的 PDF）一次性发送所有分块时可能超出嵌入 API 的
# token 限制（如 OpenAI 每请求 30 万 token 的上限）。
# 默认 50 000 字符（约 12 500 token）；可通过 MAX_CONTENT_CHARS 环境变量覆盖。
_MAX_CONTENT_CHARS = int(os.environ.get("MAX_CONTENT_CHARS", 50000))


class SearchAPIRetriever(BaseRetriever):
    """搜索 API 检索器。"""
    pages: List[Dict] = []

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:

        docs = [
            Document(
                # ``raw_content`` 可能被显式设为 None（抓取器对抓取失败的
                # 页面设为 None），对 None 切片会抛出 TypeError。
                # 截断前先强制转换为字符串。
                page_content=(page.get("raw_content") or "")[:_MAX_CONTENT_CHARS],
                metadata={
                    "title": page.get("title", ""),
                    "source": page.get("url", ""),
                },
            )
            for page in self.pages
        ]

        return docs

class SectionRetriever(BaseRetriever):
    """
    SectionRetriever：
    此类用于检索章节，同时避免冗余的子主题。
    """
    sections: List[Dict] = []
    """
    sections 示例：
    [
        {
            "section_title": "Example Title",
            "written_content": "Example content"
        },
        ...
    ]
    """
    
    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:

        docs = [
            Document(
                page_content=page.get("written_content", ""),
                metadata={
                    "section_title": page.get("section_title", ""),
                },
            )
            for page in self.sections  # 将 'self.pages' 改为 'self.sections'
        ]

        return docs