import asyncio
import os

from langchain_core.documents import Document
from typing import List, Dict


# 支持 langchain 的基础 Document 类
# - https://github.com/langchain-ai/langchain/blob/master/libs/core/langchain_core/documents/base.py
class LangChainDocumentLoader:

    def __init__(self, documents: List[Document]):
        self.documents = documents

    async def load(self, metadata_source_index="title") -> List[Dict[str, str]]:
        # 参照同目录 ``DocumentLoader.load()`` 的加固方式：
        # ``self.documents`` 是用户提供的（LangChainDocuments 报告来源），
        # 因此单个畸形行 —— None、来自 JSON 载荷的字典，或
        # ``metadata=None`` 的文档 —— 不能中断整个加载并丢弃
        # 已收集的其他文档。跳过不是 Document 形状的行，将 ``None``
        # /缺失的元数据默认为 ``{}``，并丢弃没有可用 ``page_content``
        # 的行，使下游字符串操作不会遇到 ``None``。
        docs: List[Dict[str, str]] = []
        for document in self.documents:
            if not hasattr(document, "page_content"):
                continue

            meta = getattr(document, "metadata", None)
            if not isinstance(meta, dict):
                meta = {}

            page_content = getattr(document, "page_content", None)
            if not page_content:
                continue

            docs.append(
                {
                    "raw_content": str(page_content),
                    "url": meta.get(metadata_source_index, "") or "",
                }
            )
        return docs
