"""
langchain 向量库的封装
"""
from typing import List, Dict

from langchain_core.documents import Document
from langchain_community.vectorstores import VectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter

class VectorStoreWrapper:
    """
    LangchainVectorStore 的封装，用于处理 GPT-Researcher 文档类型
    """
    def __init__(self, vector_store : VectorStore):
        self.vector_store = vector_store

    def load(self, documents):
        """
        将文档加载到 vector_store
        先转换为 langchain 文档类型，拆分为分块后加载。
        空分块批次不会改变向量库。
        """
        langchain_documents = self._create_langchain_documents(documents)
        splitted_documents = self._split_documents(langchain_documents)
        if not splitted_documents:
            return
        self.vector_store.add_documents(splitted_documents)
    
    def _create_langchain_documents(self, data: List[Dict[str, str]]) -> List[Document]:
        """将 GPT Researcher 文档转换为 Langchain 文档。

        跳过非字典行和没有可用 ``raw_content`` 的条目。缺失的
        ``url`` 仍会生成 source 为空的文档，而不是抛出 KeyError。
        """
        docs: List[Document] = []
        if not data:
            return docs
        for item in data:
            if not isinstance(item, dict):
                continue
            content = item.get("raw_content")
            if content is None:
                continue
            docs.append(
                Document(
                    page_content=str(content),
                    metadata={"source": item.get("url") or ""},
                )
            )
        return docs

    def _split_documents(self, documents: List[Document], chunk_size: int = 1000, chunk_overlap: int = 200) -> List[Document]:
        """
        将文档拆分为更小的分块
        """
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        return text_splitter.split_documents(documents)

    async def asimilarity_search(self, query, k, filter):
        """通过向量库返回查询结果"""
        results = await self.vector_store.asimilarity_search(query=query, k=k, filter=filter)
        return results
