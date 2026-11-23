"""使用 TypeSafe 的 Jev 模型对抓取内容做相关性过滤。

``compression.py`` 中的嵌入管线保留向量与查询最接近的分块。
余弦相似度衡量的是主题接近程度，而非某段文字是否有助于回答问题，
因此切题的废话会与包含答案的段落得到同样高的排名。

``JevContextCompressor`` 则直接提问：每个分块由 Jev 按四级有用性
评分标准打分，得分最高的分块成为上下文。它保留嵌入管线的候选集
（相同的拆分器、相同的每页上限）和输出预算，因此两者仅在分块
挑选方式上不同。全程不涉及嵌入模型或向量运算。

API：https://docs.typesafe.ai/api
"""

import asyncio
import logging
import os
import weakref
from dataclasses import dataclass

import httpx
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from ..prompts import PromptFamily
from .retriever import SearchAPIRetriever

logger = logging.getLogger(__name__)

TYPESAFE_BASE_URL = "https://api.typesafe.ai/v1"
SYSTEMONE_PATH = "/systemone"
#: 每百万输入 token 的美元价格；输出 token 免费。
JEV_INPUT_PRICE_PER_MTOK = 0.042

RELEVANCE_LEVELS = [
    "Unrelated to the question",
    "Same topic, but does not help answer the question",
    "Partially answers the question or gives useful supporting facts",
    "Directly answers the question with specific facts",
]

# 每个事件循环一个共享限制：子查询并发运行，否则每个压缩器都会
# 打开自己的全宽连接池。以循环为键，因为 asyncio.Semaphore
# 绑定到第一个等待它的循环。
_SEMAPHORES: "weakref.WeakKeyDictionary[asyncio.AbstractEventLoop, asyncio.Semaphore]" = weakref.WeakKeyDictionary()


def _concurrency() -> asyncio.Semaphore:
    loop = asyncio.get_running_loop()
    if loop not in _SEMAPHORES:
        _SEMAPHORES[loop] = asyncio.Semaphore(int(os.environ.get("JEV_CONCURRENCY", "64")))
    return _SEMAPHORES[loop]


class JevError(RuntimeError):
    """Jev 无法为分块评分（缺少密钥、API 或网络故障）。"""


@dataclass
class ScoredChunk:
    document: Document
    score: float
    position: int


class JevClient:
    """System One 端点的最小异步客户端。"""

    def __init__(self, api_key: str | None = None, model: str | None = None,
                 timeout: float = 30.0, max_retries: int = 3,
                 base_url: str | None = None):
        self.api_key = api_key or os.environ.get("TYPESAFE_API_KEY")
        if not self.api_key:
            raise JevError("TYPESAFE_API_KEY is not set")
        self.model = model or os.environ.get("JEV_MODEL", "jev-latest")
        self.base_url = (
            base_url or os.environ.get("TYPESAFE_BASE_URL") or TYPESAFE_BASE_URL
        ).strip().rstrip("/")
        self.api_url = f"{self.base_url}{SYSTEMONE_PATH}"
        self.timeout = timeout
        self.max_retries = max_retries
        self.input_tokens = 0

    def session(self) -> httpx.AsyncClient:
        """由一批评分调用共享的 HTTP 会话。"""
        return httpx.AsyncClient()

    async def score_relevance(self, client: httpx.AsyncClient, query: str, passage: str) -> float:
        """``passage`` 对 ``query`` 的概率加权有用性，范围 [0, 3]。"""
        payload = {
            "state": passage,
            "model": self.model,
            "questions": {
                "usefulness": {
                    "type": "score",
                    "instructions": f"How useful is this passage for answering the question: {query}",
                    "criteria": RELEVANCE_LEVELS,
                }
            },
        }
        headers = {"Authorization": f"Bearer {self.api_key}"}
        for attempt in range(self.max_retries + 1):
            async with _concurrency():
                try:
                    response = await client.post(self.api_url, json=payload, headers=headers, timeout=self.timeout)
                except httpx.HTTPError as exc:
                    error = exc
                else:
                    if response.status_code == 200:
                        try:
                            body = response.json()
                            score = float(body["answers"]["usefulness"]["score"])
                        except (ValueError, KeyError, TypeError) as exc:
                            raise JevError(f"Unexpected Jev response: {response.text[:200]}") from exc
                        self.input_tokens += body.get("usage", {}).get("input_tokens", 0)
                        return score
                    error = JevError(f"HTTP {response.status_code} from {self.api_url}: {response.text[:200]}")
                    # 只有过载和限流值得重试。
                    if response.status_code not in (429, 529):
                        raise error
            if attempt < self.max_retries:
                await asyncio.sleep(0.5 * 2 ** attempt)
        raise JevError(f"Jev request failed after {self.max_retries + 1} attempts: {error}")

    @property
    def cost(self) -> float:
        return self.input_tokens / 1_000_000 * JEV_INPUT_PRICE_PER_MTOK


class JevContextCompressor:
    """通过询问 Jev 来挑选抓取页面中最有用的分块。

    与 ``ContextCompressor`` 相同的接口：``documents`` 是带有
    ``raw_content``、``url`` 和 ``title`` 的抓取器字典。
    """

    def __init__(
        self,
        documents,
        max_results: int = 5,
        min_score: float | None = None,
        prompt_family: type[PromptFamily] | PromptFamily = PromptFamily,
        client: JevClient | None = None,
        chunk_size: int | None = None,
        **kwargs,
    ):
        self.documents = documents
        self.chunk_size = chunk_size or int(os.environ.get("JEV_CHUNK_SIZE", "1000"))
        self.max_results = max_results
        if min_score is None:
            min_score = float(os.environ.get("JEV_MIN_SCORE", "1.5"))
        self.min_score = min_score
        self.prompt_family = prompt_family
        self.client = client or JevClient()

    def _chunks(self) -> list[Document]:
        pages = SearchAPIRetriever(pages=self.documents).invoke("")
        splitter = RecursiveCharacterTextSplitter(chunk_size=self.chunk_size, chunk_overlap=self.chunk_size // 10)
        return [chunk for chunk in splitter.split_documents(pages) if chunk.page_content.strip()]

    async def rank(self, query: str) -> list[ScoredChunk]:
        """每个分块及其 Jev 分数，最优在前（平分时保持页面顺序）。"""
        chunks = self._chunks()
        async with self.client.session() as http:
            scores = await asyncio.gather(
                *[self.client.score_relevance(http, query, chunk.page_content) for chunk in chunks]
            )
        ranked = [ScoredChunk(chunk, score, i) for i, (chunk, score) in enumerate(zip(chunks, scores))]
        return sorted(ranked, key=lambda c: (-c.score, c.position))

    async def async_get_context(self, query: str, max_results: int = 5, cost_callback=None) -> str:
        # 与 ContextCompressor 相同的捷径：内容很少时无需过滤。
        total_chars = sum(len(str(doc.get("raw_content") or "")) for doc in self.documents)
        if total_chars < int(os.environ.get("COMPRESSION_THRESHOLD", "8000")) and len(self.documents) <= max_results:
            docs = [
                Document(
                    page_content=doc.get("raw_content") or "",
                    metadata={"title": doc.get("title") or "", "source": doc.get("source") or doc.get("url") or ""},
                )
                for doc in self.documents[:max_results]
            ]
            return self.prompt_family.pretty_print_docs(docs, max_results)

        ranked = await self.rank(query)
        if cost_callback:
            cost_callback(self.client.cost)
        selected = [c.document for c in ranked if c.score >= self.min_score][:max_results]
        return self.prompt_family.pretty_print_docs(selected, max_results)
