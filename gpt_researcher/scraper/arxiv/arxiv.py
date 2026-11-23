"""使用维护中的 `arxiv` Client API 的 arXiv 论文抓取器。

避免使用 langchain_community.ArxivRetriever，它仍调用已被移除的
`arxiv.Search.results()` 方法（对 arxiv>=2.2 已损坏）。
"""

from __future__ import annotations

import re
from typing import Any


_ID_RE = re.compile(
    r"(?:arxiv\.org/(?:abs|pdf|html)/)?(?P<id>\d{4}\.\d{4,5}(?:v\d+)?|[a-z\-]+/\d{7})(?:\.pdf)?",
    re.IGNORECASE,
)


def _paper_id_from_link(link: str) -> str:
    """从 URL 或裸 id 字符串提取 arXiv id。"""
    if not link:
        return ""
    m = _ID_RE.search(link.strip())
    if m:
        return m.group("id")
    # 回退到最后一个路径段（旧行为）
    return link.rstrip("/").split("/")[-1].removesuffix(".pdf")


class ArxivScraper:
    def __init__(self, link, session=None):
        self.link = link
        self.session = session

    def scrape(self):
        """通过 arxiv.Client 获取论文摘要/内容。

        返回：
            (context, images, title)，与其他抓取器一致。

        查询未匹配到论文时（畸形/非 arXiv id，或客户端结果为空），
        降级为空结果而非抛出异常，使单个坏 URL 不会中断整个
        抓取管线。
        """
        paper_id = _paper_id_from_link(self.link)
        if not paper_id:
            return "", [], ""

        import arxiv

        client = arxiv.Client()
        search = arxiv.Search(id_list=[paper_id], max_results=1)
        try:
            paper = next(client.results(search))
        except StopIteration:
            # 没有匹配论文 —— 与其他抓取器一致：空结果降级。
            return "", [], ""

        authors = ", ".join(a.name for a in (paper.authors or []))
        published = paper.published.date().isoformat() if paper.published else ""
        summary = paper.summary or ""
        title = paper.title or paper_id

        context = (
            f"Published: {published}; Author: {authors}; Content: {summary}"
        )
        image: list[Any] = []
        return context, image, title
