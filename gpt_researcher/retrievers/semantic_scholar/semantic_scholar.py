from typing import Dict, List

import requests


class SemanticScholarSearch:
    """
    Semantic Scholar API 检索器
    """

    BASE_URL = "https://api.semanticscholar.org/graph/v1/paper/search"
    VALID_SORT_CRITERIA = ["relevance", "citationCount", "publicationDate"]

    def __init__(self, query: str, sort: str = "relevance", query_domains=None):
        """
        用查询和排序标准初始化 SemanticScholarSearch 类。

        :param query: 搜索查询字符串
        :param sort: 排序标准（'relevance'、'citationCount'、'publicationDate'）
        """
        self.query = query
        assert sort in self.VALID_SORT_CRITERIA, "Invalid sort criterion"
        # 保留精确的（驼峰式）标准。Semantic Scholar API 要求
        # ``citationCount`` / ``publicationDate`` 原样传入；转小写会产生
        # ``citationcount`` / ``publicationdate``，API 会拒绝或静默忽略。
        self.sort = sort

    def search(self, max_results: int = 20) -> List[Dict[str, str]]:
        """
        在 Semantic Scholar 上执行搜索并返回结果。

        :param max_results: 要获取的最大结果数
        :return: 包含每篇论文 title、href 和 body 的字典列表
        """
        params = {
            "query": self.query,
            "limit": max_results,
            "fields": "title,abstract,url,venue,year,authors,isOpenAccess,openAccessPdf",
            "sort": self.sort,
        }

        try:
            response = requests.get(self.BASE_URL, params=params, timeout=10)
            response.raise_for_status()
            payload = response.json()
        except (requests.RequestException, ValueError) as e:
            print(f"An error occurred while accessing Semantic Scholar API: {e}")
            return []

        if not isinstance(payload, dict):
            return []
        results = payload.get("data") or []
        if not isinstance(results, list):
            return []

        search_result = []
        for result in results:
            if not isinstance(result, dict):
                continue
            pdf = result.get("openAccessPdf")
            if not (result.get("isOpenAccess") and isinstance(pdf, dict)):
                continue
            href = pdf.get("url") or ""
            if not href:
                continue
            search_result.append(
                {
                    "title": result.get("title") or "No Title",
                    "href": href,
                    "body": result.get("abstract") or "Abstract not available",
                }
            )

        return search_result
