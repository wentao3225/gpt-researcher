import os
from typing import Dict, List, Optional

import requests


class OpenAlexSearch:
    """
    OpenAlex API 检索器。

    OpenAlex（https://openalex.org）是开放的学术著作目录。
    默认使用无需 API 密钥。

    可选环境变量：
    - OPENALEX_EMAIL: 将调用方加入 OpenAlex 的礼貌池，获得更可预测的
      速率限制（生产环境推荐）。
    - OPENALEX_API_KEY: 带更高速率限制的认证访问
      （在 https://openalex.org/ 免费注册）。

    当前速率限制详情见
    https://docs.openalex.org/how-to-use-the-api/rate-limits-and-authentication。
    """

    BASE_URL = "https://api.openalex.org/works"
    VALID_SORT_CRITERIA = [
        "relevance_score:desc",
        "cited_by_count:desc",
        "publication_date:desc",
    ]

    def __init__(self, query: str, sort: str = "relevance_score:desc", query_domains=None):
        """
        用查询和排序标准初始化 OpenAlexSearch 类。

        :param query: 搜索查询字符串。
        :param sort: 排序标准。VALID_SORT_CRITERIA 之一。
        """
        self.query = query
        assert sort in self.VALID_SORT_CRITERIA, f"Invalid sort criterion: {sort}"
        self.sort = sort
        self.email: Optional[str] = os.environ.get("OPENALEX_EMAIL")
        self.api_key: Optional[str] = os.environ.get("OPENALEX_API_KEY")

    def search(self, max_results: int = 20) -> List[Dict[str, str]]:
        """
        在 OpenAlex 上执行搜索并返回结果。

        :param max_results: 要获取的最大结果数（每次请求上限 25）。
        :return: 包含每部著作 title、href 和 body 的字典列表。
        """
        params = {
            "search": self.query,
            "per_page": min(max_results, 25),
            "sort": self.sort,
        }
        if self.email:
            params["mailto"] = self.email
        if self.api_key:
            params["api_key"] = self.api_key

        try:
            response = requests.get(self.BASE_URL, params=params, timeout=10)
            response.raise_for_status()
            payload = response.json()
        except (requests.RequestException, ValueError) as e:
            print(f"An error occurred while accessing OpenAlex API: {e}")
            return []

        if not isinstance(payload, dict):
            return []
        results = payload.get("results", [])
        if not isinstance(results, list):
            return []

        search_result = []
        for result in results:
            if not isinstance(result, dict):
                continue
            title = result.get("title") or "No Title"
            href = self._pick_href(result)
            body = self._reconstruct_abstract(result.get("abstract_inverted_index"))

            if href:
                search_result.append(
                    {
                        "title": title,
                        "href": href,
                        "body": body or "Abstract not available",
                    }
                )

        return search_result

    @staticmethod
    def _pick_href(result: dict) -> Optional[str]:
        """
        优先使用开放获取 PDF URL，然后是 primary_location 的落地页 URL，
        最后回退到 OpenAlex 著作 URL。
        """
        oa_location = result.get("best_oa_location")
        if not isinstance(oa_location, dict):
            oa_location = {}
        pdf_url = oa_location.get("pdf_url")
        if pdf_url:
            return pdf_url

        primary = result.get("primary_location")
        if not isinstance(primary, dict):
            primary = {}
        landing = primary.get("landing_page_url")
        if landing:
            return landing

        return result.get("id")

    @staticmethod
    def _reconstruct_abstract(inverted: Optional[dict]) -> Optional[str]:
        """
        OpenAlex 以倒排索引（单词 -> 位置）返回摘要。
        重建原始文本。
        """
        if not inverted or not isinstance(inverted, dict):
            return None
        positions: List[tuple] = []
        for word, indexes in inverted.items():
            if not isinstance(indexes, (list, tuple)):
                continue
            for i in indexes:
                try:
                    positions.append((int(i), word))
                except (TypeError, ValueError):
                    continue
        if not positions:
            return None
        positions.sort(key=lambda x: x[0])
        return " ".join(word for _, word in positions)
