# BoCha 搜索检索器

# 库
import logging
import os
import requests


class BoChaSearch:
    """
    BoCha 搜索检索器
    """
    def __init__(self, query, query_domains=None):
        """
        初始化 BoChaSearch 对象
        参数：
            query:
        """
        self.query = query
        self.query_domains = query_domains or None
        self.api_key = self.get_api_key()

    def get_api_key(self):
        """
        获取 BoCha API 密钥
        返回：

        """
        try:
            api_key = os.environ["BOCHA_API_KEY"]
        except KeyError:
            raise Exception(
                "BoCha API key not found. Please set the BOCHA_API_KEY environment variable. "
                "Get a key at https://api.bochaai.com"
            )
        return api_key

    def search(self, max_results=7) -> list[dict[str]]:
        """
        搜索查询
        返回：

        """
        url = 'https://api.bochaai.com/v1/web-search'
        headers = {
            'Authorization': f'Bearer {self.api_key}',  # 请替换为你的API密钥
            'Content-Type': 'application/json'
        }
        data = {
            "query": self.query,
            "freshness": "noLimit",  # 搜索的时间范围，
            "summary": True,  # 是否返回长文本摘要
            "count": max_results
        }

        try:
            response = requests.post(url, headers=headers, json=data, timeout=10)
            response.raise_for_status()
            json_response = response.json()
        except (requests.RequestException, ValueError) as e:
            logging.getLogger(__name__).warning(
                f"Error: {e}. Failed fetching sources. Resulting in empty response."
            )
            return []

        # BoCha 响应形状是 data.webPages.value；错误/空载荷中任何一级
        # 都可能缺失，因此防御性地逐级访问，而不是让整个研究运行
        # 抛出 KeyError。
        results = (
            ((json_response or {}).get("data") or {}).get("webPages") or {}
        ).get("value") or []
        if not isinstance(results, list):
            return []

        search_results = []

        # 将结果规范化以匹配其他搜索 API 的格式。
        # 跳过非字典行 / 空 URL；缺失字段默认为 ""。
        for result in results:
            if not isinstance(result, dict):
                continue
            href = result.get("url") or ""
            if not href:
                continue
            search_results.append(
                {
                    "title": result.get("name") or "",
                    "href": href,
                    "body": result.get("snippet") or "",
                }
            )

        return search_results