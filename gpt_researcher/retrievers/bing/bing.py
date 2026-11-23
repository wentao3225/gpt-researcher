# Bing 搜索检索器

# 库
import os
import requests
import json
import logging


class BingSearch():
    """
    Bing 搜索检索器
    """
    def __init__(self, query, query_domains=None):
        """
        初始化 BingSearch 对象
        参数：
            query:
        """
        self.query = query
        self.query_domains = query_domains or None
        self.api_key = self.get_api_key()
        self.logger = logging.getLogger(__name__)

    def get_api_key(self):
        """
        获取 Bing API 密钥
        返回：

        """
        try:
            api_key = os.environ["BING_API_KEY"]
        except Exception:
            raise Exception(
                "Bing API key not found. Please set the BING_API_KEY environment variable.")
        return api_key

    def search(self, max_results=7) -> list[dict[str]]:
        """
        搜索查询
        返回：

        """
        print("Searching with query {0}...".format(self.query))
        """使用 Bing API 进行通用互联网搜索查询。"""

        # 搜索查询
        url = "https://api.bing.microsoft.com/v7.0/search"

        headers = {
            'Ocp-Apim-Subscription-Key': self.api_key,
            'Content-Type': 'application/json'
        }
        search_query = self.query
        if self.query_domains and len(self.query_domains) > 0:
            domain_query = " OR ".join([f"site:{domain}" for domain in self.query_domains])
            search_query = f"({domain_query}) {self.query}"

        params = {
            "responseFilter": "Webpages",
            "q": search_query,
            "count": max_results,
            "setLang": "en-GB",
            "textDecorations": False,
            "textFormat": "HTML",
            "safeSearch": "Strict"
        }

        try:
            resp = requests.get(url, headers=headers, params=params, timeout=20)
            resp.raise_for_status()
        except requests.RequestException as exc:
            self.logger.error(f"Bing search request failed: {exc}")
            return []

        # 预处理结果
        if resp is None:
            return []
        try:
            search_results = json.loads(resp.text)
            if not isinstance(search_results, dict):
                return []
            web_pages = search_results.get("webPages") or {}
            if not isinstance(web_pages, dict):
                return []
            results = web_pages.get("value") or []
        except Exception as e:
            self.logger.error(
                f"Error parsing Bing search results: {e}. Resulting in empty response.")
            return []
        if not results:
            self.logger.warning(f"No search results found for query: {self.query}")
            return []

        # 将结果规范化以匹配其他搜索 API 的格式。
        # 跳过非字典行（API 漂移 / 错误存根）和空 URL，
        # 而不是在研究中途抛出 AttributeError/'NoneType' 崩溃。
        search_response = []
        if not isinstance(results, list):
            return []
        for result in results:
            if not isinstance(result, dict):
                continue
            url = result.get("url") or ""
            if not url:
                continue
            # 跳过 youtube 结果
            if "youtube.com" in url.lower():
                continue
            search_response.append({
                "title": result.get("name") or "",
                "href": url,
                "body": result.get("snippet") or "",
            })

        return search_response
