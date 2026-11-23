"""GPT Researcher 的 fastCRW API 搜索检索器。

此模块提供使用 fastCRW（Firecrawl 兼容的网页数据引擎，
单个二进制文件；可自托管或使用托管云）执行网络搜索的
CRWRetriever 类。
"""

import json
import os

import requests


class CRWRetriever:
    """
    fastCRW API 检索器
    """

    def __init__(self, query, headers=None, topic="general", query_domains=None):
        """
        初始化 CRWRetriever 对象。

        参数：
            query (str): 搜索查询字符串。
            headers (dict, 可选): 请求中包含的额外请求头。默认为 None。
            topic (str, 可选): 搜索的主题。默认为 "general"。
            query_domains (list, 可选): 要包含在搜索中的域名列表。默认为 None。
        """
        input_headers = headers or {}
        self.query = query
        self.topic = topic
        self.base_url = self.get_base_url(input_headers)
        self.api_key = self.get_api_key(input_headers)
        self.headers = {
            "Content-Type": "application/json",
        }
        if self.api_key:
            self.headers["Authorization"] = f"Bearer {self.api_key}"
        self.query_domains = query_domains or None

    def get_api_key(self, headers):
        """
        获取 fastCRW API 密钥
        参数：
            headers (dict): 传给检索器的请求头。
        返回：

        """
        api_key = headers.get("crw_api_key")
        if not api_key:
            try:
                api_key = os.environ["CRW_API_KEY"]
            except KeyError:
                print(
                    "CRW API key not found, set to blank. If you need a retriever, please set the CRW_API_KEY environment variable."
                )
                return ""
        return api_key

    def get_base_url(self, headers):
        """
        获取 fastCRW 基础 URL，允许自托管覆盖。

        默认为托管云 https://fastcrw.com/api。通过 CRW_API_URL 环境变量
        （或 crw_api_url 请求头）覆盖，指向自托管服务器。
        参数：
            headers (dict): 传给检索器的请求头。
        返回：
            基础 URL（不带尾部斜杠）。
        """
        base_url = headers.get("crw_api_url") or os.environ.get(
            "CRW_API_URL", "https://fastcrw.com/api"
        )
        return base_url.rstrip("/")

    def _search(self, query: str, max_results: int = 10) -> dict:
        """
        向 API 发送请求的内部搜索方法。
        """

        data = {
            "query": query,
            "limit": max_results,
        }

        response = requests.post(
            f"{self.base_url}/v1/search",
            data=json.dumps(data),
            headers=self.headers,
            timeout=100,
        )
        # 如果 HTTP 请求返回不成功的状态码则抛出 HTTPError
        response.raise_for_status()
        results = response.json()
        # fastCRW 将响应包装在 {success, error, data} 信封中。
        if results.get("success") is False:
            raise Exception(results.get("error", "fastCRW API search failed."))
        return results

    def search(self, max_results=10):
        """
        搜索查询
        返回：

        """
        try:
            # 搜索查询
            results = self._search(self.query, max_results=max_results)
            sources = results.get("data") or []
            if not isinstance(sources, list) or not sources:
                raise Exception("No results found with fastCRW API search.")
            # 返回结果。缺少 "url" 的来源不可用，跳过它而不是抛出
            # 丢弃整个结果集的 KeyError。
            search_response = []
            for obj in sources:
                if not isinstance(obj, dict):
                    continue
                href = obj.get("url") or ""
                if not href:
                    continue
                search_response.append(
                    {
                        "href": href,
                        "body": obj.get("markdown") or obj.get("description") or "",
                    }
                )
        except Exception as e:
            print(f"Error: {e}. Failed fetching sources. Resulting in empty response.")
            search_response = []
        return search_response
