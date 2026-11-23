# Tavily API 检索器

# 库
import os
import requests
import json
from urllib.parse import urlencode, urlparse


class GoogleSearch:
    """
    Google API 检索器
    """
    def __init__(self, query, headers=None, query_domains=None):
        """
        初始化 GoogleSearch 对象
        参数：
            query:
        """
        self.query = query
        self.headers = headers or {}
        self.query_domains = query_domains or None
        self.api_key = self.headers.get("google_api_key") or self.get_api_key()  # 使用传入的 api_key 或回退到环境变量
        self.cx_key = self.headers.get("google_cx_key") or self.get_cx_key()  # 使用传入的 cx_key 或回退到环境变量

    def get_api_key(self):
        """
        获取 Google API 密钥
        返回：

        """
        # 获取 API 密钥
        try:
            api_key = os.environ["GOOGLE_API_KEY"]
        except Exception:
            raise Exception("Google API key not found. Please set the GOOGLE_API_KEY environment variable. "
                            "You can get a key at https://developers.google.com/custom-search/v1/overview")
        return api_key

    def get_cx_key(self):
        """
        获取 Google CX 密钥
        返回：

        """
        # 获取 API 密钥
        try:
            api_key = os.environ["GOOGLE_CX_KEY"]
        except Exception:
            raise Exception("Google CX key not found. Please set the GOOGLE_CX_KEY environment variable. "
                            "You can get a key at https://developers.google.com/custom-search/v1/overview")
        return api_key

    def search(self, max_results=7):
        """
        使用 Google Custom Search API 搜索查询，可选限制到特定域名
        返回：
            list: 包含 title、href 和 body 的搜索结果列表
        """
        # 如果指定则构建带域名限制的查询
        search_query = self.query
        if self.query_domains and len(self.query_domains) > 0:
            domain_query = " OR ".join([f"site:{domain}" for domain in self.query_domains])
            search_query = f"({domain_query}) {self.query}"

        print("Searching with query {0}...".format(search_query))

        # 对每个参数做 URL 编码。直接内插原始查询会破坏任何包含
        # 保留字符的搜索（如 "AT&T" 中的 "&" 会添加虚假参数，
        # "#" 会截断查询的其余部分）。
        query_string = urlencode(
            {
                "key": self.api_key,
                "cx": self.cx_key,
                "q": search_query,
                "start": 1,
            }
        )
        url = f"https://www.googleapis.com/customsearch/v1?{query_string}"
        resp = requests.get(url, timeout=20)

        if resp.status_code < 200 or resp.status_code >= 300:
            print("Google search: unexpected response status: ", resp.status_code)

        if resp is None:
            return []
        try:
            search_results = json.loads(resp.text)
        except Exception:
            return []
        if not isinstance(search_results, dict):
            return []

        results = search_results.get("items", []) or []
        search_response = []

        # 将结果规范化以匹配其他搜索 API 的格式。
        # 使用 .get 使缺失的标题/摘要不会丢弃有效链接；
        # 直接跳过非字典行和空链接。
        for result in results:
            if not isinstance(result, dict):
                continue
            link = result.get("link") or ""
            if not link:
                continue
            hostname = urlparse(link).hostname or ""
            if hostname == "youtube.com" or hostname.endswith(".youtube.com"):
                continue
            search_response.append(
                {
                    "title": result.get("title") or "",
                    "href": link,
                    "body": result.get("snippet") or "",
                }
            )

        return search_response[:max_results]
