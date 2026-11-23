# SearchApi 检索器

# 库
import logging
import os
import requests
import urllib.parse


class SearchApiSearch():
    """
    SearchApi 检索器
    """
    def __init__(self, query, query_domains=None):
        """
        初始化 SearchApiSearch 对象
        参数：
            query:
            query_domains: 可选的域名列表，用于限制搜索范围
        """
        self.query = query
        self.query_domains = query_domains or None
        self.api_key = self.get_api_key()

    def get_api_key(self):
        """
        获取 SearchApi API 密钥
        返回：

        """
        try:
            api_key = os.environ["SEARCHAPI_API_KEY"]
        except Exception:
            raise Exception("SearchApi key not found. Please set the SEARCHAPI_API_KEY environment variable. "
                            "You can get a key at https://www.searchapi.io/")
        return api_key

    def search(self, max_results=7):
        """
        搜索查询
        返回：

        """
        """使用 SearchApi 进行通用互联网搜索查询。"""
        # 与 google 和 serper 检索器一样限制到请求的域名；
        # 否则过滤器会被接受但被忽略。
        search_query = self.query
        if self.query_domains and len(self.query_domains) > 0:
            domain_query = " OR ".join([f"site:{domain}" for domain in self.query_domains])
            search_query = f"({domain_query}) {self.query}"

        print("SearchApiSearch: Searching with query {0}...".format(search_query))

        url = "https://www.searchapi.io/api/v1/search"
        params = {
            "q": search_query,
            "engine": "google",
        }

        headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {self.api_key}',
            'X-SearchApi-Source': 'gpt-researcher'
        }

        encoded_url = url + "?" + urllib.parse.urlencode(params)
        search_response = []

        try:
            response = requests.get(encoded_url, headers=headers, timeout=20)
            if response.status_code != 200:
                # 之前失败的调用会返回空列表且不记录任何日志，
                # 使过期密钥看起来像"未找到结果"。
                logging.getLogger(__name__).warning(
                    "SearchApiSearch: request failed with status %s (%s). "
                    "Returning empty response.",
                    response.status_code,
                    response.text[:200],
                )
            else:
                search_results = response.json() or {}
                # ``organic_results`` 可能缺失（如无匹配、错误载荷，
                # 或非 google 引擎的响应）。默认为 []，使缺失的键不会
                # 抛出 KeyError 并通过下面的宽泛 ``except`` 静默丢弃
                # 所有结果。
                results = search_results.get("organic_results") or []
                results_processed = 0
                for result in results:
                    href = result.get("link") or ""
                    # 跳过 youtube 结果
                    if "youtube.com" in href:
                        continue
                    if results_processed >= max_results:
                        break
                    search_result = {
                        "title": result.get("title") or "",
                        "href": href,
                        "body": result.get("snippet") or "",
                    }
                    search_response.append(search_result)
                    results_processed += 1
        except Exception as e:
            logging.getLogger(__name__).warning(
                "SearchApiSearch: failed fetching sources (%s). "
                "Returning empty response.",
                e,
            )
            search_response = []

        return search_response
