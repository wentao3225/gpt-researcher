"""GPT Researcher 的 Tavily API 搜索检索器。

此模块提供使用 Tavily API 执行网络搜索的 TavilySearch 类。
"""

import json
import os
import re
from typing import Literal, Optional, Sequence

import requests

# Google 风格的 site:域名 操作符，Tavily API 不支持。
_SITE_OPERATOR_PATTERN = re.compile(r"site:(\S+)", re.IGNORECASE)


class TavilySearch:
    """
    Tavily API 检索器
    """

    # Tavily 的 search() 从不设置 include_raw_content，因此结果始终是
    # 链接加摘要 —— 页面仍需抓取。
    requires_scraping = True

    def __init__(self, query, headers=None, topic="general", query_domains=None):
        """
        初始化 TavilySearch 对象。

        参数：
            query (str): 搜索查询字符串。
            headers (dict, 可选): 请求中包含的额外请求头。默认为 None。
            topic (str, 可选): 搜索的主题。默认为 "general"。
            query_domains (list, 可选): 要包含在搜索中的域名列表。默认为 None。
        """
        self.query = query
        self.headers = headers or {}
        self.topic = topic
        self.base_url = "https://api.tavily.com/search"
        self.api_key = self.get_api_key()
        self.headers = {
            "Content-Type": "application/json",
        }
        self.query_domains = query_domains or None

    def get_api_key(self):
        """
        获取 Tavily API 密钥
        返回：

        """
        api_key = self.headers.get("tavily_api_key")
        if not api_key:
            try:
                api_key = os.environ["TAVILY_API_KEY"]
            except KeyError:
                print(
                    "Tavily API key not found, set to blank. If you need a retriever, please set the TAVILY_API_KEY environment variable."
                )
                return ""
        return api_key


    def _search(
        self,
        query: str,
        search_depth: Literal["basic", "advanced"] = "basic",
        topic: str = "general",
        days: int = 2,
        max_results: int = 10,
        include_domains: Sequence[str] = None,
        exclude_domains: Sequence[str] = None,
        include_answer: bool = False,
        include_raw_content: bool = False,
        include_images: bool = False,
        use_cache: bool = True,
    ) -> dict:
        """
        向 API 发送请求的内部搜索方法。
        """

        data = {
            "query": query,
            "search_depth": search_depth,
            "topic": topic,
            "days": days,
            "include_answer": include_answer,
            "include_raw_content": include_raw_content,
            "max_results": max_results,
            "include_domains": include_domains,
            "exclude_domains": exclude_domains,
            "include_images": include_images,
            "api_key": self.api_key,
            "use_cache": use_cache,
        }

        response = requests.post(
            self.base_url, data=json.dumps(data), headers=self.headers, timeout=100
        )

        if response.status_code == 200:
            return response.json()
        else:
            # 如果 HTTP 请求返回不成功的状态码则抛出 HTTPError
            response.raise_for_status()

    def search(self, max_results=10):
        """
        搜索查询
        返回：

        """
        try:
            # LLM 生成的查询常使用 Google 风格的 site: 操作符，
            # Tavily 会拒绝（返回零结果）。改为将其转换成
            # Tavily 的 include_domains 参数。
            query = self.query
            include_domains = self.query_domains
            site_domains = _SITE_OPERATOR_PATTERN.findall(query)
            if site_domains:
                query = _SITE_OPERATOR_PATTERN.sub("", query).strip()
                # 只保留域名部分（Tavily 匹配域名，不匹配路径）
                site_domains = [d.strip(",").split("/")[0] for d in site_domains]
                include_domains = list(dict.fromkeys(site_domains + (include_domains or [])))

            # 搜索查询（Tavily 拒绝超过 400 字符的查询）
            results = self._search(
                query[:400],
                search_depth="basic",
                max_results=max_results,
                topic=self.topic,
                include_domains=include_domains,
            )
            # API/代理故障可能产生列表或标量 JSON 体；只有字典响应
            # 才有我们理解顶层 "results" 键。
            if not isinstance(results, dict):
                raise Exception("No results found with Tavily API search.")
            sources = results.get("results", [])
            if not isinstance(sources, list) or not sources:
                raise Exception("No results found with Tavily API search.")
            # 返回结果。对每个来源做缺失/None 字段防护，
            # 使单个畸形命中不会丢弃整个页面。
            search_response = []
            for obj in sources:
                if not isinstance(obj, dict):
                    continue
                href = obj.get("url")
                if not href:
                    continue
                body = obj.get("content") or obj.get("snippet") or ""
                search_response.append({"href": href, "body": body})
        except Exception as e:
            print(f"Error: {e}. Failed fetching sources. Resulting in empty response.")
            search_response = []
        return search_response
