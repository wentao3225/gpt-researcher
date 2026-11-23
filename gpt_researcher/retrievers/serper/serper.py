# Google Serper 检索器

# 库
import os
import requests
import json


class SerperSearch():
    """
    支持 国家、语言和日期过滤 的 Google Serper 检索器
    """
    def __init__(self, query, query_domains=None, country=None, language=None, time_range=None, exclude_sites=None):
        """
        初始化 SerperSearch 对象
        参数：
            query (str): 搜索查询字符串。
            query_domains (list, 可选): 要包含在搜索中的域名列表。默认为 None。
            country (str, 可选): 搜索结果的国家代码（如 'us'、'kr'、'jp'）。默认为 None。
            language (str, 可选): 搜索结果的语言代码（如 'en'、'ko'、'ja'）。默认为 None。
            time_range (str, 可选): 时间范围过滤器（如 'qdr:h'、'qdr:d'、'qdr:w'、'qdr:m'、'qdr:y'）。默认为 None。
            exclude_sites (list, 可选): 要从搜索结果中排除的站点列表。默认为 None。
        """
        self.query = query
        self.query_domains = query_domains or None
        self.country = country or os.getenv("SERPER_REGION")
        self.language = language or os.getenv("SERPER_LANGUAGE")
        self.time_range = time_range or os.getenv("SERPER_TIME_RANGE")
        self.exclude_sites = exclude_sites or self._get_exclude_sites_from_env()
        self.api_key = self.get_api_key()
        self.api_base = os.getenv("SERPER_API_BASE", "https://google.serper.dev").rstrip("/")

    def _get_exclude_sites_from_env(self):
        """
        从环境变量获取要排除的站点列表
        返回：
            list: 要排除的站点列表
        """
        exclude_sites_env = os.getenv("SERPER_EXCLUDE_SITES", "")
        if exclude_sites_env:
            # 按逗号分割并去除空白
            return [site.strip() for site in exclude_sites_env.split(",") if site.strip()]
        return []

    def get_api_key(self):
        """
        获取 Serper API 密钥
        返回：

        """
        try:
            api_key = os.environ["SERPER_API_KEY"]
        except Exception:
            raise Exception("Serper API key not found. Please set the SERPER_API_KEY environment variable. "
                            "You can get a key at https://serper.dev/")
        return api_key

    def search(self, max_results=7):
        """
        搜索查询，支持可选的国家、语言和时间过滤
        返回：
            list: 包含 title、href 和 body 的搜索结果列表
        """
        print("Searching with query {0}...".format(self.query))
        """使用 Serper API 进行通用互联网搜索查询。"""

        # 搜索查询（格式见 https://serper.dev/playground）。
        # 基础 URL 可通过 SERPER_API_BASE 配置，使检索器无需改代码
        # 即可指向代理、区域端点或任何 Serper 兼容服务。
        # 默认指向 Serper 本身。
        url = f"{self.api_base}/search"

        headers = {
            'X-API-KEY': self.api_key,
            'Content-Type': 'application/json'
        }

        # 构建搜索参数
        query_with_filters = self.query

        # 使用 Google 搜索语法排除站点
        if self.exclude_sites:
            for site in self.exclude_sites:
                query_with_filters += f" -site:{site}"

        # 如果指定则添加域名过滤
        if self.query_domains:
            # 向搜索查询添加 site:domain1 OR site:domain2 OR ...
            domain_query = " site:" + " OR site:".join(self.query_domains)
            query_with_filters += domain_query

        search_params = {
            "q": query_with_filters,
            "num": max_results
        }

        # 如果存在则添加可选参数
        if self.country:
            search_params["gl"] = self.country  # 地理位置（国家）

        if self.language:
            search_params["hl"] = self.language  # 主机语言

        if self.time_range:
            search_params["tbs"] = self.time_range  # 基于时间的搜索

        data = json.dumps(search_params)

        resp = requests.request("POST", url, timeout=10, headers=headers, data=data)

        # 预处理结果。始终返回列表，使调用方（执行 `len(...)` 或
        # 迭代结果）永远不会收到 None。
        if resp is None:
            return []
        try:
            search_results = json.loads(resp.text)
        except Exception:
            return []
        if search_results is None or not isinstance(search_results, dict):
            return []

        results = search_results.get("organic") or []
        if not isinstance(results, list):
            return []
        search_results = []

        # 将结果规范化以匹配其他搜索 API 的格式
        # 排除的站点应已被查询参数过滤掉
        for result in results:
            if not isinstance(result, dict):
                continue
            href = result.get("link") or result.get("url") or ""
            if not href:
                continue
            search_results.append(
                {
                    "title": result.get("title") or "",
                    "href": href,
                    "body": result.get("snippet") or result.get("body") or "",
                }
            )

        return search_results
