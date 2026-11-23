import os
import json
import requests
from typing import List, Dict
from urllib.parse import urljoin

# gpt_researcher.skills.researcher._search_relevant_source_urls() 会把任何
# raw_content/body 超过 100 字符的搜索结果当作已取回的全文 —— 这条
# 启发式是为真正内联返回全文的检索器（如 PubMed Central）设计的。
# SearxNG 在 "body" 中放的是普通搜索结果摘要，通常超过 100 字符，
# 因此不设上限时每个结果都会被错误地当作已取回，真正的页面
# 永远不会被实际抓取（get_source_urls() 随后返回 []，报告带着零条
# 可验证引用发布）。在源头这里设上限，而不是在
# _search_relevant_source_urls() 本身打补丁，这样该函数分类逻辑的
# 任何未来上游变更（bug 修复、新的记账方式）都会自动继承。
# 截断后的摘要只用于那次分类决策和早期子查询规划 —— 一旦 URL
# 走真正的抓取路径，gpt-researcher 会抓取并使用实际页面内容，
# 而非这个摘要。
_MAX_PREFETCHED_LEN = 100


class SearxSearch():
    """
    SearxNG API 检索器
    """

    # SearxNG 在 "content"/"body" 中放普通结果摘要；真正的页面文本仍需抓取。
    requires_scraping = True
    def __init__(self, query: str, query_domains=None):
        """
        初始化 SearxSearch 对象
        参数：
            query: 搜索查询字符串
        """
        self.query = query
        self.query_domains = query_domains or None
        self.base_url = self.get_searxng_url()

    def get_searxng_url(self) -> str:
        """
        从环境变量获取 SearxNG 实例 URL
        返回：
            str: SearxNG 实例的基础 URL
        """
        try:
            base_url = os.environ["SEARX_URL"]
            if not base_url.endswith('/'):
                base_url += '/'
            return base_url
        except KeyError:
            raise Exception(
                "SearxNG URL not found. Please set the SEARX_URL environment variable. "
                "You can find public instances at https://searx.space/"
            )

    def search(self, max_results: int = 10) -> List[Dict[str, str]]:
        """
        使用 SearxNG API 搜索查询
        参数：
            max_results: 返回结果的最大数量
        返回：
            包含搜索结果的字典列表
        """
        search_url = urljoin(self.base_url, "search")
        # TODO: 添加对查询域名的支持
        params = {
            # 搜索查询。
            'q': self.query,
            # 结果的输出格式。格式需要在 searxng 配置中激活。
            'format': 'json'
        }

        try:
            response = requests.get(
                search_url,
                params=params,
                headers={'Accept': 'application/json'},
                timeout=20,
            )
            response.raise_for_status()
            results = response.json()
        except requests.exceptions.RequestException as e:
            raise Exception(f"Error querying SearxNG: {str(e)}")
        except json.JSONDecodeError:
            raise Exception("Error parsing SearxNG response")

        if not isinstance(results, dict):
            return []

        search_response = []
        raw_results = results.get('results', [])
        if not isinstance(raw_results, list):
            return []

        for result in raw_results:
            if not isinstance(result, dict):
                continue
            href = result.get('url') or result.get('href') or ''
            if not href:
                continue
            body = result.get('content') or result.get('snippet') or ''
            search_response.append({
                "href": href,
                "body": body[:_MAX_PREFETCHED_LEN],
            })
            if len(search_response) >= max_results:
                break

        return search_response
