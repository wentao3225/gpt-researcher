from itertools import islice
from ..utils import check_pkg

# gpt_researcher.skills.researcher._search_relevant_source_urls() 会把任何
# raw_content/body 超过 100 字符的搜索结果当作已取回的全文 —— 这条
# 启发式是为真正内联返回全文的检索器（如 PubMed Central）设计的。
# ddgs 在 "body" 中放的是普通搜索结果摘要，通常超过 100 字符，
# 因此不设上限时每个结果都会被错误地当作已取回，真正的页面
# 永远不会被实际抓取（get_source_urls() 随后返回 []，报告带着零条
# 可验证引用发布）。在源头这里设上限，而不是在
# _search_relevant_source_urls() 本身打补丁，这样该函数分类逻辑的
# 任何未来上游变更（bug 修复、新的记账方式）都会自动继承。
# 截断后的摘要只用于那次分类决策和早期子查询规划 —— 一旦 URL
# 走真正的抓取路径，gpt-researcher 会抓取并使用实际页面内容，
# 而非这个摘要。
_MAX_PREFETCHED_LEN = 100


class Duckduckgo:
    """
    Duckduckgo API 检索器
    """

    # ddgs 在 "body" 中返回摘要；真正的页面文本仍需抓取。
    requires_scraping = True
    def __init__(self, query, query_domains=None):
        check_pkg('ddgs')
        from ddgs import DDGS
        self.ddg = DDGS()
        self.query = query
        self.query_domains = query_domains or None

    def search(self, max_results=5):
        """
        执行搜索并将 ddgs 载荷规范化为其他检索器使用的共享
        ``{href, body, title?}`` 契约。

        ``ddgs`` 历史上返回带 ``href``/`body` 键的字典；较新版本输出
        ``link``/`url` 和 ``body`/`snippet`/`description``。下游研究步骤
        期望 ``href``，如果原样返回原始 ddgs 形状，会导致 KeyError
        或跳过来源。
        """
        # TODO: 添加对查询域名的支持
        try:
            search_response = self.ddg.text(self.query, region='wt-wt', max_results=max_results)
        except Exception as e:
            print(f"Error: {e}. Failed fetching sources. Resulting in empty response.")
            search_response = []

        if not search_response:
            return []

        normalized = []
        for result in search_response:
            if not isinstance(result, dict):
                continue
            href = (
                result.get("href")
                or result.get("link")
                or result.get("url")
                or ""
            )
            if not href:
                continue
            body = (
                result.get("body")
                or result.get("snippet")
                or result.get("description")
                or ""
            )
            item = {"href": href, "body": body[:_MAX_PREFETCHED_LEN]}
            title = result.get("title")
            if title:
                item["title"] = title
            normalized.append(item)

        # 即使客户端忽略了调用方要求的 max_results，也予以保留。
        return list(islice(normalized, max_results))
