import os
from ..utils import check_pkg


class ExaSearch:
    """
    Exa API 检索器
    """

    def __init__(self, query, query_domains=None):
        """
        初始化 ExaSearch 对象。
        参数：
            query: 搜索查询。
        """
        # 此校验是必需的，因为 exa_py 是可选的
        check_pkg("exa_py")
        from exa_py import Exa
        self.query = query
        self.api_key = self._retrieve_api_key()
        self.client = Exa(api_key=self.api_key)
        self.query_domains = query_domains or None

    def _retrieve_api_key(self):
        """
        从环境变量获取 Exa API 密钥。
        返回：
            API 密钥。
        抛出：
            Exception: 如果未找到 API 密钥。
        """
        try:
            api_key = os.environ["EXA_API_KEY"]
        except KeyError:
            raise Exception(
                "Exa API key not found. Please set the EXA_API_KEY environment variable. "
                "You can obtain your key from https://exa.ai/"
            )
        return api_key

    def search(
        self, max_results=10, use_autoprompt=False, search_type="neural", **filters
    ):
        """
        使用 Exa API 搜索查询。
        参数：
            max_results: 返回结果的最大数量。
            use_autoprompt: 是否使用自动提示。
            search_type: 搜索类型（如 "neural"、"keyword"）。
            **filters: 额外的过滤器（如日期范围、域名）。
        返回：
            搜索结果列表。
        """
        try:
            results = self.client.search(
                self.query,
                type=search_type,
                use_autoprompt=use_autoprompt,
                num_results=max_results,
                include_domains=self.query_domains,
                **filters
            )
        except Exception as e:
            print(f"Error: {e}. Failed fetching sources from Exa. Empty response.")
            return []

        search_response = []
        rows = getattr(results, "results", None) or []
        if not isinstance(rows, list):
            return []
        for result in rows:
            href = getattr(result, "url", None)
            if not href:
                continue
            body = getattr(result, "text", None) or getattr(result, "summary", None) or ""
            search_response.append({"href": href, "body": body})
        return search_response

    def find_similar(self, url, exclude_source_domain=False, **filters):
        """
        使用 Exa API 查找与所提供 URL 相似的文档。
        参数：
            url: 要查找相似文档的 URL。
            exclude_source_domain: 是否在结果中排除来源域名。
            **filters: 额外的过滤器。
        返回：
            相似文档列表。
        """
        # 参照 search() 的加固方式：此前的任何失败都会传播到研究运行中，
        # 而返回 ``None`` 而非结果对象的客户端会在 ``.results`` 访问时
        # 抛出 AttributeError。
        try:
            results = self.client.find_similar(
                url, exclude_source_domain=exclude_source_domain, **filters
            )
        except Exception as e:
            print(
                f"Error: {e}. Failed fetching similar sources from Exa. Empty response."
            )
            return []

        similar_response = []
        rows = getattr(results, "results", None) or []
        if not isinstance(rows, list):
            return []
        for result in rows:
            href = getattr(result, "url", None)
            if not href:
                continue
            body = getattr(result, "text", None) or getattr(result, "summary", None) or ""
            similar_response.append({"href": href, "body": body})
        return similar_response

    def get_contents(self, ids, **options):
        """
        使用 Exa API 获取指定 ID 的内容。
        参数：
            ids: 要获取内容的文档 ID。
            **options: 内容获取的额外选项。
        返回：
            文档内容列表。
        """
        try:
            results = self.client.get_contents(ids, **options)
        except Exception as e:
            print(
                f"Error: {e}. Failed fetching contents from Exa. Empty response."
            )
            return []

        contents_response = []
        rows = getattr(results, "results", None) or []
        if not isinstance(rows, list):
            return []
        for result in rows:
            result_id = getattr(result, "id", None)
            if result_id is None:
                continue
            content = getattr(result, "text", None) or ""
            contents_response.append({"id": result_id, "content": content})
        return contents_response
