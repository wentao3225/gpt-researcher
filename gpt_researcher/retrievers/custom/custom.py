from typing import Any, Dict, List
import requests
import os


class CustomRetriever:
    """
    自定义 API 检索器
    """

    # 文档化契约是 list[{url, raw_content}] —— 调用方自己的端点提供内容。
    requires_scraping = False

    def __init__(self, query: str, query_domains=None):
        self.endpoint = os.getenv('RETRIEVER_ENDPOINT')
        if not self.endpoint:
            raise ValueError("RETRIEVER_ENDPOINT environment variable not set")

        self.params = self._populate_params()
        self.query = query

    def _populate_params(self) -> Dict[str, Any]:
        """
        从以 'RETRIEVER_ARG_' 为前缀的环境变量填充参数
        """
        return {
            key[len('RETRIEVER_ARG_'):].lower(): value
            for key, value in os.environ.items()
            if key.startswith('RETRIEVER_ARG_')
        }

    def search(self, max_results: int = 5) -> List[Dict[str, Any]]:
        """
        使用自定义检索器端点执行搜索。

        :param max_results: 返回结果的最大数量（当前未使用）
        :return: JSON 响应，格式如下：
            [
              {
                "url": "http://example.com/page1",
                "raw_content": "第 1 页的内容"
              },
              {
                "url": "http://example.com/page2",
                "raw_content": "第 2 页的内容"
              }
            ]
        """
        try:
            response = requests.get(
                self.endpoint,
                params={**self.params, "query": self.query},
                timeout=20,
            )
            response.raise_for_status()
            payload = response.json()
        except (requests.RequestException, ValueError) as e:
            # ValueError 覆盖 JSONDecodeError（子类）和其他解析失败。
            print(f"Failed to retrieve search results: {e}")
            return []

        # 契约：调用方迭代返回值。null JSON 体或非列表载荷以前会在
        # 后续表现为 TypeError（或文档中提到但令人意外的 Optional）。
        # 始终返回列表。
        if payload is None:
            return []
        if not isinstance(payload, list):
            print(
                "Custom retriever response must be a JSON list of "
                "{url, raw_content} objects; got "
                f"{type(payload).__name__}"
            )
            return []

        # 契约是 list[{url, raw_content}]。下游对每项调用 .get；
        # 过滤非字典和没有可用 URL 的行，使单个畸形边缘情况
        # 不会让研究管线崩溃。
        cleaned: List[Dict[str, Any]] = []
        for item in payload:
            if not isinstance(item, dict):
                continue
            url = item.get("url") or item.get("href") or ""
            if not url:
                continue
            cleaned.append(
                {
                    "url": url,
                    "raw_content": item.get("raw_content") or item.get("body") or "",
                }
            )
        return cleaned
