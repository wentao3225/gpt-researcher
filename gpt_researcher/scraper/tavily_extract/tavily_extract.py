from bs4 import BeautifulSoup
import os
from ..utils import get_relevant_images, extract_title

class TavilyExtract:

    def __init__(self, link, session=None):
        self.link = link
        self.session = session
        from tavily import TavilyClient
        self.tavily_client = TavilyClient(api_key=self.get_api_key())

    def get_api_key(self) -> str:
        """
        获取 Tavily API 密钥
        返回：
        Api key (str)
        """
        try:
            api_key = os.environ["TAVILY_API_KEY"]
        except KeyError:
            raise Exception(
                "Tavily API key not found. Please set the TAVILY_API_KEY environment variable.")
        return api_key

    def scrape(self) -> tuple:
        """
        此函数使用 Tavily Python SDK 从指定链接提取内容，标题和图片
        使用 `gpt_researcher/scraper/utils.py` 中的函数提取。

        返回：
          `scrape` 方法返回一个元组，包含提取的内容、图片 URL 列表，
        以及 `self.link` 属性指定的网页标题。它使用 Tavily Python SDK
        提取并清理网页内容。如果过程中出现任何异常，会打印错误消息
        并返回空结果。
        """

        try:
            response = self.tavily_client.extract(urls=self.link)
            if not isinstance(response, dict):
                return "", [], ""

            # failed_results 可能缺失、为 null 或非空列表。
            failed = response.get("failed_results") or []
            if failed:
                return "", [], ""

            results = response.get("results") or []
            if not isinstance(results, list) or not results:
                return "", [], ""
            first = results[0]
            if not isinstance(first, dict):
                return "", [], ""
            # 优先使用 raw_content；即使 extract 载荷不完整也绝不 KeyError。
            content = first.get("raw_content") or ""
            if not content:
                return "", [], ""

            # 可选的 HTML 分支，用于图片/标题。session 可能未设置
            # （构造函数默认 session=None）—— 直接阻断该分支，而不是
            # 在宽泛 except 内出现 AttributeError。
            image_urls = []
            title = ""
            if self.session is not None:
                response_bs = self.session.get(self.link, timeout=4)
                soup = BeautifulSoup(
                    response_bs.content, "lxml", from_encoding=response_bs.encoding
                )
                image_urls = get_relevant_images(soup, self.link)
                title = extract_title(soup) or ""

            return content, image_urls, title

        except Exception as e:
            print("Error! : " + str(e))
            return "", [], ""
