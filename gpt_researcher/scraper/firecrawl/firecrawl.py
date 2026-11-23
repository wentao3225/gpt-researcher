import asyncio
import os

from bs4 import BeautifulSoup

from ..utils import get_relevant_images

# 在所有 FireCrawl 实例间共享的模块级信号量。
# 限制并发 API 调用以避免超出 FireCrawl 速率限制。
# FireCrawl 免费层允许 2 个并发浏览器；可通过 FIRECRAWL_CONCURRENCY 配置。
_semaphore: asyncio.Semaphore | None = None


def _get_semaphore() -> asyncio.Semaphore:
    global _semaphore
    if _semaphore is None:
        max_concurrent = int(os.environ.get("FIRECRAWL_CONCURRENCY", "2"))
        _semaphore = asyncio.Semaphore(max_concurrent)
    return _semaphore


class FireCrawl:

    def __init__(self, link, session=None):
        self.link = link
        self.session = session
        from firecrawl import FirecrawlApp
        self.firecrawl = FirecrawlApp(api_key=self.get_api_key(), api_url=self.get_server_url())

    def get_api_key(self) -> str:
        """
        获取 FireCrawl API 密钥
        返回：
        Api key (str)
        """
        try:
            api_key = os.environ["FIRECRAWL_API_KEY"]
        except KeyError:
            raise Exception(
                "FireCrawl API key not found. Please set the FIRECRAWL_API_KEY environment variable.")
        return api_key

    def get_server_url(self) -> str:
        """
        获取 FireCrawl 服务器 URL。
        默认为官方 FireCrawl 服务器（'https://api.firecrawl.dev'）。
        返回：
        server url (str)
        """
        try:
            server_url = os.environ["FIRECRAWL_SERVER_URL"]
        except KeyError:
            server_url = 'https://api.firecrawl.dev'
        return server_url

    def scrape(self) -> tuple:
        """
        此函数使用 FireCrawl Python SDK 从指定链接提取内容和标题，
        链接中的图片使用 `gpt_researcher/scraper/utils.py` 中的函数提取。

        返回：
          `scrape` 方法返回一个元组，包含提取的内容、图片 URL 列表，
        以及 `self.link` 属性指定的网页标题。它使用 FireCrawl Python SDK
        提取并清理网页内容。如果过程中出现任何异常，会打印错误消息
        并返回空结果。
        """

        try:
            # 修复：从 scrape_url() 改为 scrape() 以匹配 FireCrawl SDK v4.6.0+
            response = self.firecrawl.scrape(url=self.link, formats=["markdown"])

            # 检查页面是否已成功抓取
            # 修复：直接访问元数据属性（而非字典键）
            if response.metadata and response.metadata.error:
                print("Scrape failed! : " + str(response.metadata.error))
                return "", [], ""
            elif response.metadata and response.metadata.status_code and response.metadata.status_code != 200:
                print(f"Scrape failed! Status code: {response.metadata.status_code}")
                return "", [], ""

            # 从 FireCrawl 响应中提取内容（markdown）和标题
            # 修复：直接访问属性（而非字典键）
            content = response.markdown if response.markdown else ""
            title = response.metadata.title if response.metadata and response.metadata.title else ""

            # 可选的图片提取。在 Scraper 之外使用 FireCrawl 时
            # session 可能为 None；绝不因 session.get 抛出属性错误。
            image_urls = []
            if self.session is not None:
                try:
                    response_bs = self.session.get(self.link, timeout=4)
                    soup = BeautifulSoup(
                        response_bs.content,
                        "lxml",
                        from_encoding=response_bs.encoding,
                    )
                    image_urls = get_relevant_images(soup, self.link)
                except Exception as img_err:
                    print(f"FireCrawl image extraction skipped: {img_err}")

            return content, image_urls, title

        except Exception as e:
            print("Error! : " + str(e))
            return "", [], ""

    async def scrape_async(self) -> tuple:
        """
        scrape() 的异步版本，带并发限制以避免 FireCrawl API 速率限制。

        FireCrawl 免费层将并发浏览器限制为 2。深度研究模式启动许多并行
        请求时，大多数会静默失败并返回空内容。此方法使用共享的模块级
        信号量，使所有 FireCrawl 实例中同时至多有 FIRECRAWL_CONCURRENCY
        个请求运行（默认：2）。

        返回：
            (content, image_urls, title) 元组 —— 与 scrape() 相同。
        """
        async with _get_semaphore():
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(None, self.scrape)
