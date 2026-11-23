from bs4 import BeautifulSoup
from urllib.parse import urljoin
import requests
from ..utils import get_relevant_images, extract_title

class WebBaseLoaderScraper:

    def __init__(self, link, session=None):
        self.link = link
        self.session = session or requests.Session()

    def scrape(self) -> tuple:
        """
        此 Python 函数使用 WebBaseLoader 对象抓取网页内容，并返回
        拼接后的页面内容。
        
        返回：
          `scrape` 方法返回名为 `content` 的字符串变量，包含由
        `WebBaseLoader` 加载的文档的拼接页面内容。如果过程中出现异常，
        会打印错误消息并返回空字符串。
        """
        try:
            from langchain_community.document_loaders import WebBaseLoader
            loader = WebBaseLoader(self.link)
            loader.requests_kwargs = {"verify": False}
            docs = loader.load() or []
            content = ""

            for doc in docs:
                if doc is None:
                    continue
                page = getattr(doc, "page_content", None)
                if page is None:
                    continue
                content += str(page)

            image_urls, title = [], ""
            try:
                response = self.session.get(self.link)
                soup = BeautifulSoup(response.content, 'html.parser')
                image_urls = get_relevant_images(soup, self.link)

                # 使用工具函数提取标题
                title = extract_title(soup)
            except Exception as e:
                print("Error extracting images/title! : " + str(e))

            return content, image_urls, title

        except Exception as e:
            print("Error! : " + str(e))
            return "", [], ""
