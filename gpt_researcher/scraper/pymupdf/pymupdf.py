import os
import requests
import tempfile
from urllib.parse import urlparse
from langchain_community.document_loaders import PyMuPDFLoader


class PyMuPDFScraper:

    def __init__(self, link, session=None):
        """
        用链接和可选的 session 初始化抓取器。

        参数：
          link (str): PDF 文档的 URL 或本地文件路径。
          session (requests.Session, 可选): 用于发出 HTTP 请求的可选 session。
        """
        self.link = link
        self.session = session

    def is_url(self) -> bool:
        """
        检查提供的 `link` 是否为有效 URL。

        返回：
          bool: 如果链接是有效 URL 则返回 True，否则返回 False。
        """
        try:
            result = urlparse(self.link)
            return all([result.scheme, result.netloc])  # 检查有效协议和网络位置
        except Exception:
            return False

    def scrape(self) -> tuple[str, list[str], str]:
        """
        `scrape` 函数使用 PyMuPDFLoader 从提供的链接（URL 或本地文件）
        加载文档，并以字符串形式返回文档。

        返回：
          str: 已加载文档的字符串表示。
        """
        try:
            if self.is_url():
                http = self.session or requests
                try:
                    response = http.get(self.link, timeout=(5, 30), stream=True)
                    response.raise_for_status()
                except requests.exceptions.SSLError:
                    import logging
                    logging.getLogger(__name__).warning(
                        f"SSL verification failed for {self.link}, retrying without verification"
                    )
                    response = http.get(self.link, timeout=(5, 30), stream=True, verify=False)
                    response.raise_for_status()

                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
                    temp_filename = temp_file.name  # 获取临时文件名
                    for chunk in response.iter_content(chunk_size=8192):
                        temp_file.write(chunk)  # 将下载内容写入临时文件

                # 始终清理下载的临时文件，即使加载失败
                # （PyMuPDFLoader.load() 对畸形/不完整 PDF 可能抛出异常）。
                try:
                    loader = PyMuPDFLoader(temp_filename)
                    doc = loader.load()
                finally:
                    try:
                        os.remove(temp_filename)
                    except OSError:
                        pass
            else:
                loader = PyMuPDFLoader(self.link)
                doc = loader.load()

            # 从文档中提取内容、图片（如有）和标题。
            image = []
            # 检索所有页面的内容，确保有封面页的 PDF 也能通过校验。
            content = "\n".join(page.page_content for page in doc)
            title = doc[0].metadata.get("title", "") if doc else ""
            return content, image, title

        except requests.exceptions.Timeout:
            print(f"Download timed out. Please check the link : {self.link}")
            return "", [], ""
        except Exception as e:
            print(f"Error loading PDF : {self.link} {e}")
            return "", [], ""
