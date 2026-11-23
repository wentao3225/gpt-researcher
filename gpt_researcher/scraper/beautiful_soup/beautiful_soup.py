import logging
import time

from bs4 import BeautifulSoup

from ..utils import get_relevant_images, extract_title, get_text_from_soup, clean_soup

logger = logging.getLogger(__name__)

# 值得重试一次的响应码：速率限制和瞬态服务器错误。
RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}
MAX_CONTENT_BYTES = 10 * 1024 * 1024  # 跳过超过 10MB 的页面
TRANSPARENT_RESEARCH_USER_AGENT = (
    "GPT-Researcher/1.0 "
    "(+https://github.com/assafelovic/gpt-researcher; source retrieval)"
)


class BeautifulSoupScraper:

    def __init__(self, link, session=None):
        self.link = link
        self.session = session
        # 保持 scrape() 既定的三项返回值。需要诊断信息的调用方
        # 可以检查这个可选的、有边界的原因。
        self.failure_reason = None

    def scrape(self):
        """抓取页面并提取清理后的文本、图片和标题。

        返回：
            (content, image_urls, title) 元组。页面无法抓取或没有
            可用内容时返回空值。
        """
        response = self._fetch()
        if response is None:
            return "", [], ""

        try:
            # 当 Content-Type 头省略字符集时，response.encoding 默认为
            # ISO-8859-1，会弄乱许多 UTF-8 页面。只在服务器确实声明了
            # 字符集时才信任它；否则让 BeautifulSoup 从文档本身
            # 检测编码。
            content_type = response.headers.get("Content-Type", "")
            declared_encoding = response.encoding if "charset" in content_type.lower() else None
            soup = BeautifulSoup(
                response.content, "lxml", from_encoding=declared_encoding
            )

            soup = clean_soup(soup)

            content = get_text_from_soup(soup)

            image_urls = get_relevant_images(soup, self.link)

            # 使用工具函数提取标题
            title = extract_title(soup)

            return content, image_urls, title

        except Exception as e:
            logger.error(f"Error parsing {self.link}: {e}")
            return "", [], ""

    def _fetch(self):
        """GET 页面，瞬态失败时重试一次。

        成功时返回响应；页面不可达、状态为错误、或大到不值得
        解析时返回 None。
        """
        if self.session is None:
            logger.warning(f"No session provided for {self.link}; cannot fetch")
            return None

        self.failure_reason = None
        retry_with_transparent_identity = False
        for attempt in (1, 2):
            request_kwargs = {"timeout": 10}
            if retry_with_transparent_identity:
                request_kwargs["headers"] = {
                    "User-Agent": TRANSPARENT_RESEARCH_USER_AGENT,
                }
            try:
                response = self.session.get(self.link, **request_kwargs)
            except Exception as e:
                logger.warning(f"Request failed for {self.link} (attempt {attempt}): {e}")
                if attempt == 1:
                    time.sleep(1)
                    continue
                return None

            if response.status_code in RETRYABLE_STATUS_CODES and attempt == 1:
                logger.warning(
                    f"Got HTTP {response.status_code} for {self.link}, retrying once"
                )
                time.sleep(1)
                continue

            if response.status_code == 403 and attempt == 1:
                logger.warning(
                    f"Got HTTP 403 for {self.link}, retrying once with a "
                    "transparent GPT Researcher identity"
                )
                retry_with_transparent_identity = True
                continue

            if response.status_code >= 400:
                # 不要把错误/付费墙页面当作内容解析
                self.failure_reason = f"HTTP_{response.status_code}"
                logger.warning(f"Got HTTP {response.status_code} for {self.link}, skipping")
                return None

            content_length = response.headers.get("Content-Length")
            if content_length:
                try:
                    length_val = int(str(content_length).strip())
                except (TypeError, ValueError):
                    length_val = None
                if length_val is not None and length_val > MAX_CONTENT_BYTES:
                    logger.warning(
                        f"Content too large for {self.link} ({content_length} bytes), skipping"
                    )
                    return None

            return response

        return None
