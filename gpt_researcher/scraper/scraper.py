"""GPT Researcher 的网页抓取模块。

此模块提供 Scraper 类，使用各种抓取后端
（BeautifulSoup、PyMuPDF、Browser 等）从 URL 提取内容。
"""

import asyncio
import importlib
import logging
import subprocess
import sys
from urllib.parse import urlparse

import requests
from colorama import Fore, init

from gpt_researcher.observability import emit_research_event
from gpt_researcher.utils.workers import WorkerPool
from gpt_researcher.utils.url_security import UnsafeURLError, validate_url

from . import (
    ArxivScraper,
    BeautifulSoupScraper,
    BrowserScraper,
    FireCrawl,
    NoDriverScraper,
    PyMuPDFScraper,
    TavilyExtract,
    WebBaseLoaderScraper,
)

# 已知的反爬/挑战页标记，不区分大小写的子串匹配。
# 这些页面返回 HTTP 200 和真实（通常很大）的 HTML 体，因此
# 异常和下方的短内容检查都捕获不到它们 —— 没有此检查，
# 拦截/挑战页会被当作文章的真实内容摄入。每个标记都锚定得
# 足够具体，避免与普通文字冲突（如 "researchgate - temporarily
# unavailable"，而不是裸短语 "temporarily unavailable"，后者完全可能
# 出现在一篇关于无关故障的正当文章里）。
# 并不穷尽；实践中观察到新的拦截器时继续添加。
_BLOCK_PAGE_MARKERS = (
    "anubis uses a proof-of-work scheme",  # HAL 和其他 Anubis 前置的站点
    "making sure you're not a bot",
    "checking your browser before accessing",
    "enable javascript and cookies to continue",
    "researchgate - temporarily unavailable",  # ResearchGate 的特定措辞
    "please verify you are a human",
    "attention required! | cloudflare",
    "sorry, you have been blocked",
    # "Client Challenge" 插页（如 wikiHow、Clas Ohlson）：代替页面
    # 提供给纯 HTTP 客户端，约 300 字符，因此能通过 100 字符下限。
    # 其句子各占一行，因此标记是一句话，两种撇号风格都有。
    "a required part of this site couldn\u2019t load",
    "a required part of this site couldn't load",
)

# 拦截页就是整个响应体 —— 一条"请稍候"消息，而非真实文章 ——
# 因此总是很短，且总是出现在内容的最开头。只检查前缀，
# 避免每次抓取都对多兆字节的正当文档做小写化和扫描。
_BLOCK_PAGE_CHECK_PREFIX_LEN = 5_000

# 词表/词汇表转储（不相关单词的纯列表，无散文）能干净地抓取，
# 并可能主导报告的上下文，因为它们在词法上几乎匹配任何查询。
# 相对于其大小，它们几乎没有句末标点，与任何真实散文不同
# （即使是密集的技术写作每约 100-200 字符也有一个句号）。
# 不单独使用大小作为信号 —— 长的正当文档是存在的 —— 只检查
# 句子结构近乎完全缺失且确实很大这一组合，以保持对真实内容的
# 低误报率。在拉丁句末符之外包含 CJK 全角句末符（。！？），
# 使长篇中文/日文/韩文散文 —— 从不使用 "." "!" "?" —— 不会仅因
# 缺少 ASCII 标点而被误判为词表。
_MIN_LENGTH_FOR_WORDLIST_CHECK = 200_000
_MAX_SENTENCE_DENSITY = 1 / 5000  # 每 5000 字符至多 1 个句末标记
_SENTENCE_ENDING_CHARS = frozenset(".!?。！？")


def _looks_like_block_page(text: str) -> bool:
    prefix = text[:_BLOCK_PAGE_CHECK_PREFIX_LEN].lower()
    return any(marker in prefix for marker in _BLOCK_PAGE_MARKERS)


def _looks_like_word_list(text: str) -> bool:
    if len(text) < _MIN_LENGTH_FOR_WORDLIST_CHECK:
        return False
    sentence_endings = sum(1 for ch in text if ch in _SENTENCE_ENDING_CHARS)
    return (sentence_endings / len(text)) < _MAX_SENTENCE_DENSITY


# get_scraper() 根据 URL 后缀选择后端，但机构知识库（DSpace/EPrints
# 风格的下载端点）提供的 PDF 按内容类型路由，而非文件扩展名 —— 例如
# "scholarspace.manoa.hawaii.edu/bitstreams/<uuid>/download" 没有后缀。
# 这些会落入文本/HTML 抓取器，它无法解码 PDF 体，会把原始字节
# （FlateDecode 流、xref 表、/Annot 对象）当作页面的真实文本返回 ——
# 而且是静默的：没有异常，content_length 看起来正常，只是内容不可用。
#
# 这些标记是 PDF 自身的内部结构语法；它们几乎从不出现在真实散文中，
# 即使偶然出现也只是孤立的词。两个独立命中就足以确信这是原始 PDF，
# 而非恰好包含其中一个词的文本。
_PDF_STRUCTURE_MARKERS = ("endobj", "endstream", "/FlateDecode", "xref", "trailer")
_MIN_PDF_MARKER_HITS = 2


def _looks_like_unextracted_pdf(text: str) -> bool:
    if text.startswith("%PDF-"):
        return True
    hits = sum(1 for marker in _PDF_STRUCTURE_MARKERS if marker in text)
    return hits >= _MIN_PDF_MARKER_HITS


class Scraper:
    """
    从链接提取内容的 Scraper 类
    """

    def __init__(
        self,
        urls,
        user_agent,
        scraper,
        worker_pool: WorkerPool,
        event_handler=None,
    ):
        """
        初始化 Scraper 类。
        参数：
            urls: 要抓取的 URL 列表（重复项将被移除）
        """
        # 优化：移除重复 URL 以避免冗余抓取
        unique_urls = list(dict.fromkeys(urls))  # 去重的同时保持顺序
        duplicates_removed = len(urls) - len(unique_urls)

        self.urls = unique_urls
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": user_agent})
        self.scraper = scraper
        if self.scraper == "tavily_extract":
            self._check_pkg(self.scraper)
        if self.scraper == "firecrawl":
            self._check_pkg(self.scraper)
        self.logger = logging.getLogger(__name__)
        self.worker_pool = worker_pool
        self.event_handler = event_handler

        # 如果发现重复则记录去重结果
        if duplicates_removed > 0:
            self.logger.info(
                f"Removed {duplicates_removed} duplicate URL(s). "
                f"Scraping {len(unique_urls)} unique URLs instead of {len(urls)}."
            )

    async def run(self):
        """
        从链接提取内容
        """
        contents = await asyncio.gather(
            *(self.extract_data_from_url(url, self.session) for url in self.urls)
        )

        for content in contents:
            if not isinstance(content, dict):
                continue
            raw_content = content.get("raw_content")
            event = "fetch_succeeded" if raw_content is not None else "fetch_failed"
            fields = {
                "url": content.get("url") or "",
                "scraper": self.scraper,
                "content_length": len(raw_content or ""),
            }
            if raw_content is None:
                fields["reason_code"] = (
                    content.get("failure_reason") or "CONTENT_UNAVAILABLE"
                )
            await emit_research_event(
                self.event_handler,
                event,
                **fields,
            )

        # extract_data_from_url 预期返回字典，但有 bug 的后端或被取消的
        # 工作进程仍可能产出 None / 非字典。用 content["raw_content"]
        # 索引它们会让整个 gather 结果崩溃并丢弃好的行。
        res = [
            content
            for content in contents
            if isinstance(content, dict) and content.get("raw_content") is not None
        ]
        return res

    def _check_pkg(self, scrapper_name: str) -> None:
        """
        检查并确保需要 requirements.txt 之外依赖的抓取器所需的 Python 包可用。
        向仓库添加新抓取器时，用其所需信息更新 `pkg_map` 并在初始化时调用 check_pkg()。
        """
        pkg_map = {
            "tavily_extract": {
                "package_installation_name": "tavily-python",
                "import_name": "tavily",
            },
            "firecrawl": {
                "package_installation_name": "firecrawl-py",
                "import_name": "firecrawl",
            },
        }
        pkg = pkg_map[scrapper_name]
        if not importlib.util.find_spec(pkg["import_name"]):
            pkg_inst_name = pkg["package_installation_name"]
            init(autoreset=True)
            print(Fore.YELLOW + f"{pkg_inst_name} not found. Attempting to install...")
            try:
                subprocess.check_call(
                    [sys.executable, "-m", "pip", "install", pkg_inst_name]
                )
                importlib.invalidate_caches()
                print(Fore.GREEN + f"{pkg_inst_name} installed successfully.")
            except subprocess.CalledProcessError:
                raise ImportError(
                    Fore.RED
                    + f"Unable to install {pkg_inst_name}. Please install manually with "
                    f"`pip install -U {pkg_inst_name}`"
                )

    async def extract_data_from_url(self, link, session):
        """
        带日志地从链接提取数据
        """
        async with self.worker_pool.throttle():
            try:
                # 在发出任何请求之前拒绝 SSRF / 本地文件目标
                # （内部主机、云元数据端点、file:// 路径等）。
                try:
                    validate_url(link)
                except UnsafeURLError as e:
                    self.logger.warning(f"Skipping unsafe URL {link}: {e}")
                    return {
                        "url": link,
                        "raw_content": None,
                        "image_urls": [],
                        "title": "",
                    }

                Scraper = self.get_scraper(link)
                scraper = Scraper(link, session)

                # 获取抓取器名称
                scraper_name = scraper.__class__.__name__
                self.logger.info(f"\n=== Using {scraper_name} ===")

                # 获取内容
                if hasattr(scraper, "scrape_async"):
                    content, image_urls, title = await scraper.scrape_async()
                else:
                    (
                        content,
                        image_urls,
                        title,
                    ) = await asyncio.get_running_loop().run_in_executor(
                        self.worker_pool.executor, scraper.scrape
                    )

                if not content or len(content) < 100:
                    reason = getattr(scraper, "failure_reason", None)
                    return self._reject(
                        link, title, reason or "Content too short or empty"
                    )

                # 记录结果
                self.logger.info(f"\nTitle: {title}")
                self.logger.info(f"Content length: {len(content)} characters")
                self.logger.info(f"Number of images: {len(image_urls)}")
                self.logger.info(f"URL: {link}")
                self.logger.info("=" * 50)

                if _looks_like_block_page(content):
                    return self._reject(link, title, "Anti-bot/challenge page detected")

                if _looks_like_word_list(content):
                    return self._reject(link, title, "Word-list-like content detected")

                if _looks_like_unextracted_pdf(content) and Scraper is not PyMuPDFScraper:
                    self.logger.warning(
                        f"{link} looks like unextracted PDF binary via {scraper_name} "
                        f"(no .pdf suffix, so get_scraper() didn't route it to "
                        f"PyMuPDFScraper) -- retrying with PyMuPDFScraper"
                    )
                    retried = await self._retry_as_pdf(link, session)
                    if retried is not None:
                        return retried
                    self.logger.warning(f"PyMuPDFScraper retry also failed for {link}")
                    return {
                        "url": link,
                        "raw_content": None,
                        "image_urls": [],
                        "title": title,
                    }

                return {
                    "url": link,
                    "raw_content": content,
                    "image_urls": image_urls,
                    "title": title,
                }

            except Exception as e:
                self.logger.error(f"Error processing {link}: {str(e)}")
                return {"url": link, "raw_content": None, "image_urls": [], "title": ""}

    def _reject(self, link, title, reason):
        """将已抓取但不可用的页面视为抓取失败，返回与抛出异常或
        内容过短时相同的形状 —— 下游代码（Scraper.run）会丢弃任何
        raw_content: None 的条目。

        ``failure_reason`` 是附加元数据，供在该过滤步骤之前检查
        失败行的调用方使用。
        """
        self.logger.warning(f"{reason} for {link}, treating as fetch failure")
        return {
            "url": link,
            "raw_content": None,
            "image_urls": [],
            "title": title,
            "failure_reason": reason,
        }


    async def _retry_as_pdf(self, link, session):
        """在最初选择的抓取器 get_scraper() 返回未提取的 PDF 二进制后，
        用 PyMuPDFScraper 重新抓取链接。

        成功时返回正常的 extract_data_from_url 结果字典；重试也无法
        产出可用内容时返回 None（因此调用方回退到将其视为普通抓取
        失败，而不是让二进制通过）。
        """
        scraper = PyMuPDFScraper(link, session)
        content, image_urls, title = await asyncio.get_running_loop().run_in_executor(
            self.worker_pool.executor, scraper.scrape
        )
        if not content or len(content) < 100:
            return None
        if (
            _looks_like_unextracted_pdf(content)
            or _looks_like_block_page(content)
            or _looks_like_word_list(content)
        ):
            self.logger.warning(
                f"PyMuPDFScraper retry for {link} recovered content that still "
                f"fails quality checks -- treating as failed recovery"
            )
            return None
        self.logger.info(f"PyMuPDFScraper retry recovered {len(content)} characters for {link}")
        return {
            "url": link,
            "raw_content": content,
            "image_urls": image_urls,
            "title": title,
        }

    def get_scraper(self, link):
        """
        函数 `get_scraper` 根据提供的链接确定合适的抓取器类，
        无匹配时使用默认抓取器。

        参数：
          link: `get_scraper` 方法接受一个 `link` 参数，即指向网页或
        PDF 文件的 URL 链接。根据链接指向的内容类型，该方法确定用于
        从该内容提取数据的合适抓取器类。

        返回：
          `get_scraper` 方法根据提供的链接返回抓取器类。该方法检查
        链接，依据 `SCRAPER_CLASSES` 字典中的预定义映射确定合适的
        抓取器类。如果链接以 ".pdf" 结尾，选择 `PyMuPDFScraper` 类。
        arxiv.org 或其子域托管的 URL 选择 `ArxivScraper`。
        """

        SCRAPER_CLASSES = {
            "pdf": PyMuPDFScraper,
            "arxiv": ArxivScraper,
            "bs": BeautifulSoupScraper,
            "web_base_loader": WebBaseLoaderScraper,
            "browser": BrowserScraper,
            "nodriver": NoDriverScraper,
            "tavily_extract": TavilyExtract,
            "firecrawl": FireCrawl,
        }

        scraper_key = None

        # 只检查路径组件，使查询字符串/片段不会隐藏扩展名
        # （如带签名的 CDN/S3 链接 "…/doc.pdf?sig=…"）。
        # 不区分大小写匹配，因为 ".PDF" 是完全有效的后缀。
        parsed_url = urlparse(link)
        path = parsed_url.path
        # 尾部的 DNS 根点仍标识同一主机。
        hostname = (parsed_url.hostname or "").removesuffix(".")
        if path.lower().endswith(".pdf"):
            scraper_key = "pdf"
        elif hostname == "arxiv.org" or hostname.endswith(".arxiv.org"):
            scraper_key = "arxiv"
        else:
            scraper_key = self.scraper

        scraper_class = SCRAPER_CLASSES.get(scraper_key)
        if scraper_class is None:
            raise Exception("Scraper not found.")

        return scraper_class
