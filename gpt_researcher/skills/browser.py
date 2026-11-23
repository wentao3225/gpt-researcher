"""GPT Researcher 的浏览器管理器技能。

此模块提供 BrowserManager 类，处理网页抓取
和从 URL 提取内容。
"""

from gpt_researcher.utils.workers import WorkerPool

from ..actions.utils import stream_output
from ..actions.web_scraping import scrape_urls
from ..scraper.utils import get_image_hash


class BrowserManager:
    """管理研究的网页浏览和内容抓取。

    此类处理 URL 抓取、内容提取，以及研究过程
    中的图片选择。

    属性：
        researcher: 上级 GPTResearcher 实例。
        worker_pool: 并行抓取的工作进程池。
    """

    def __init__(self, researcher):
        """初始化 BrowserManager。

        参数：
            researcher: 拥有此管理器的 GPTResearcher 实例。
        """
        self.researcher = researcher
        self.worker_pool = WorkerPool(
            researcher.cfg.max_scraper_workers,
            researcher.cfg.scraper_rate_limit_delay
        )

    async def browse_urls(self, urls: list[str]) -> list[dict]:
        """
        从 URL 列表抓取内容。

        参数：
            urls (list[str]): 要抓取的 URL 列表。

        返回：
            list[dict]: 抓取内容结果列表。
        """
        if self.researcher.verbose:
            await stream_output(
                "logs",
                "scraping_urls",
                f"🌐 Scraping content from {len(urls)} URLs...",
                self.researcher.websocket,
            )

        scraped_content, images = await scrape_urls(
            urls,
            self.researcher.cfg,
            self.worker_pool,
            event_handler=getattr(self.researcher, "research_observer", None),
        )
        self.researcher.add_research_sources(scraped_content)
        new_images = self.select_top_images(images, k=4)  # 选择前 4 张图片
        self.researcher.add_research_images(new_images)

        if self.researcher.verbose:
            await stream_output(
                "logs",
                "scraping_content",
                f"📄 Scraped {len(scraped_content)} pages of content",
                self.researcher.websocket,
            )
            await stream_output(
                "logs",
                "scraping_images",
                f"🖼️ Selected {len(new_images)} new images from {len(images)} total images",
                self.researcher.websocket,
                True,
                new_images,
            )
            await stream_output(
                "logs",
                "scraping_complete",
                f"🌐 Scraping complete",
                self.researcher.websocket,
            )

        return scraped_content

    def select_top_images(self, images: list[dict], k: int = 2) -> list[str]:
        """
        基于图片内容选择最相关的图片并去重。

        参数：
            images (list[dict]): 带 'url' 和 'score' 键的图片字典列表。
            k (int): 如果没有找到高分图片时选择的顶部图片数量。

        返回：
            list[str]: 选定图片 URL 列表。
        """
        unique_images = []
        seen_hashes = set()
        current_research_images = self.researcher.get_research_images()

        # 按分数降序处理图片
        for img in sorted(images, key=lambda im: im["score"], reverse=True):
            img_hash = get_image_hash(img['url'])
            if (
                img_hash
                and img_hash not in seen_hashes
                and img['url'] not in current_research_images
            ):
                seen_hashes.add(img_hash)
                unique_images.append(img["url"])

                if len(unique_images) == k:
                    break

        return unique_images
