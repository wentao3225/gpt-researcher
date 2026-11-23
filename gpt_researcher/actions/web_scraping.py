from typing import Any
from colorama import Fore, Style

from gpt_researcher.utils.workers import WorkerPool
from ..scraper import Scraper
from ..config.config import Config
from ..utils.logger import get_formatted_logger

logger = get_formatted_logger()


async def scrape_urls(
    urls, cfg: Config, worker_pool: WorkerPool, event_handler=None
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """
    抓取 URL
    参数：
        urls: URL 列表
        cfg: 配置（可选）

    返回：
        tuple[list[dict[str, Any]], list[dict[str, Any]]]: 包含抓取内容和图片的元组

    """
    scraped_data = []
    images = []
    user_agent = (
        cfg.user_agent
        if cfg
        else "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36"
    )

    scraper = None
    try:
        scraper = Scraper(
            urls,
            user_agent,
            cfg.scraper,
            worker_pool=worker_pool,
            event_handler=event_handler,
        )
        scraped_data = await scraper.run()
        for item in scraped_data:
            if not isinstance(item, dict):
                continue
            image_urls = item.get("image_urls")
            if image_urls:
                images.extend(image_urls)
    except Exception as e:
        print(f"{Fore.RED}Error in scrape_urls: {e}{Style.RESET_ALL}")
    finally:
        # 关闭 requests.Session，释放其底层连接池（及其保持存活的套接字）
        if scraper is not None and getattr(scraper, "session", None) is not None:
            scraper.session.close()

    return scraped_data, images


async def filter_urls(urls: list[str], config: Config) -> list[str]:
    """
    根据配置设置过滤 URL。

    参数：
        urls (list[str]): 要过滤的 URL 列表。
        config (Config): 配置对象。

    返回：
        list[str]: 过滤后的 URL 列表。
    """
    filtered_urls = []
    excluded = getattr(config, "excluded_domains", None) or []
    if not isinstance(excluded, (list, tuple, set)):
        excluded = []
    for url in urls or []:
        # URL 必须是非空字符串；null/int 噪声会让排除域名的
        # 子串检查抛出 TypeError。
        if not isinstance(url, str) or not url:
            continue
        if not any(isinstance(ex, str) and ex and ex in url for ex in excluded):
            filtered_urls.append(url)
    return filtered_urls

async def extract_main_content(html_content: str) -> str:
    """
    从 HTML 中提取主要内容。

    参数：
        html_content (str): 原始 HTML 内容。

    返回：
        str: 提取的主要内容。
    """
    # 在此实现内容提取逻辑
    # 可以使用 BeautifulSoup 之类的库或自定义解析逻辑
    # 目前仅返回原始 HTML 作为占位
    return html_content

async def process_scraped_data(scraped_data: list[dict[str, Any]], config: Config) -> list[dict[str, Any]]:
    """
    处理抓取的数据以提取并清理主要内容。

    参数：
        scraped_data (list[dict[str, Any]]): 包含抓取数据的字典列表。
        config (Config): 配置对象。

    返回：
        list[dict[str, Any]]: 处理后的抓取数据。
    """
    processed_data = []
    for item in scraped_data:
        if not isinstance(item, dict):
            continue
        # 部分抓取器载荷历史上使用严格的键访问，会在批次中途崩溃；
        # 跳过/重新发出的保护措施让其余运行得以继续。
        status = item.get("status")
        if status == "success":
            main_content = await extract_main_content(item.get("content") or "")
            processed_data.append({
                "url": item.get("url") or "",
                "content": main_content,
                "status": "success",
            })
        else:
            processed_data.append(item)
    return processed_data
