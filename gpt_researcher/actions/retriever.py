"""GPT Researcher 的检索器工厂与工具。

此模块提供实例化和管理各种搜索检索器实现的函数。
"""

import logging
from importlib.metadata import entry_points

logger = logging.getLogger(__name__)

#: 第三方包用来注册检索器的入口点组。
RETRIEVER_ENTRY_POINT_GROUP = "gpt_researcher.retrievers"


def get_retriever(retriever: str):
    """按名称获取检索器类。

    参数：
        retriever: 要获取的检索器名称（如 'google'、'tavily'、'duckduckgo'）。

    返回：
        找到时返回检索器类，否则返回 None。

    支持的检索器：
        - google: Google 自定义搜索
        - searx: SearX 搜索引擎
        - searchapi: SearchAPI 服务
        - serpapi: SerpAPI 服务
        - serper: Serper API
        - duckduckgo: DuckDuckGo 搜索
        - bing: Bing 搜索
        - bocha: BoCha 搜索 API
        - brave: Brave 搜索 API
        - arxiv: arXiv 学术搜索
        - tavily: Tavily 搜索 API
        - exa: Exa 搜索
        - crw: fastCRW 搜索（Firecrawl 兼容的网页抓取器）
        - semantic_scholar: Semantic Scholar 学术搜索
        - pubmed_central: PubMed Central 医学文献
        - openalex: OpenAlex 学术著作目录
        - custom: 用户自定义检索器
        - mcp: Model Context Protocol 检索器
        - xquik: Xquik X/Twitter 搜索
        - getxapi: GetXAPI X/Twitter 搜索

    其他名称会在已安装插件中查找，这些插件注册在
    ``gpt_researcher.retrievers`` 入口点组下。内置名称始终优先。
    """
    match retriever:
        case "google":
            from gpt_researcher.retrievers import GoogleSearch

            return GoogleSearch
        case "searx":
            from gpt_researcher.retrievers import SearxSearch

            return SearxSearch
        case "searchapi":
            from gpt_researcher.retrievers import SearchApiSearch

            return SearchApiSearch
        case "serpapi":
            from gpt_researcher.retrievers import SerpApiSearch

            return SerpApiSearch
        case "serper":
            from gpt_researcher.retrievers import SerperSearch

            return SerperSearch
        case "duckduckgo":
            from gpt_researcher.retrievers import Duckduckgo

            return Duckduckgo
        case "bing":
            from gpt_researcher.retrievers import BingSearch

            return BingSearch
        case "brave":
            from gpt_researcher.retrievers import BraveSearch

            return BraveSearch
        case "bocha":
            from gpt_researcher.retrievers import BoChaSearch

            return BoChaSearch
        case "arxiv":
            from gpt_researcher.retrievers import ArxivSearch

            return ArxivSearch
        case "tavily":
            from gpt_researcher.retrievers import TavilySearch

            return TavilySearch
        case "groundroute":
            from gpt_researcher.retrievers import GroundRouteSearch

            return GroundRouteSearch
        case "exa":
            from gpt_researcher.retrievers import ExaSearch

            return ExaSearch
        case "crw":
            from gpt_researcher.retrievers import CRWRetriever

            return CRWRetriever
        case "semantic_scholar":
            from gpt_researcher.retrievers import SemanticScholarSearch

            return SemanticScholarSearch
        case "pubmed_central":
            from gpt_researcher.retrievers import PubMedCentralSearch

            return PubMedCentralSearch
        case "custom":
            from gpt_researcher.retrievers import CustomRetriever

            return CustomRetriever
        case "mcp":
            from gpt_researcher.retrievers import MCPRetriever

            return MCPRetriever
        case "xquik":
            from gpt_researcher.retrievers import XquikSearch

            return XquikSearch
        case "openalex":
            from gpt_researcher.retrievers import OpenAlexSearch

            return OpenAlexSearch
        case "getxapi":
            from gpt_researcher.retrievers import GetXAPISearch

            return GetXAPISearch

        case _:
            return _load_plugin_retriever(retriever)


def _load_plugin_retriever(name: str):
    """加载已安装包按名称注册的检索器类。"""
    for entry_point in entry_points(group=RETRIEVER_ENTRY_POINT_GROUP, name=name):
        try:
            return entry_point.load()
        except Exception as exc:
            logger.warning(f"Failed to load retriever plugin '{name}' ({entry_point.value}): {exc}")
            return None
    return None


def get_retrievers(headers: dict[str, str], cfg):
    """
    根据请求头、配置或默认值确定要使用的检索器。

    参数：
        headers (dict): 请求头字典
        cfg: 配置对象

    返回：
        list: 用于搜索的检索器类列表。
    """
    # 先检查请求头中的多个检索器
    if headers.get("retrievers"):
        retrievers = headers.get("retrievers").split(",")
    # 未找到时检查请求头中的单个检索器
    elif headers.get("retriever"):
        retrievers = [headers.get("retriever")]
    # 请求头没有时检查配置中的多个检索器
    elif cfg.retrievers:
        # 配置中的检索器同时兼容列表和字符串格式
        if isinstance(cfg.retrievers, str):
            retrievers = cfg.retrievers.split(",")
        else:
            retrievers = cfg.retrievers
    # 未找到时检查配置中的单个检索器
    elif cfg.retriever:
        retrievers = [cfg.retriever]
    # 仍未设置时使用默认检索器
    else:
        retrievers = [get_default_retriever().__name__]

    # 去除每个检索器名称的空白，使带空格的逗号分隔列表
    # （如来自请求头或配置的 "tavily, exa"）能正确解析，
    # 而不是静默回退到默认检索器。
    retrievers = [r.strip() for r in retrievers if r and r.strip()]

    if not retrievers:
        return [get_default_retriever()]

    # 将检索器名称转换为实际的检索器类
    # 对无效的检索器名称使用 get_default_retriever() 作为回退
    retriever_classes = [get_retriever(r) or get_default_retriever() for r in retrievers]
    
    return retriever_classes


def get_default_retriever():
    """获取默认检索器类。

    返回：
        作为默认搜索提供者的 TavilySearch 检索器类。
    """
    from gpt_researcher.retrievers import TavilySearch

    return TavilySearch
