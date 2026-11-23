"""GPT Researcher 检索器的工具函数。

此模块提供各种搜索检索器实现使用的辅助函数和常量。
"""

import importlib.util
import logging
import os
import sys

logger = logging.getLogger(__name__)

async def stream_output(log_type, step, content, websocket=None, with_data=False, data=None):
    """
    将输出流式传输到客户端。
    
    参数：
        log_type (str): 日志类型
        step (str): 正在执行的步骤
        content (str): 要流式传输的内容
        websocket: 要流式传输到的 websocket
        with_data (bool): 是否包含数据
        data: 额外的数据
    """
    if websocket:
        try:
            if with_data:
                await websocket.send_json({
                    "type": log_type,
                    "step": step,
                    "content": content,
                    "data": data
                })
            else:
                await websocket.send_json({
                    "type": log_type,
                    "step": step,
                    "content": content
                })
        except Exception as e:
            logger.error(f"Error streaming output: {e}")

def check_pkg(pkg: str) -> None:
    """
    检查包是否已安装，未安装则抛出错误。
    
    参数：
        pkg (str): 包名
    
    抛出：
        ImportError: 如果包未安装
    """
    if not importlib.util.find_spec(pkg):
        pkg_kebab = pkg.replace("_", "-")
        raise ImportError(
            f"Unable to import {pkg_kebab}. Please install with "
            f"`pip install -U {pkg_kebab}`"
        )

# 用于回退的有效检索器
VALID_RETRIEVERS = [
    "tavily",
    "groundroute",
    "custom",
    "duckduckgo",
    "searchapi",
    "serper",
    "serpapi",
    "google",
    "searx",
    "bing",
    "brave",
    "arxiv",
    "semantic_scholar",
    "pubmed_central",
    "exa",
    "crw",
    "getxapi",
    "mcp",
    "xquik",
    "openalex",
    "mock"
]

def get_all_retriever_names():
    """
    获取所有可用的检索器名称
    :return: 所有可用检索器名称的列表
    :rtype: list
    """
    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        
        # 获取当前目录中的所有条目
        all_items = os.listdir(current_dir)
        
        # 只过滤目录，排除 __pycache__
        retrievers = [
            item for item in all_items 
            if os.path.isdir(os.path.join(current_dir, item)) and not item.startswith('__')
        ]
    except Exception as e:
        logger.error(f"Error getting retrievers: {e}")
        retrievers = list(VALID_RETRIEVERS)

    return retrievers + [name for name in get_plugin_retriever_names() if name not in retrievers]


def get_plugin_retriever_names():
    """已安装包注册为插件的检索器名称。"""
    from importlib.metadata import entry_points

    from gpt_researcher.actions.retriever import RETRIEVER_ENTRY_POINT_GROUP

    try:
        return sorted({ep.name for ep in entry_points(group=RETRIEVER_ENTRY_POINT_GROUP)})
    except Exception as e:
        logger.error(f"Error discovering retriever plugins: {e}")
        return []
