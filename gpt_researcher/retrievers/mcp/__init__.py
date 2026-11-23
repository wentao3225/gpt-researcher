"""
MCP 检索器模块

此模块只包含 MCP 检索器实现。
核心 MCP 功能已移至 gpt_researcher.mcp 模块。
"""
import logging

logger = logging.getLogger(__name__)

try:
    # 检查 langchain-mcp-adapters 是否可用
    from langchain_mcp_adapters.client import MultiServerMCPClient
    HAS_MCP_ADAPTERS = True
    logger.debug("langchain-mcp-adapters is available")
    
    # 导入检索器
    from .retriever import MCPRetriever
    __all__ = ["MCPRetriever"]
    logger.debug("MCPRetriever imported successfully")
    
except ImportError as e:
    # 记录具体的导入错误以便调试
    logger.warning(f"Failed to import MCPRetriever: {e}")
    # MCP 包未安装或其他导入错误，提供占位符
    MCPRetriever = None
    __all__ = []
except Exception as e:
    # 捕获可能发生的任何其他异常
    logger.error(f"Unexpected error importing MCPRetriever: {e}")
    MCPRetriever = None
    __all__ = [] 