"""
MCP 客户端管理模块

处理 MCP 客户端创建、配置转换和连接管理。
"""
import asyncio
import logging
from typing import List, Dict, Any, Optional
from urllib.parse import urlsplit

try:
    from langchain_mcp_adapters.client import MultiServerMCPClient
    HAS_MCP_ADAPTERS = True
except ImportError:
    HAS_MCP_ADAPTERS = False

logger = logging.getLogger(__name__)


class MCPClientManager:
    """
    管理 MCP 客户端生命周期和配置。
    
    负责以下事项：
    - 将 GPT Researcher 的 MCP 配置转换为 langchain 格式
    - 创建和管理 MultiServerMCPClient 实例
    - 处理客户端清理和资源管理
    """

    def __init__(self, mcp_configs: List[Dict[str, Any]]):
        """
        初始化 MCP 客户端管理器。
        
        参数：
            mcp_configs: 来自 GPT Researcher 的 MCP 服务器配置列表
        """
        self.mcp_configs = mcp_configs or []
        self._client = None
        self._client_lock = asyncio.Lock()

    def convert_configs_to_langchain_format(self) -> Dict[str, Dict[str, Any]]:
        """
        将 GPT Researcher 的 MCP 配置转换为 langchain-mcp-adapters 格式。
        
        返回：
            Dict[str, Dict[str, Any]]: MultiServerMCPClient 的服务器配置
        """
        server_configs = {}
        
        for i, config in enumerate(self.mcp_configs):
            # GPT Researcher 的 MCP 配置是字典；容忍列表/json 漂移，
            # 使单个坏条目不会让整个转换抛出 AttributeError。
            if not isinstance(config, dict):
                logger.warning(
                    "Skipping MCP server config at index %s: expected dict, got %s",
                    i,
                    type(config).__name__,
                )
                continue
            # 生成服务器名称
            server_name = config.get("name", f"mcp_server_{i+1}")
            
            # 构建服务器配置
            server_config = {}
            
            # 如果提供了 URL 则自动检测传输类型。URI 协议名不区分
            # 大小写，因此只规范化解析出的协议，并为下游客户端
            # 保留原始 URL。
            connection_url = config.get("connection_url")
            if connection_url:
                scheme = urlsplit(connection_url).scheme.lower()
                if scheme in {"ws", "wss"}:
                    server_config["transport"] = "websocket"
                    server_config["url"] = connection_url
                elif scheme in {"http", "https"}:
                    server_config["transport"] = "streamable_http"
                    server_config["url"] = connection_url
                else:
                    # 回退到指定的 connection_type 或 stdio
                    connection_type = config.get("connection_type", "stdio")
                    server_config["transport"] = connection_type
                    if connection_type in ["websocket", "streamable_http", "http"]:
                        server_config["url"] = connection_url
            else:
                # 未提供 URL，使用 stdio（默认）或指定的 connection_type
                connection_type = config.get("connection_type", "stdio")
                server_config["transport"] = connection_type
            
            if server_config.get("transport") in ["streamable_http", "http", "websocket"]:
                connection_headers = config.get("connection_headers")
                if connection_headers and isinstance(connection_headers, dict):
                    server_config["headers"] = connection_headers
            
            # 处理 stdio 传输配置
            if server_config.get("transport") == "stdio":
                if config.get("command"):
                    server_config["command"] = config["command"]
                    
                    # 处理 server_args
                    server_args = config.get("args", [])
                    if isinstance(server_args, str):
                        server_args = server_args.split()
                    server_config["args"] = server_args
                    
                    # 处理环境变量
                    server_env = config.get("env", {})
                    if server_env:
                        server_config["env"] = server_env
                        
            # 如果提供了认证信息则添加
            if config.get("connection_token"):
                server_config["token"] = config["connection_token"]
                
            server_configs[server_name] = server_config
            
        return server_configs

    async def get_or_create_client(self) -> Optional[object]:
        """
        获取或创建 MultiServerMCPClient，并做妥善的生命周期管理。
        
        返回：
            MultiServerMCPClient: 客户端实例；创建失败时返回 None
        """
        async with self._client_lock:
            if self._client is not None:
                return self._client
                
            if not HAS_MCP_ADAPTERS:
                logger.error("langchain-mcp-adapters not installed")
                return None
                
            if not self.mcp_configs:
                logger.error("No MCP server configurations found")
                return None
                
            try:
                # 将配置转换为 langchain 格式
                server_configs = self.convert_configs_to_langchain_format()
                logger.info(f"Creating MCP client for {len(server_configs)} server(s)")
                
                # 初始化 MultiServerMCPClient
                self._client = MultiServerMCPClient(server_configs)
                
                return self._client
                
            except Exception as e:
                logger.error(f"Error creating MCP client: {e}")
                return None

    async def close_client(self):
        """
        正确关闭 MCP 客户端并清理资源。
        """
        async with self._client_lock:
            if self._client is not None:
                try:
                    # 由于 langchain-mcp-adapters 0.1.0 中的 MultiServerMCPClient
                    # 不支持上下文管理器或显式关闭方法，
                    # 我们只清除引用，让垃圾回收处理
                    logger.debug("Releasing MCP client reference")
                except Exception as e:
                    logger.error(f"Error during MCP client cleanup: {e}")
                finally:
                    # 始终清除引用
                    self._client = None

    async def get_all_tools(self) -> List:
        """
        获取 MCP 服务器的所有可用工具。
        
        返回：
            List: 所有可用的 MCP 工具
        """
        client = await self.get_or_create_client()
        if not client:
            return []
            
        try:
            # 从所有服务器获取工具
            all_tools = await client.get_tools()
            
            if all_tools:
                logger.info(f"Loaded {len(all_tools)} total tools from MCP servers")
                return all_tools
            else:
                logger.warning("No tools available from MCP servers")
                return []
                
        except Exception as e:
            logger.error(f"Error getting MCP tools: {e}")
            return [] 
