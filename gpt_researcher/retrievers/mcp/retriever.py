"""
基于 MCP 的研究检索器

使用 Model Context Protocol（MCP）工具进行智能研究的检索器。
此检索器实现两阶段方法：
1. 工具选择：LLM 从所有可用的 MCP 工具中选择 2-3 个最相关的工具
2. 研究执行：LLM 使用所选工具进行智能研究
"""
import asyncio
import logging
from typing import List, Dict, Any, Optional

try:
    from langchain_mcp_adapters.client import MultiServerMCPClient
    HAS_MCP_ADAPTERS = True
except ImportError:
    HAS_MCP_ADAPTERS = False

from ...mcp.client import MCPClientManager
from ...mcp.tool_selector import MCPToolSelector
from ...mcp.research import MCPResearchSkill
from ...mcp.streaming import MCPStreamer

logger = logging.getLogger(__name__)


class MCPRetriever:
    """
    GPT Researcher 的 Model Context Protocol（MCP）检索器。
    
    此检索器实现两阶段方法：
    1. 工具选择：LLM 从所有可用的 MCP 工具中选择 2-3 个最相关的工具
    2. 研究执行：带绑定工具的 LLM 进行智能研究
    
    这种方法比调用所有工具更高效，并提供更好、
    更有针对性的研究结果。
    
    此检索器需要研究员实例来访问：
    - mcp_configs: MCP 服务器配置列表
    - cfg: 包含 LLM 设置和参数的配置对象
    - add_costs: 跟踪研究成本的方法
    """

    def __init__(
        self, 
        query: str, 
        headers: Optional[Dict[str, str]] = None,
        query_domains: Optional[List[str]] = None,
        websocket=None,
        researcher=None,
        **kwargs
    ):
        """
        初始化 MCP 检索器。
        
        参数：
            query (str): 搜索查询字符串。
            headers (dict, 可选): 包含 MCP 配置的请求头。
            query_domains (list, 可选): 要搜索的域名列表（MCP 不使用）。
            websocket: 用于流式日志的 WebSocket。
            researcher: 包含 mcp_configs 和 cfg 的研究员实例。
            **kwargs: 额外参数（用于兼容性）。
        """
        self.query = query
        self.headers = headers or {}
        self.query_domains = query_domains or []
        self.websocket = websocket
        self.researcher = researcher
        
        # 从研究员实例提取 mcp_configs 和配置
        self.mcp_configs = self._get_mcp_configs()
        self.cfg = self._get_config()
        
        # 初始化模块化组件
        self.client_manager = MCPClientManager(self.mcp_configs)
        self.tool_selector = MCPToolSelector(self.cfg, self.researcher)
        self.mcp_researcher = MCPResearchSkill(self.cfg, self.researcher)
        self.streamer = MCPStreamer(self.websocket)
        
        # 初始化缓存
        self._all_tools_cache = None
        
        # 记录初始化
        if self.mcp_configs:
            self.streamer.stream_log_sync(f"🔧 Initializing MCP retriever for query: {self.query}")
            self.streamer.stream_log_sync(f"🔧 Found {len(self.mcp_configs)} MCP server configurations")
        else:
            logger.error("No MCP server configurations found. The retriever will fail during search.")
            self.streamer.stream_log_sync("❌ CRITICAL: No MCP server configurations found. Please check documentation.")

    def _get_mcp_configs(self) -> List[Dict[str, Any]]:
        """
        从研究员实例获取 MCP 配置。
        
        返回：
            List[Dict[str, Any]]: MCP 服务器配置列表。
        """
        if self.researcher and hasattr(self.researcher, 'mcp_configs'):
            return self.researcher.mcp_configs or []
        return []

    def _get_config(self):
        """
        从研究员实例获取配置。
        
        返回：
            Config: 包含 LLM 设置的配置对象。
        """
        if self.researcher and hasattr(self.researcher, 'cfg'):
            return self.researcher.cfg
        
        # If no config available, this is a critical error
        logger.error("No config found in researcher instance. MCPRetriever requires a researcher instance with cfg attribute.")
        raise ValueError("MCPRetriever requires a researcher instance with cfg attribute containing LLM configuration")

    async def search_async(self, max_results: int = 10) -> List[Dict[str, str]]:
        """
        使用 MCP 工具和智能两阶段方法执行异步搜索。
        
        参数：
            max_results: 返回结果的最大数量。
            
        返回：
            List[Dict[str, str]]: 搜索结果。
        """
        # 检查是否有服务器配置
        if not self.mcp_configs:
            error_msg = "No MCP server configurations available. Please provide mcp_configs parameter to GPTResearcher."
            logger.error(error_msg)
            await self.streamer.stream_error("MCP retriever cannot proceed without server configurations.")
            return []  # 返回空而不是抛出异常，让研究得以继续
            
        # 记录日志以帮助调试集成流程
        logger.info(f"MCPRetriever.search_async called for query: {self.query}")
            
        try:
            # 阶段 1：获取所有可用工具
            await self.streamer.stream_stage_start("Stage 1", "Getting all available MCP tools")
            all_tools = await self._get_all_tools()
            
            if not all_tools:
                await self.streamer.stream_warning("No MCP tools available, skipping MCP research")
                return []
            
            # 阶段 2：选择最相关的工具
            await self.streamer.stream_stage_start("Stage 2", "Selecting most relevant tools")
            selected_tools = await self.tool_selector.select_relevant_tools(self.query, all_tools, max_tools=3)
            
            if not selected_tools:
                await self.streamer.stream_warning("No relevant tools selected, skipping MCP research")
                return []
            
            # 阶段 3：使用所选工具进行研究
            await self.streamer.stream_stage_start("Stage 3", "Conducting research with selected tools")
            results = await self.mcp_researcher.conduct_research_with_tools(self.query, selected_tools)
            
            # 限制结果数量
            if len(results) > max_results:
                logger.info(f"Limiting {len(results)} MCP results to {max_results}")
                results = results[:max_results]
            
            # 记录带实际内容样本的结果摘要
            logger.info(f"MCPRetriever returning {len(results)} results")
            
            # 计算总内容长度用于摘要
            total_content_length = sum(len(result.get("body", "")) for result in results)
            await self.streamer.stream_research_results(len(results), total_content_length)
            
            # 记录详细内容样本以便调试
            if results:
                # 展示前几个结果的样本
                for i, result in enumerate(results[:3]):  # 展示前 3 个结果
                    title = result.get("title", "No title")
                    url = result.get("href", "No URL")
                    content = result.get("body", "")
                    content_length = len(content)
                    content_sample = content[:400] + "..." if len(content) > 400 else content
                    
                    logger.debug(f"Result {i+1}/{len(results)}: '{title}'")
                    logger.debug(f"URL: {url}")
                    logger.debug(f"Content ({content_length:,} chars): {content_sample}")
                    
                if len(results) > 3:
                    remaining_results = len(results) - 3
                    remaining_content = sum(len(result.get("body", "")) for result in results[3:])
                    logger.debug(f"... and {remaining_results} more results ({remaining_content:,} chars)")
                    
            return results
            
        except Exception as e:
            logger.error(f"Error in MCP search: {e}")
            await self.streamer.stream_error(f"Error in MCP search: {str(e)}")
            return []
        finally:
            # 搜索完成后确保客户端清理
            try:
                await self.client_manager.close_client()
            except Exception as e:
                logger.error(f"Error during client cleanup: {e}")

    def search(self, max_results: int = 10) -> List[Dict[str, str]]:
        """
        使用 MCP 工具和智能两阶段方法执行搜索。
        
        这是 GPT Researcher 要求的同步接口。
        它包装了异步的 search_async 方法。
        
        参数：
            max_results: 返回结果的最大数量。
            
        返回：
            List[Dict[str, str]]: 搜索结果。
        """
        # 检查是否有服务器配置
        if not self.mcp_configs:
            error_msg = "No MCP server configurations available. Please provide mcp_configs parameter to GPTResearcher."
            logger.error(error_msg)
            self.streamer.stream_log_sync("❌ MCP retriever cannot proceed without server configurations.")
            return []  # 返回空而不是抛出异常，让研究得以继续
            
        # 记录日志以帮助调试集成流程
        logger.info(f"MCPRetriever.search called for query: {self.query}")
        
        try:
            # 正确处理异步/同步边界
            try:
                # 尝试获取当前事件循环
                loop = asyncio.get_running_loop()
                # 如果处于异步上下文中，需要调度协程
                # 这有点棘手 —— 我们创建一个任务并让它运行
                import concurrent.futures
                import threading
                
                # 在单独的线程中创建新的事件循环
                def run_in_thread():
                    new_loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(new_loop)
                    try:
                        result = new_loop.run_until_complete(self.search_async(max_results))
                        return result
                    finally:
                        # 针对 MCP 连接的增强清理流程
                        try:
                            # 带超时地取消所有待处理任务
                            pending = asyncio.all_tasks(new_loop)
                            for task in pending:
                                task.cancel()
                            
                            # 带超时地等待已取消的任务完成
                            if pending:
                                try:
                                    new_loop.run_until_complete(
                                        asyncio.wait_for(
                                            asyncio.gather(*pending, return_exceptions=True),
                                            timeout=5.0  # 清理的 5 秒超时
                                        )
                                    )
                                except asyncio.TimeoutError:
                                    logger.debug("Timeout during task cleanup, continuing...")
                                except Exception:
                                    pass  # 忽略其他清理错误
                        except Exception:
                            pass  # 忽略清理错误
                        finally:
                            try:
                                # 给循环一点时间完成最后的清理
                                import time
                                time.sleep(0.1)
                                
                                # 强制垃圾回收以清理剩余引用
                                import gc
                                gc.collect()
                                
                                # 给 HTTP 客户端额外的清理时间
                                time.sleep(0.2)
                                
                                # 关闭循环
                                if not new_loop.is_closed():
                                    new_loop.close()
                            except Exception:
                                pass  # 忽略关闭错误
                
                # 在线程池中运行以避免阻塞主事件循环
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(run_in_thread)
                    results = future.result(timeout=300)  # 5 分钟超时
                    
            except RuntimeError:
                # 没有事件循环在运行，可以直接运行
                results = asyncio.run(self.search_async(max_results))
            
            return results
            
        except Exception as e:
            logger.error(f"Error in MCP search: {e}")
            self.streamer.stream_log_sync(f"❌ Error in MCP search: {str(e)}")
            # 返回空结果而不是抛出异常，让研究得以继续
            return []

    async def _get_all_tools(self) -> List:
        """
        获取 MCP 服务器的所有可用工具。
        
        返回：
            List: 所有可用的 MCP 工具
        """
        if self._all_tools_cache is not None:
            return self._all_tools_cache
            
        try:
            all_tools = await self.client_manager.get_all_tools()
            
            if all_tools:
                await self.streamer.stream_log(f"📋 Loaded {len(all_tools)} total tools from MCP servers")
                self._all_tools_cache = all_tools
                return all_tools
            else:
                await self.streamer.stream_warning("No tools available from MCP servers")
                return []
                
        except Exception as e:
            logger.error(f"Error getting MCP tools: {e}")
            await self.streamer.stream_error(f"Error getting MCP tools: {str(e)}")
            return [] 