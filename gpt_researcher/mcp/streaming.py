"""
MCP 流式工具模块

处理 MCP 操作的 websocket 流式传输和日志记录。
"""
import asyncio
import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)


class MCPStreamer:
    """
    处理 MCP 操作的流式输出。
    
    负责以下事项：
    - 将日志流式传输到 websocket
    - 同步/异步日志记录
    - 流式传输中的错误处理
    """

    def __init__(self, websocket=None):
        """
        初始化 MCP 流式传输器。
        
        参数：
            websocket: 用于流式输出的 WebSocket
        """
        self.websocket = websocket

    async def stream_log(self, message: str, data: Any = None):
        """如果 websocket 可用则将日志消息流式传输过去。"""
        logger.info(message)
        
        if self.websocket:
            try:
                from ..actions.utils import stream_output
                await stream_output(
                    type="logs", 
                    content="mcp_retriever", 
                    output=message, 
                    websocket=self.websocket,
                    metadata=data
                )
            except Exception as e:
                logger.error(f"Error streaming log: {e}")
                
    def stream_log_sync(self, message: str, data: Any = None):
        """stream_log 的同步版本，供同步上下文使用。"""
        logger.info(message)
        
        if self.websocket:
            try:
                try:
                    loop = asyncio.get_event_loop()
                    if loop.is_running():
                        asyncio.create_task(self.stream_log(message, data))
                    else:
                        loop.run_until_complete(self.stream_log(message, data))
                except RuntimeError:
                    logger.debug("Could not stream log: no running event loop")
            except Exception as e:
                logger.error(f"Error in sync log streaming: {e}")

    async def stream_stage_start(self, stage: str, description: str):
        """流式传输研究阶段的开始。"""
        await self.stream_log(f"🔧 {stage}: {description}")

    async def stream_stage_complete(self, stage: str, result_count: int = None):
        """流式传输研究阶段的完成。"""
        if result_count is not None:
            await self.stream_log(f"✅ {stage} completed: {result_count} results")
        else:
            await self.stream_log(f"✅ {stage} completed")

    async def stream_tool_selection(self, selected_count: int, total_count: int):
        """流式传输工具选择信息。"""
        await self.stream_log(f"🧠 Using LLM to select {selected_count} most relevant tools from {total_count} available")

    async def stream_tool_execution(self, tool_name: str, step: int, total: int):
        """流式传输工具执行进度。"""
        await self.stream_log(f"🔍 Executing tool {step}/{total}: {tool_name}")

    async def stream_research_results(self, result_count: int, total_chars: int = None):
        """流式传输研究结果摘要。"""
        if total_chars:
            await self.stream_log(f"✅ MCP research completed: {result_count} results obtained ({total_chars:,} chars)")
        else:
            await self.stream_log(f"✅ MCP research completed: {result_count} results obtained")

    async def stream_error(self, error_msg: str):
        """流式传输错误消息。"""
        await self.stream_log(f"❌ {error_msg}")

    async def stream_warning(self, warning_msg: str):
        """流式传输警告消息。"""
        await self.stream_log(f"⚠️ {warning_msg}")

    async def stream_info(self, info_msg: str):
        """流式传输信息消息。"""
        await self.stream_log(f"ℹ️ {info_msg}") 