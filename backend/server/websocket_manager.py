import os
import asyncio
import datetime
import json
import logging
import os
import traceback
from typing import Dict, List

from fastapi import WebSocket

from backend.report_type import BasicReport, DetailedReport

from gpt_researcher.utils.enum import ReportType, Tone
from gpt_researcher.actions import stream_output  # 导入 stream_output
from .multi_agent_runner import run_multi_agent_task
from .server_utils import CustomLogsHandler

logger = logging.getLogger(__name__)

class WebSocketManager:
    """管理 websocket"""

    def __init__(self):
        """初始化 WebSocketManager 类。"""
        self.active_connections: List[WebSocket] = []
        self.sender_tasks: Dict[WebSocket, asyncio.Task] = {}
        self.message_queues: Dict[WebSocket, asyncio.Queue] = {}

    async def start_sender(self, websocket: WebSocket):
        """启动发送任务。"""
        queue = self.message_queues.get(websocket)
        if not queue:
            return

        while True:
            try:
                message = await queue.get()
                if message is None:  # 关闭信号
                    break
                    
                if websocket in self.active_connections:
                    if message == "ping":
                        await websocket.send_text("pong")
                    else:
                        await websocket.send_text(message)
                else:
                    break
            except Exception as e:
                print(f"Error in sender task: {e}")
                break

    async def connect(self, websocket: WebSocket):
        """连接 websocket。"""
        try:
            await websocket.accept()
            self.active_connections.append(websocket)
            self.message_queues[websocket] = asyncio.Queue()
            self.sender_tasks[websocket] = asyncio.create_task(
                self.start_sender(websocket))
        except Exception as e:
            print(f"Error connecting websocket: {e}")
            if websocket in self.active_connections:
                await self.disconnect(websocket)

    async def disconnect(self, websocket: WebSocket):
        """断开 websocket。"""
        try:
            if websocket in self.active_connections:
                self.active_connections.remove(websocket)
                
                # 如有发送任务则取消
                if websocket in self.sender_tasks:
                    try:
                        self.sender_tasks[websocket].cancel()
                        await self.message_queues[websocket].put(None)
                    except Exception as e:
                        logger.error(f"Error canceling sender task: {e}")
                    finally:
                        # 无论如何始终尝试清理
                        if websocket in self.sender_tasks:
                            del self.sender_tasks[websocket]
                
                # 清理消息队列
                if websocket in self.message_queues:
                    del self.message_queues[websocket]
                
                # 最后关闭 WebSocket
                try:
                    await websocket.close()
                except Exception as e:
                    logger.info(f"WebSocket already closed: {e}")
        except Exception as e:
            logger.error(f"Error during WebSocket disconnection: {e}")
            # 仍尝试关闭连接（如可能）
            try:
                await websocket.close()
            except Exception:
                pass  # 如果这也失败，就没有更多可做的了

    async def start_streaming(self, task, report_type, report_source, source_urls, document_urls, tone, websocket, headers=None, query_domains=[], mcp_enabled=False, mcp_strategy="fast", mcp_configs=[], max_search_results=None):
        """开始流式输出。"""
        tone = Tone[tone]
        # 在此添加自定义 JSON 配置文件路径
        config_path = os.environ.get("CONFIG_PATH", "default")

        # 将 MCP 参数传给 run_agent
        report = await run_agent(
            task, report_type, report_source, source_urls, document_urls, tone, websocket, 
            headers=headers, query_domains=query_domains, config_path=config_path,
            mcp_enabled=mcp_enabled, mcp_strategy=mcp_strategy, mcp_configs=mcp_configs,
            max_search_results=max_search_results
        )
        return report

async def run_agent(task, report_type, report_source, source_urls, document_urls, tone: Tone, websocket, stream_output=stream_output, headers=None, query_domains=[], config_path="", return_researcher=False, mcp_enabled=False, mcp_strategy="fast", mcp_configs=[], max_search_results=None):
    """运行智能体。"""
    # 为此研究任务创建日志处理器
    logs_handler = CustomLogsHandler(websocket, task)

    # 记录 MCP 初始化。检索器和策略在 GPTResearcher 内
    # 通过 mcp_configs/mcp_strategy 参数按请求配置 ——
    # 这里不需要改动 os.environ（改动 os.environ 会跨请求持久化
    # 并影响无关会话，见 issue #1676）。
    if mcp_enabled and mcp_configs:
        print(f"🔧 MCP enabled with strategy '{mcp_strategy}' and {len(mcp_configs)} server(s)")
        await logs_handler.send_json({
            "type": "logs",
            "content": "mcp_init",
            "output": f"🔧 MCP enabled with strategy '{mcp_strategy}' and {len(mcp_configs)} server(s)"
        })

    # 根据报告类型初始化研究员
    if report_type == "multi_agents":
        report = await run_multi_agent_task(
            query=task, 
            websocket=logs_handler,  # 使用 logs_handler 而非原始 websocket
            stream_output=stream_output, 
            tone=tone, 
            headers=headers
        )
        report = report.get("report", "")

    elif report_type == ReportType.DetailedReport.value:
        researcher = DetailedReport(
            query=task,
            query_domains=query_domains,
            report_type=report_type,
            report_source=report_source,
            source_urls=source_urls,
            document_urls=document_urls,
            tone=tone,
            config_path=config_path,
            websocket=logs_handler,  # Use logs_handler instead of raw websocket
            headers=headers,
            mcp_configs=mcp_configs if mcp_enabled else None,
            mcp_strategy=mcp_strategy if mcp_enabled else None,
            max_search_results=max_search_results,
        )
        report = await researcher.run()
        
    else:
        researcher = BasicReport(
            query=task,
            query_domains=query_domains,
            report_type=report_type,
            report_source=report_source,
            source_urls=source_urls,
            document_urls=document_urls,
            tone=tone,
            config_path=config_path,
            websocket=logs_handler,  # Use logs_handler instead of raw websocket
            headers=headers,
            mcp_configs=mcp_configs if mcp_enabled else None,
            mcp_strategy=mcp_strategy if mcp_enabled else None,
            max_search_results=max_search_results,
        )
        report = await researcher.run()

    if report_type != "multi_agents" and return_researcher:
        return report, researcher.gpt_researcher
    else:
        return report
