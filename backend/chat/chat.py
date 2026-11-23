import logging
import os
import json
from fastapi import WebSocket
from typing import List, Dict, Any

from gpt_researcher.config.config import Config
from gpt_researcher.context.select import select_context
from gpt_researcher.utils.llm import create_chat_completion
from gpt_researcher.utils.tools import create_chat_completion_with_tools, create_search_tool
try:
    from tavily import TavilyClient
except ImportError:  # 聊天网页搜索的可选依赖
    TavilyClient = None
from datetime import datetime

# 设置日志
# 获取日志器实例
logger = logging.getLogger(__name__)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler()  # 只记录到控制台
    ]
)

# 注意：LLM 客户端现在由 GPT Researcher 统一 LLM 系统处理
# 它支持所有已配置的提供者（OpenAI、Google Gemini、Anthropic 等）

def get_tools():
    """为 LLM 函数调用定义工具（主要面向 OpenAI 兼容提供者）"""
    tools = [
        {
            "type": "function",
            "function": {
                "name": "quick_search",
                "description": "Search for current events or online information when you need new knowledge that doesn't exist in the current context",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The search query"
                        }
                    },
                    "required": ["query"]
                }
            }
        }
    ]
    return tools

class ChatAgentWithMemory:
    def __init__(
        self,
        report: str,
        config_path="default",
        headers=None,
        vector_store=None
    ):
        self.report = report
        self.headers = headers
        self.config = Config(config_path)
        self.vector_store = vector_store
        self.retriever = None
        self.search_metadata = None
        
        # 初始化 Tavily 客户端（可选 - 仅当 API 密钥可用时）
        tavily_api_key = os.environ.get("TAVILY_API_KEY")
        if tavily_api_key and TavilyClient is not None:
            self.tavily_client = TavilyClient(api_key=tavily_api_key)
        else:
            self.tavily_client = None
            if TavilyClient is None:
                logger.warning("tavily package not installed - web search in chat will be disabled")
            else:
                logger.warning("TAVILY_API_KEY not set - web search in chat will be disabled")
        
        # 调用方提供的向量库由其自行查询。否则报告
        # 走与研究相同的上下文过滤器（CONTEXT_FILTER）。
        if self.vector_store is not None:
            try:
                self.retriever = self.vector_store.as_retriever(search_kwargs={"k": 4})
            except TypeError:
                self.retriever = self.vector_store.as_retriever()

    def quick_search(self, query):
        """使用 Tavily 执行网页搜索获取最新信息"""
        try:
            # 检查 Tavily 客户端是否可用
            if self.tavily_client is None:
                logger.warning(f"Tavily client not available, skipping web search for: {query}")
                self.search_metadata = {
                    "query": query,
                    "sources": [],
                    "error": "Web search is disabled - TAVILY_API_KEY not configured"
                }
                return {
                    "error": "Web search is disabled - TAVILY_API_KEY not configured",
                    "results": []
                }
            
            logger.info(f"Performing web search for: {query}")
            results = self.tavily_client.search(query=query, max_results=5)
            
            # 存储搜索元数据供前端使用
            self.search_metadata = self._build_search_metadata(query, results)
            
            return results
        except Exception as e:
            logger.error(f"Error performing web search: {str(e)}", exc_info=True)
            results = {
                "error": str(e),
                "results": []
            }
            self.search_metadata = self._build_search_metadata(query, results)
            return results

    @staticmethod
    def _build_search_metadata(query, results):
        """描述一条搜索结果，不再发起额外的提供者请求。"""
        metadata = {"query": query, "sources": []}
        for result in results.get("results", []):
            content = result.get("content", "")
            metadata["sources"].append({
                "title": result.get("title", ""),
                "url": result.get("url", ""),
                "content": content[:200] + "..." if len(content) > 200 else content,
            })
        if "error" in results:
            metadata["error"] = results["error"]
        return metadata


    async def process_chat_completion(self, messages: List[Dict[str, str]]):
        """使用配置的 LLM 提供者处理带工具调用支持的聊天补全"""
        processed_metadata = []

        def search_with_metadata(query):
            results = self.quick_search(query)
            # 将来源保持在本补全内，并与返回给模型的
            # 证据绑定，包括重复查询和失败。
            processed_metadata.append({
                "tool": "quick_search",
                "query": query,
                "search_metadata": self._build_search_metadata(query, results),
            })
            return results

        search_tool = create_search_tool(search_with_metadata)
        
        # 使用工具启用型聊天补全工具
        response, tool_calls_metadata = await create_chat_completion_with_tools(
            messages=messages,
            tools=[search_tool],
            model=self.config.smart_llm_model,
            llm_provider=self.config.smart_llm_provider,
            llm_kwargs=self.config.llm_kwargs,
        )
        
        # 该工具在回退到普通补全时不返回工具元数据；
        # 该答案并非基于这些搜索结果生成。
        return response, processed_metadata if tool_calls_metadata else []



    async def _retrieve_context(self, user_message: str) -> str:
        """报告中与最新用户消息相关的部分。

        有注入的向量库时使用它，否则使用
        与研究相同的上下文过滤器。回退到完整报告，
        使聊天总有作答依据。
        """
        if not self.report or not user_message:
            return self.report or ""
        if self.retriever is not None:
            try:
                docs = self.retriever.invoke(user_message)
                chunks = [str(getattr(d, "page_content", "") or "") for d in docs or []]
                if any(chunks):
                    return "\n\n".join(c for c in chunks if c)
            except Exception as exc:  # noqa: BLE001 - 检索不得中断聊天
                logger.warning(f"Report retrieval failed, using the context filter: {exc}")
        try:
            context = await select_context(
                user_message,
                [{"url": "report", "title": "Research report", "raw_content": self.report}],
                self.config,
            )
        except Exception as exc:  # noqa: BLE001 - 聊天必须持续可用
            logger.warning(f"Report context selection failed, using full report: {exc}")
            return self.report
        return context or self.report

    async def chat(self, messages, websocket=None):
        """与配置的 LLM 提供者聊天（支持 OpenAI、Google Gemini、Anthropic 等）
        
        参数：
            messages: 带角色和内容的聊天消息列表
            websocket: 用于流式响应的可选 websocket
        
        返回：
            tuple: (str: AI 响应消息, dict: 关于工具使用的元数据)
        """
        try:
            
            # 优先使用检索到的报告切片，而不是每轮都把整份报告塞进去
            last_user = ""
            for msg in reversed(messages or []):
                if isinstance(msg, dict) and msg.get("role") == "user" and msg.get("content"):
                    last_user = str(msg.get("content"))
                    break
            report_context = await self._retrieve_context(last_user)

            # 用报告上下文格式化系统提示词
            system_prompt = f"""
            You are GPT Researcher, an autonomous research agent created by an open source community at https://github.com/assafelovic/gpt-researcher, homepage: https://gptr.dev. 
            To learn more about GPT Researcher you can suggest to check out: https://docs.gptr.dev.
            
            This is a chat about a research report that you created. Answer based on the given context and report.
            You must include citations to your answer based on the report.
            
            You may use the quick_search tool when the user asks about information that might require current data 
            not found in the report, such as recent events, updated statistics, or news. If there's no report available,
            you can use the quick_search tool to find information online.
            
            You must respond in markdown format. You must make it readable with paragraphs, tables, etc when possible. 
            Remember that you're answering in a chat not a report.
            
            Assume the current time is: {datetime.now()}.
            
            Report: {report_context}
            
            """
            
            # 为 OpenAI 输入格式化消息历史
            formatted_messages = []
            
            # 先添加系统消息
            formatted_messages.append({
                "role": "system", 
                "content": system_prompt
            })
            
            # 添加用户/助手消息历史 - 过滤掉非必要字段
            for msg in messages:
                if isinstance(msg, dict) and 'role' in msg and 'content' in msg:
                    formatted_messages.append({
                        "role": msg["role"],
                        "content": msg["content"]
                    })
                else:
                    logger.warning(f"Skipping message with missing role or content: {msg}")
            
            # 使用配置的 LLM 提供者处理聊天
            ai_message, tool_calls_metadata = await self.process_chat_completion(formatted_messages)
            
            # 如果消息为空则提供回退响应
            if not ai_message:
                logger.warning("No AI message content found in response, using fallback message")
                ai_message = "I apologize, but I couldn't generate a proper response. Please try asking your question again."
            
            logger.info(f"Generated response: {ai_message[:100]}..." if len(ai_message) > 100 else f"Generated response: {ai_message}")
            
            # 返回消息和关于所用工具的任何元数据
            return ai_message, tool_calls_metadata
            
        except Exception as e:
            logger.error(f"Error in chat: {str(e)}", exc_info=True)
            raise

    def get_context(self):
        """返回当前聊天的上下文"""
        return self.report
