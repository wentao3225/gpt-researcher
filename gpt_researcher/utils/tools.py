"""
GPT Researcher 的工具启用型 LLM 工具

此模块基于 LangChain 的统一接口提供与提供者无关的工具调用功能。
任何支持函数调用的 LLM 提供者都可以无缝使用这些工具。
"""

import asyncio
import logging
from typing import Any, Dict, List, Tuple, Callable, Optional
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.tools import tool

from .costs import calculate_llm_cost
from .llm import create_chat_completion

logger = logging.getLogger(__name__)


def _track_response_cost(
    *,
    llm_provider: str | None,
    model: str | None,
    input_payload: Any,
    response_message: Any,
    request_options: Dict[str, Any],
    cost_callback: Callable | None,
) -> None:
    if not cost_callback:
        return

    response_content = getattr(response_message, "text", "") or ""
    llm_costs = calculate_llm_cost(
        llm_provider=llm_provider,
        model=model,
        input_content=str(input_payload),
        output_content=str(response_content),
        response_metadata=getattr(response_message, "response_metadata", None),
        usage_metadata=getattr(response_message, "usage_metadata", None),
        request_options=request_options,
    )
    cost_callback(llm_costs)


async def create_chat_completion_with_tools(
    messages: List[Dict[str, str]],
    tools: List[Callable],
    model: str | None = None,
    temperature: float | None = 0.4,
    max_tokens: int | None = 4000,
    llm_provider: str | None = None,
    llm_kwargs: Dict[str, Any] | None = None,
    cost_callback: Callable = None,
    websocket: Any | None = None,
    **kwargs
) -> Tuple[str, List[Dict[str, Any]]]:
    """
    在所有 LLM 提供者中创建支持工具调用的聊天补全。
    
    此函数使用 LangChain 的 bind_tools() 以与提供者无关的方式启用函数调用。
    AI 自主决定何时以及如何使用工具。
    
    参数：
        messages: 包含 role 和 content 的聊天消息列表
        tools: LangChain 工具函数列表（用 @tool 装饰）
        model: 要使用的模型（来自配置）
        temperature: 生成温度
        max_tokens: 可生成的最大 token 数
        llm_provider: LLM 提供者名称（来自配置）
        llm_kwargs: 额外的 LLM 关键字参数
        cost_callback: 用于成本跟踪的回调函数
        websocket: 用于流式输出的可选 websocket
        **kwargs: 额外参数
        
    返回：
        (response_content, tool_calls_metadata) 元组
        
    抛出：
        Exception: 如果工具启用补全失败，则回退到简单补全
    """
    try:
        from ..llm_provider.generic.base import GenericLLMProvider
        
        # 使用配置创建 LLM 提供者
        provider_kwargs = {
            'model': model,
            **(llm_kwargs or {})
        }
        
        llm_provider_instance = GenericLLMProvider.from_provider(
            llm_provider, 
            **provider_kwargs
        )
        
        # 将消息转换为 LangChain 格式
        lc_messages = []
        for msg in messages:
            if msg["role"] == "system":
                lc_messages.append(SystemMessage(content=msg["content"]))
            elif msg["role"] == "user":
                lc_messages.append(HumanMessage(content=msg["content"]))
            elif msg["role"] == "assistant":
                lc_messages.append(AIMessage(content=msg["content"]))
        
        # 将工具绑定到 LLM —— 适用于所有支持函数调用的 LangChain 提供者
        llm_with_tools = llm_provider_instance.llm.bind_tools(tools)
        
        # 携带工具调用 LLM —— 由它处理完整的对话流程
        logger.info(f"Invoking LLM with {len(tools)} available tools")
        
        # 工具调用需要处理包含工具响应的完整对话
        from langchain_core.messages import ToolMessage
        
        # 首次调用 LLM
        response = await llm_with_tools.ainvoke(lc_messages)
        _track_response_cost(
            llm_provider=llm_provider,
            model=model,
            input_payload=lc_messages,
            response_message=response,
            request_options=provider_kwargs,
            cost_callback=cost_callback,
        )
        
        # 如果有工具调用则进行处理
        tool_calls_metadata = []
        if hasattr(response, 'tool_calls') and response.tool_calls:
            logger.info(f"LLM made {len(response.tool_calls)} tool calls")
            
            # 将带工具调用的助手响应加入对话
            lc_messages.append(response)
            
            # 执行每个工具调用并将结果加入对话
            for tool_call in response.tool_calls:
                tool_name = tool_call.get('name', 'unknown')
                tool_args = tool_call.get('args', {})
                tool_id = tool_call.get('id', '')
                
                logger.info(f"Tool called: {tool_name}")
                if tool_args:
                    args_str = ", ".join([f"{k}={v}" for k, v in tool_args.items()])
                    logger.debug(f"Tool arguments: {args_str}")
                
                # 查找并执行工具
                tool_result = "Tool execution failed"
                for tool in tools:
                    if tool.name == tool_name:
                        try:
                            if hasattr(tool, 'ainvoke'):
                                tool_result = await tool.ainvoke(tool_args)
                            elif hasattr(tool, 'invoke'):
                                tool_result = tool.invoke(tool_args)
                            else:
                                tool_result = await tool(**tool_args) if asyncio.iscoroutinefunction(tool) else tool(**tool_args)
                            break
                        except Exception as e:
                            error_type = type(e).__name__
                            error_msg = str(e)
                            logger.error(
                                f"Error executing tool '{tool_name}': {error_type}: {error_msg}",
                                exc_info=True
                            )
                            # 提供用户友好的错误消息
                            if "timeout" in error_msg.lower() or "timed out" in error_msg.lower():
                                tool_result = f"Tool '{tool_name}' timed out. The operation took too long to complete. Please try again or check your network connection."
                            elif "connection" in error_msg.lower() or "network" in error_msg.lower():
                                tool_result = f"Tool '{tool_name}' failed due to a network issue. Please check your internet connection and try again."
                            elif "permission" in error_msg.lower() or "access" in error_msg.lower():
                                tool_result = f"Tool '{tool_name}' failed due to insufficient permissions. Please check your API keys or access credentials."
                            else:
                                tool_result = f"Tool '{tool_name}' encountered an error: {error_msg}. Please check the logs for more details."
                
                # 将工具结果加入对话
                tool_message = ToolMessage(content=str(tool_result), tool_call_id=tool_id)
                lc_messages.append(tool_message)
                
                # 加入元数据
                tool_calls_metadata.append({
                    "tool": tool_name,
                    "args": tool_args,
                    "call_id": tool_id,
                    "result": str(tool_result)[:200] + "..." if len(str(tool_result)) > 200 else str(tool_result)
                })
            
            # 工具执行完成后从 LLM 获取最终响应
            logger.info("Getting final response from LLM after tool execution")
            final_response = await llm_with_tools.ainvoke(lc_messages)
             
            # 如果提供了回调则跟踪成本
            _track_response_cost(
                llm_provider=llm_provider,
                model=model,
                input_payload=lc_messages,
                response_message=final_response,
                request_options=provider_kwargs,
                cost_callback=cost_callback,
            )
             
            # 绑定工具后，部分模型通过 Responses API 应答，其内容是块列表 :-)
            return final_response.text, tool_calls_metadata
         
        else:
            # 没有工具调用，返回常规响应
            return response.text, []
        
    except Exception as e:
        error_type = type(e).__name__
        error_msg = str(e)
        logger.error(
            f"Error in tool-enabled chat completion: {error_type}: {error_msg}",
            exc_info=True
        )
        logger.info("Falling back to simple chat completion without tools")
        
        # 回退到不带工具的简单聊天补全
        response = await create_chat_completion(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            llm_provider=llm_provider,
            llm_kwargs=llm_kwargs,
            cost_callback=cost_callback,
            websocket=websocket,
            **kwargs
        )
        return response, []


def create_search_tool(search_function: Callable[[str], Dict]) -> Callable:
    """
    创建用于工具启用型聊天补全的标准化搜索工具。
    
    参数：
        search_function: 接受查询字符串并返回搜索结果的函数
        
    返回：
        用 @tool 装饰的 LangChain 工具函数
    """
    @tool
    def search_tool(query: str) -> str:
        """Search for current events or online information when you need new knowledge that doesn't exist in the current context"""
        try:
            results = search_function(query)
            if results and 'results' in results:
                search_content = f"Search results for '{query}':\n\n"
                for result in results['results'][:5]:
                    search_content += f"Title: {result.get('title', '')}\n"
                    search_content += f"Content: {result.get('content', '')[:300]}...\n"
                    search_content += f"URL: {result.get('url', '')}\n\n"
                return search_content
            else:
                return f"No search results found for: {query}"
        except Exception as e:
            error_type = type(e).__name__
            error_msg = str(e)
            logger.error(
                f"Search tool error: {error_type}: {error_msg}",
                exc_info=True
            )
            # 提供上下文相关的错误消息
            if "api" in error_msg.lower() or "key" in error_msg.lower():
                return f"Search failed: API key issue. Please verify your search API credentials are configured correctly."
            elif "timeout" in error_msg.lower() or "timed out" in error_msg.lower():
                return f"Search timed out. The search request took too long. Please try again with a different query."
            elif "rate limit" in error_msg.lower() or "quota" in error_msg.lower():
                return f"Search rate limit exceeded. Please wait a moment before trying again."
            else:
                return f"Search encountered an error: {error_msg}. Please check your search provider configuration."
    
    return search_tool


def create_custom_tool(
    name: str,
    description: str, 
    function: Callable,
    parameter_schema: Optional[Dict] = None
) -> Callable:
    """
    创建用于工具启用型聊天补全的自定义工具。
    
    参数：
        name: 工具名称
        description: 工具功能的描述
        function: 要执行的实际函数
        parameter_schema: 可选的函数参数模式
        
    返回：
        用 @tool 装饰的 LangChain 工具函数
    """
    @tool
    def custom_tool(*args, **kwargs) -> str:
        try:
            result = function(*args, **kwargs)
            return str(result) if result is not None else "Tool executed successfully"
        except Exception as e:
            error_type = type(e).__name__
            error_msg = str(e)
            logger.error(
                f"Custom tool '{name}' error: {error_type}: {error_msg}",
                exc_info=True
            )
            # 提供信息性错误消息而不暴露内部细节
            if "validation" in error_msg.lower() or "invalid" in error_msg.lower():
                return f"Tool '{name}' received invalid input. Please check the parameters and try again."
            elif "not found" in error_msg.lower() or "missing" in error_msg.lower():
                return f"Tool '{name}' could not find required resources. Please verify the input data is correct."
            else:
                return f"Tool '{name}' encountered an error: {error_msg}. Please check the tool configuration."
    
    # 设置工具元数据
    custom_tool.name = name
    custom_tool.description = description
    
    return custom_tool


# 常见工具模式的工具函数
def get_available_providers_with_tools() -> List[str]:
    """
    获取支持工具调用的 LLM 提供者列表。
    
    返回：
        支持函数调用的提供者名称列表
    """
    # 这些是已知在 LangChain 中支持函数调用的提供者
    return [
        "openai",
        "anthropic", 
        "google_genai",
        "azure_openai",
        "fireworks",
        "groq",
        # 注意：随着更多提供者加入函数调用支持，此列表可能会扩展
    ]


def supports_tools(provider: str) -> bool:
    """
    检查给定提供者是否支持工具调用。
    
    参数：
        provider: LLM 提供者名称
        
    返回：
        提供者支持工具则返回 True，否则返回 False
    """
    return provider in get_available_providers_with_tools()
