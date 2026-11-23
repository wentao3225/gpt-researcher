"""
MCP 工具选择模块

使用 LLM 分析处理智能工具选择。
"""
import asyncio
import json
import logging

import json_repair
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)


class MCPToolSelector:
    """
    使用 LLM 分析处理 MCP 工具的智能选择。
    
    负责以下事项：
    - 使用 LLM 分析可用工具
    - 为查询选择最相关的工具
    - 提供回退选择机制
    """

    def __init__(self, cfg, researcher=None):
        """
        初始化工具选择器。
        
        参数：
            cfg: 包含 LLM 设置的配置对象
            researcher: 用于成本跟踪的研究员实例
        """
        self.cfg = cfg
        self.researcher = researcher

    async def select_relevant_tools(self, query: str, all_tools: List, max_tools: int = 3) -> List:
        """
        使用 LLM 为研究查询选择最相关的工具。
        
        参数：
            query: 研究查询
            all_tools: 所有可用工具的列表
            max_tools: 可选择的最大工具数（默认：3）
            
        返回：
            List: 与查询最相关的所选工具
        """
        if not all_tools:
            return []

        if len(all_tools) < max_tools:
            max_tools = len(all_tools)
            
        logger.info(f"Using LLM to select {max_tools} most relevant tools from {len(all_tools)} available")
        
        # 为 LLM 分析创建工具描述
        tools_info = []
        for i, tool in enumerate(all_tools):
            tool_info = {
                "index": i,
                "name": tool.name,
                "description": tool.description or "No description available"
            }
            tools_info.append(tool_info)
        
        # 在此导入以避免循环导入
        from ..prompts import PromptFamily
        
        # 为智能工具选择创建提示词
        prompt = PromptFamily.generate_mcp_tool_selection_prompt(query, tools_info, max_tools)

        try:
            # 调用 LLM 进行工具选择
            response = await self._call_llm_for_tool_selection(prompt)
            
            if not response:
                logger.warning("No LLM response for tool selection, using fallback")
                return self._fallback_tool_selection(all_tools, max_tools)
            
            # 记录 LLM 响应预览以便调试
            response_preview = response[:500] + "..." if len(response) > 500 else response
            logger.debug(f"LLM tool selection response: {response_preview}")
            
            # 解析 LLM 响应 —— json_repair 处理围栏/前言（与 deep_research 相同）
            try:
                selection_result = json_repair.loads(response)
            except Exception:
                logger.warning("Could not parse LLM tool selection JSON, using fallback")
                return self._fallback_tool_selection(all_tools, max_tools)

            if not isinstance(selection_result, dict):
                logger.warning("Tool selection response was not a JSON object, using fallback")
                return self._fallback_tool_selection(all_tools, max_tools)

            selected_tools = []
            selected_indexes = set()

            # 处理所选工具。模型的回答是建议而非约束：它可能列出
            # 超出要求数量的工具，或重复同一索引。像回退路径的
            # scored_tools[:max_tools] 一样在此强制执行 max_tools，
            # 因为调用方会把返回的所有内容都绑定给 LLM。
            for tool_selection in selection_result.get("selected_tools", []) or []:
                if len(selected_tools) >= max_tools:
                    break
                if not isinstance(tool_selection, dict):
                    continue
                tool_index = tool_selection.get("index")
                tool_name = tool_selection.get("name", "")
                reason = tool_selection.get("reason", "")
                relevance_score = tool_selection.get("relevance_score", 0)

                if tool_index is not None and 0 <= tool_index < len(all_tools):
                    if tool_index in selected_indexes:
                        continue
                    selected_indexes.add(tool_index)
                    selected_tools.append(all_tools[tool_index])
                    logger.info(f"Selected tool '{tool_name}' (score: {relevance_score}): {reason}")
            
            if len(selected_tools) == 0:
                logger.warning("No tools selected by LLM, using fallback selection")
                return self._fallback_tool_selection(all_tools, max_tools)
            
            # 记录整体选择理由
            selection_reasoning = selection_result.get("selection_reasoning", "No reasoning provided")
            logger.info(f"LLM selection strategy: {selection_reasoning}")
            
            logger.info(f"LLM selected {len(selected_tools)} tools for research")
            return selected_tools
            
        except Exception as e:
            logger.error(f"Error in LLM tool selection: {e}")
            logger.warning("Falling back to pattern-based selection")
            return self._fallback_tool_selection(all_tools, max_tools)

    async def _call_llm_for_tool_selection(self, prompt: str) -> str:
        """
        使用现有的 create_chat_completion 函数调用 LLM 进行工具选择。
        
        参数：
            prompt (str): 要发送给 LLM 的提示词。
            
        返回：
            str: 生成的文本响应。
        """
        if not self.cfg:
            logger.warning("No config available for LLM call")
            return ""
            
        try:
            from ..utils.llm import create_chat_completion
            
            # 创建 LLM 消息
            messages = [{"role": "user", "content": prompt}]
            
            # 使用 strategic LLM 进行工具选择（因为它更擅长复杂推理）
            result = await create_chat_completion(
                model=self.cfg.strategic_llm_model,
                messages=messages,
                temperature=0.0,  # 低温度以获得一致的工具选择
                llm_provider=self.cfg.strategic_llm_provider,
                llm_kwargs=self.cfg.llm_kwargs,
                cost_callback=self.researcher.add_costs if self.researcher and hasattr(self.researcher, 'add_costs') else None,
            )
            return result
        except Exception as e:
            logger.error(f"Error calling LLM for tool selection: {e}")
            return ""

    def _fallback_tool_selection(self, all_tools: List, max_tools: int) -> List:
        """
        LLM 选择失败时使用模式匹配的回退工具选择。
        
        参数：
            all_tools: 所有可用工具的列表
            max_tools: 可选择的最大工具数
            
        返回：
            List: 所选工具
        """
        # 定义与研究相关工具的模式
        research_patterns = [
            'search', 'get', 'read', 'fetch', 'find', 'list', 'query', 
            'lookup', 'retrieve', 'browse', 'view', 'show', 'describe'
        ]
        
        scored_tools = []
        
        for tool in all_tools:
            tool_name = tool.name.lower()
            tool_description = (tool.description or "").lower()
            
            # 基于模式匹配计算相关性分数
            score = 0
            for pattern in research_patterns:
                if pattern in tool_name:
                    score += 3
                if pattern in tool_description:
                    score += 1
            
            if score > 0:
                scored_tools.append((tool, score))
        
        # 按分数排序并取前几个工具
        scored_tools.sort(key=lambda x: x[1], reverse=True)
        selected_tools = [tool for tool, score in scored_tools[:max_tools]]
        
        for i, (tool, score) in enumerate(scored_tools[:max_tools]):
            logger.info(f"Fallback selected tool {i+1}: {tool.name} (score: {score})")
        
        return selected_tools 