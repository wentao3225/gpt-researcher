"""GPT Researcher 的来源策展技能。

此模块提供 SourceCurator 类，基于相关性、可信度和可靠性
评估和排序研究来源。
"""

import logging
from typing import Dict, List, Optional

import json_repair
from langchain_core.utils.json import parse_json_markdown

from ..actions import stream_output
from ..config.config import Config
from ..utils.llm import create_chat_completion

logger = logging.getLogger(__name__)


class SourceCurator:
    """基于相关性、可信度和可靠性排序和策展来源。

    此类使用基于 LLM 的评估来评价研究来源，
    并为报告生成挑选最合适的来源。

    属性：
        researcher: 上级 GPTResearcher 实例。
    """

    def __init__(self, researcher):
        """初始化 SourceCurator。

        参数：
            researcher: 拥有此策展器的 GPTResearcher 实例。
        """
        self.researcher = researcher

    async def curate_sources(
        self,
        source_data: List,
        max_results: int = 10,
    ) -> List:
        """
        根据研究数据和指南对来源排序。

        参数：
            query: 研究查询/任务
            source_data: 要排序的来源文档列表
            max_results: 返回的来源顶部最大数量

        返回：
            str: 带理由的来源 URL 排序列表
        """
        logger.debug(f"Curating {len(source_data)} sources")
        if self.researcher.verbose:
            await stream_output(
                "logs",
                "research_plan",
                f"⚖️ Evaluating and curating sources by credibility and relevance...",
                self.researcher.websocket,
            )

        response = ""
        try:
            response = await create_chat_completion(
                model=self.researcher.cfg.smart_llm_model,
                messages=[
                    {"role": "system", "content": f"{self.researcher.role}"},
                    {"role": "user", "content": self.researcher.prompt_family.curate_sources(
                        self.researcher.query, source_data, max_results)},
                ],
                temperature=0.2,
                max_tokens=8000,
                llm_provider=self.researcher.cfg.smart_llm_provider,
                llm_kwargs=self.researcher.cfg.llm_kwargs,
                cost_callback=self.researcher.add_costs,
            )

            # LLM 经常无视提示词指示，把 JSON 包在 ```json 围栏里或加散文，
            # 普通 json.loads 无法解析。以代码库其余部分相同的方式恢复
            # 载荷（见 actions/query_processing.py、multi_agents/agents/utils/
            # llms.py），使格式正确但带围栏的响应不被丢弃。
            curated_sources = parse_json_markdown(response, parser=json_repair.loads)
            # json_repair 从不抛出异常：对不可用输出它返回 "" 或字典
            # 而非列表。做防护，使此类结果仍回退到下方未策展的来源，
            # 而不是清空上下文。
            if not isinstance(curated_sources, list):
                raise ValueError(
                    f"expected a JSON list of sources, got "
                    f"{type(curated_sources).__name__}"
                )
            logger.debug(f"Curated {len(curated_sources)} of {len(source_data)} sources")

            if self.researcher.verbose:
                await stream_output(
                    "logs",
                    "research_plan",
                    f"🏅 Verified and ranked top {len(curated_sources)} most reliable sources",
                    self.researcher.websocket,
                )

            return curated_sources

        except Exception as e:
            logger.error(f"Error in curate_sources: {e}")
            if self.researcher.verbose:
                await stream_output(
                    "logs",
                    "research_plan",
                    f"🚫 Source verification failed: {str(e)}",
                    self.researcher.websocket,
                )
            return source_data
