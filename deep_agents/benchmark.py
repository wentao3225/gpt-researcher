"""基准智能体：深度智能体 + GPT Researcher vs 深度智能体 + 原始搜索。

定义两个仅在 research 工具上不同的深度智能体：

- baseline：deepagents 快速入门设置 - 一个原始 Tavily `internet_search`
  工具（https://docs.langchain.com/oss/python/deepagents/quickstart）。
- gptr：本示例的设置 - 把 GPT Researcher 暴露为 `quick_search` 和
  `deep_research` 工具。

两者使用相同的模型、相同的系统提示词和相同的框架，因此输出质量的
任何差异都只来自研究引擎。由 `drb_generate.py` 用于为
DeepResearch Bench（https://github.com/Ayanami0730/deep_research_bench）
生成报告。结果和复现步骤见 BENCHMARK.md。
"""

from dotenv import load_dotenv
import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

load_dotenv()

from tavily import TavilyClient
from deepagents import create_deep_agent

from deep_agents.tools import build_research_tools

# 两个系统共享，因此唯一变量是研究工具。
# 每个系统另外获得如何使用自己工具的指引，
# 遵循记录了 `internet_search` 的 deepagents 快速入门提示词精神。
SYSTEM_PROMPT = f"""You are an expert researcher. Today's date is \
{datetime.now().strftime('%B %d, %Y')}. Use your tools to research the user's \
question and answer it accurately. Always ground your answer in researched \
sources rather than your own memory - your training data predates today, so \
anything time-sensitive (versions, prices, events, results) must come from \
research. If the research does not surface the answer, say you could not find \
it instead of guessing. End your final reply with a short, direct answer to \
the question."""

BASELINE_TOOL_GUIDE = """

## `internet_search`

Use this to run an internet search for a given query. You can specify the max
number of results to return, the topic, and whether raw content should be
included. Run as many searches with refined queries as you need."""

GPTR_TOOL_GUIDE = """

## `quick_search`

Your primary tool for factual lookups and for discovering what is currently
true. Run it as many times as you need with refined queries (names, dates,
alternate phrasings). For multi-part questions, decompose them and look up
each fact separately. Before any deep research, use it to establish the
current state of the topic (latest names, versions, events, dates) so later
queries do not carry stale assumptions.

## `deep_research`

A full research pipeline that is slow but has much deeper recall (it scrapes
and reads dozens of pages). Escalate to it when repeated quick_search calls
cannot surface the fact, or the task genuinely requires deep multi-source
research. Phrase its queries neutrally ("the current X lineup and pricing") -
never bake names, versions or facts recalled from your own memory into a
query, because the pipeline will faithfully research whatever you assert.
When several sources disagree, prefer the most authoritative source (e.g. an
encyclopedia or official page). Before finalizing an answer or report, verify
load-bearing specifics (scores, prices, versions, names, dates) with
quick_search rather than trusting a single summary."""


def message_text(message) -> str:
    text = getattr(message, "text", "")
    if text is not None and not isinstance(text, str) and callable(text):
        text = text()
    return text or ""


def build_baseline_agent(model: str, system_prompt: str | None = None):
    """deepagents 快速入门研究智能体：一个原始 Tavily 搜索工具。"""
    tavily_client = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])

    def internet_search(
        query: str,
        max_results: int = 5,
        topic: str = "general",
        include_raw_content: bool = False,
    ):
        """执行网页搜索。保持查询简短（400 字符以内）。

        topic 必须是 "general"、"news" 或 "finance" 之一。
        """
        if topic not in ("general", "news", "finance"):
            topic = "general"
        return tavily_client.search(
            query[:400],  # Tavily 拒绝超过 400 字符的查询
            max_results=max_results,
            include_raw_content=include_raw_content,
            topic=topic,
        )

    agent = create_deep_agent(
        model=model,
        tools=[internet_search],
        system_prompt=(system_prompt or SYSTEM_PROMPT) + BASELINE_TOOL_GUIDE,
    )
    return agent, None


def build_gptr_agent(model: str, source: str = "web", system_prompt: str | None = None):
    """本示例的研究智能体：GPT Researcher 作为研究引擎。"""
    cost_tracker = {"total": 0.0}
    quick_search, deep_research = build_research_tools({"source": source}, cost_tracker)
    agent = create_deep_agent(
        model=model,
        tools=[quick_search, deep_research],
        system_prompt=(system_prompt or SYSTEM_PROMPT) + GPTR_TOOL_GUIDE,
    )
    return agent, cost_tracker
