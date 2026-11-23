"""混合研究基准：私有文档 + 网络 vs 仅网络工具。

GPT Researcher 的 ``hybrid`` 模式同时通过本地文档（DOC_PATH）
和网络跑相同的研究管线。此基准度量它相对于两个现成 deepagents
替代方案的价值：

- ``baseline``：仅原始 Tavily 网页搜索（快速入门设置）。
  完全看不到私有语料库。
- ``baseline+files``：原始网页搜索加上挂载到语料库的 deepagents
  文件系统工具（``ls``/``read_file``） —— 把框架自带的文件工具指向
  你的文档这种显而易见的 DIY 方案。
- ``gptr``：GPT Researcher 混合模式作为研究工具。

设置：一份虚构的内部公司文档语料库（Veltrix Dynamics，一家虚构的
AMR 机器人公司 —— 使任何事实都不会从公共网络泄漏）由
``benchmark_data/build_corpus.py`` 生成到
``benchmark_data/internal_docs/``。它形状像真实文档共享库：
27 个文件分布在部门子目录，其中四份承载事实的文档是 PDF、
DOCX 和 markdown，周围是真实的干扰项（HR 政策、IT 手册、营销
笔记）以及同一报告陈旧归档年份版本，其过时数字被评分为 WRONG。

所有智能体收到相同的简报：一份结合内部公司事实和真实市场背景的
尽职调查报告。检查点分为：

- internal：只在私有语料库中陈述的事实（融资、定价、车队指标、路线图）
- web：从公开来源核验的真实市场事实（竞争者融资/IPO 状态）

每份报告由 LLM 按检查点（CORRECT/PARTIAL/MISSING/WRONG）评分，
仅判断与所提供真值的一致性。

用法（从仓库根目录）：
    python deep_agents/hybrid_benchmark.py
"""

from dotenv import load_dotenv
import argparse
import asyncio
import json
import os
import shutil
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

load_dotenv()

DOCS_DIR = Path(__file__).parent / "benchmark_data" / "internal_docs"
os.environ["DOC_PATH"] = str(DOCS_DIR)

from langchain_openai import ChatOpenAI
from deepagents import create_deep_agent
from deepagents.backends import FilesystemBackend

from deep_agents.benchmark import (
    SYSTEM_PROMPT,
    BASELINE_TOOL_GUIDE,
    build_baseline_agent,
    build_gptr_agent,
)
from deep_agents.report_benchmark import write_report
from deep_agents.recency_benchmark import grade_report

RESULTS_DIR = Path(__file__).parent / "benchmark_results"

FILES_GUIDE = """

## 内部文档

此任务的内部公司文档已挂载到你的文件工具上：
用 `ls` 浏览目录，用 `read_file` 读取文档。它们
是内部公司事实的唯一来源。当同一报告存在多个
年份版本时，优先使用最新的文档。"""


def build_baseline_files_agent(model: str):
    """带网页搜索且文件工具挂载到语料库的现成 deepagents。

    语料库被复制到临时目录，使智能体不能修改原件
    （deepagents 还暴露了 write_file/edit_file）。
    """
    workdir = Path(tempfile.mkdtemp(prefix="baseline_files_"))
    shutil.copytree(DOCS_DIR, workdir, dirs_exist_ok=True)

    # 与纯 baseline 相同的搜索工具。
    from tavily import TavilyClient

    tavily_client = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])

    def internet_search(
        query: str,
        max_results: int = 5,
        topic: str = "general",
        include_raw_content: bool = False,
    ):
        """Run a web search. Keep queries short (under 400 characters).

        topic must be one of "general", "news" or "finance".
        """
        if topic not in ("general", "news", "finance"):
            topic = "general"
        return tavily_client.search(
            query[:400],
            max_results=max_results,
            include_raw_content=include_raw_content,
            topic=topic,
        )

    return create_deep_agent(
        model=model,
        tools=[internet_search],
        system_prompt=SYSTEM_PROMPT + BASELINE_TOOL_GUIDE + FILES_GUIDE,
        backend=FilesystemBackend(root_dir=str(workdir), virtual_mode=True),
    )

TASK = {
    "id": "veltrix_dd",
    "prompt": (
        "You are preparing a due-diligence brief on Veltrix Dynamics B.V., a "
        "Rotterdam-based warehouse robotics (AMR) company we are evaluating. "
        "Internal company documents are available to your research tools if "
        "they support document research. Write a brief covering: (1) the "
        "company's funding history and current financial trajectory, (2) its "
        "product line and 2026 roadmap including pricing, (3) operational "
        "performance and key risks, and (4) how it is positioned against the "
        "current competitive landscape in warehouse AMRs (Geek+, Locus "
        "Robotics, Exotec), using up-to-date market facts. Be specific about "
        "numbers, dates and names, and cite your sources."
    ),
    # 仅存在于私有语料库中的事实。
    "internal_checkpoints": [
        "Veltrix has raised $142M total, most recently an $85M Series C in October 2025 led by Meridian Growth Partners at a $610M post-money valuation.",
        "Q1 2026 revenue was $21.4M (+67% YoY) with RaaS ARR of $46.9M; the 2026 revenue target is $96M.",
        "The current VLX-400 lists at $48,500 per robot, with RaaS at $1,950 per robot per month.",
        "The VLX-500 launches in Q4 2026 at a $56,000 list price, with 950 kg payload and a -30°C cold rating.",
        "The deployed fleet is 3,120 robots across 47 sites, with 99.2% availability in Q1 2026.",
        "Key risks include customer concentration (top 3 customers = 44% of ARR) and the VLX-500's solid-state LiDAR cold-chamber certification (in test cycle 2 of 3), which could slip the launch to Q2 2027.",
        "Veltrix's moat is cold-chain: certified operation down to -25°C today, 61% of ARR from frozen/chilled distribution centers, and a 71% win rate in cold-chain RFPs.",
        "A February 11, 2026 firmware regression (v4.7.2) caused localization drift at 14 sites and was resolved within 36 hours with no SLA penalties.",
    ],
    # 从公开来源（2026 年 7 月）核验的真实市场事实。
    "web_checkpoints": [
        "Geek+ (Geekplus) went public on the Hong Kong Stock Exchange in July 2025 - the first IPO in the AMR warehouse robotics market - raising roughly $280M at about a $2.8B market cap.",
        "Geek+ is the global warehouse-fulfillment AMR market share leader by revenue (per Interact Analysis, for roughly seven consecutive years).",
        "Locus Robotics remains private, with total funding of roughly $410-440M and 13,000+ robots deployed.",
        "Exotec is a French warehouse robotics unicorn known for its Skypod ASRS system and remains private (roughly $446M raised).",
    ],
    "official_domains": [],
}


def coverage(grades: list[dict]) -> dict:
    gs = [g["grade"] for g in grades]
    n = len(gs)
    return {
        "correct": gs.count("CORRECT"),
        "partial": gs.count("PARTIAL"),
        "missing": gs.count("MISSING"),
        "wrong": gs.count("WRONG"),
        "coverage": round((gs.count("CORRECT") + 0.5 * gs.count("PARTIAL")) / n, 3) if n else 0.0,
    }


async def main() -> None:
    parser = argparse.ArgumentParser(description="Hybrid-research benchmark: private docs + web vs web-only")
    parser.add_argument("--model", type=str, default=os.environ.get("STRATEGIC_LLM", "openai:gpt-5.4"))
    parser.add_argument("--grader-model", type=str, default="gpt-5.4")
    args = parser.parse_args()

    grader = ChatOpenAI(model=args.grader_model)
    baseline_agent, _ = build_baseline_agent(args.model)
    baseline_files_agent = build_baseline_files_agent(args.model)
    gptr_agent, gptr_costs = build_gptr_agent(args.model, source="hybrid")

    print(f"Hybrid benchmark: model {args.model}, grader {args.grader_model}")
    print(f"Internal corpus: {DOCS_DIR}\n")

    record = {"task": TASK["id"], "prompt": TASK["prompt"]}
    for system, agent in (
        ("baseline", baseline_agent),
        ("baseline+files", baseline_files_agent),
        ("gptr", gptr_agent),
    ):
        print(f"[{system}] writing...")
        outcome = await write_report(agent, TASK["prompt"])
        internal = await asyncio.to_thread(
            grade_report, grader, {"checkpoints": TASK["internal_checkpoints"]}, outcome["report"]
        )
        web = await asyncio.to_thread(
            grade_report, grader, {"checkpoints": TASK["web_checkpoints"]}, outcome["report"]
        )
        record[system] = {
            "report": outcome["report"],
            "latency_seconds": outcome["latency_seconds"],
            "internal_grades": internal,
            "web_grades": web,
            "internal": coverage(internal),
            "web": coverage(web),
        }
        print(f"[{system}] internal: {record[system]['internal']}")
        print(f"[{system}] web:      {record[system]['web']}")

    summary = {
        s: {
            "internal_coverage": record[s]["internal"]["coverage"],
            "web_coverage": record[s]["web"]["coverage"],
            "latency_seconds": record[s]["latency_seconds"],
        }
        for s in ("baseline", "baseline+files", "gptr")
    }
    summary["gptr_internal_cost_usd"] = round(gptr_costs.get("total", 0.0), 4)
    print("\n=== SUMMARY ===")
    print(json.dumps(summary, indent=2))

    RESULTS_DIR.mkdir(exist_ok=True)
    results_path = RESULTS_DIR / f"hybrid_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump({
            "run_metadata": {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "model": args.model,
                "grader_model": args.grader_model,
                "doc_path": str(DOCS_DIR),
            },
            "summary": summary,
            "records": [record],
        }, f, indent=2)
    print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    asyncio.run(main())
