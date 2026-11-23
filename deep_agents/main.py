from dotenv import load_dotenv
import sys
import os
import json
import uuid
import asyncio

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

load_dotenv()

if os.environ.get("LANGCHAIN_API_KEY"):
    os.environ["LANGCHAIN_TRACING_V2"] = "true"

from deep_agents.agent import build_agent

# 任务配置固定从本目录的 task.json 读取


def open_task():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    task_json_path = os.path.join(current_dir, "task.json")

    with open(task_json_path, "r") as f:
        task = json.load(f)

    if not task:
        raise Exception(
            "No task found. Please ensure a valid task.json file is present in "
            "the deep_agents directory and contains the necessary task information."
        )
    return task


def build_request(task: dict) -> str:
    """把任务字典拼装成发送给智能体自然语言研究请求"""
    return (
        f"Research the following topic and produce a report: {task['query']}\n"
        f"The report should have at most {task.get('max_sections', 3)} sections."
    )


def print_progress(update: dict):
    """把流式更新的工具调用与文本消息打印为一行行进度信息"""
    for node, payload in update.items():
        if not isinstance(payload, dict):
            continue
        for message in payload.get("messages", []):
            for tool_call in getattr(message, "tool_calls", None) or []:
                name = tool_call.get("name", "")
                args = tool_call.get("args", {})
                if name == "task":
                    detail = f"{args.get('subagent_type', '')}: {args.get('description', '')}"
                elif name == "write_todos":
                    detail = ", ".join(t.get("content", "") for t in args.get("todos", []))
                elif name in ("write_file", "edit_file", "read_file"):
                    detail = args.get("file_path", "")
                else:
                    detail = str(args.get("query", ""))[:120]
                print(f"  [{node}] -> {name}({detail})")
            text = getattr(message, "text", None)
            if text is not None and not isinstance(text, str) and callable(text):
                text = text()
            if text and getattr(message, "type", "") == "ai":
                print(f"  [{node}] {text}")


async def run_research_task(query, task=None):
    """运行一次完整研究任务，返回落盘的 report.md 文本"""
    task = task or open_task()
    task["query"] = query

    # 每次运行使用独立的输出目录，避免相互覆盖
    run_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "outputs", f"run_{uuid.uuid4().hex[:8]}"
    )
    os.makedirs(run_dir, exist_ok=True)
    print(f"Output directory: {run_dir}")

    agent = build_agent(task, run_dir)

    # stream_mode="updates" 表示只推送发生变化的节点
    async for update in agent.astream(
        {"messages": [{"role": "user", "content": build_request(task)}]},
        config={"recursion_limit": 200},
        stream_mode="updates",
    ):
        if task.get("verbose", True):
            print_progress(update)

    # 主智能体负责把最终报告写到 report.md
    report_path = os.path.join(run_dir, "report.md")
    if os.path.exists(report_path):
        print(f"\nReport written to {report_path}")
        with open(report_path, "r") as f:
            return f.read()

    print("\nNo report.md was produced - check the run output above.")
    return None


async def main():
    """命令行入口：读取本地任务并执行"""
    task = open_task()
    return await run_research_task(task["query"], task)


if __name__ == "__main__":
    asyncio.run(main())
