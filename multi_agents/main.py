from dotenv import load_dotenv
import sys
import os
import uuid

# 在评估下面的可选追踪/插桩开关之前，先加载项目本地环境配置。
load_dotenv()

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# 可选的 Monocle 可观测性，由 MONOCLE_TRACING 控制（与下面的 LangSmith
# 开关一致）。在框架导入之前运行，使 Monocle 能在它们加载时进行插桩。
# 应用拥有 MONOCLE_EXPORTERS：它验证选择后，把原始逗号分隔字符串
# 转发给 Monocle。未设置时为空操作。
if os.environ.get("MONOCLE_TRACING", "").strip().lower() in ("1", "true", "yes", "on"):
    _exporters = os.environ.get("MONOCLE_EXPORTERS", "").strip() or "file"
    _allowed = ("file", "console", "okahu", "s3", "blob", "gcs")
    _selected = [e.strip() for e in _exporters.split(",") if e.strip()]
    _unknown = [e for e in _selected if e not in _allowed]
    if _unknown:
        raise ValueError(
            f"MONOCLE_EXPORTERS has unknown exporter(s): {', '.join(_unknown)}. "
            f"Allowed: {', '.join(_allowed)}."
        )
    if "okahu" in _selected and not os.environ.get("OKAHU_API_KEY"):
        raise ValueError("Monocle 'okahu' exporter is selected but OKAHU_API_KEY is not set.")
    try:
        from monocle_apptrace import setup_monocle_telemetry
    except ImportError as exc:
        raise RuntimeError(
            "MONOCLE_TRACING is enabled but monocle_apptrace is not installed. "
            'Install the "monocle" extra: pip install "gpt-researcher[monocle]".'
        ) from exc
    setup_monocle_telemetry(workflow_name="gpt-researcher", monocle_exporters_list=_exporters)

from multi_agents.agents import ChiefEditorAgent
import asyncio
import json
from gpt_researcher.utils.enum import Tone

# 如果设置了 API 密钥，则使用 LangSmith 运行
if os.environ.get("LANGCHAIN_API_KEY"):
    os.environ["LANGCHAIN_TRACING_V2"] = "true"

def open_task():
    # 获取当前脚本所在目录
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # 构建 task.json 的绝对路径
    task_json_path = os.path.join(current_dir, 'task.json')
    
    with open(task_json_path, 'r') as f:
        task = json.load(f)

    if not task:
        raise Exception("No task found. Please ensure a valid task.json file is present in the multi_agents directory and contains the necessary task information.")

    # 如果环境定义了 STRATEGIC_LLM，则用它覆盖模型
    strategic_llm = os.environ.get("STRATEGIC_LLM")
    if strategic_llm and ":" in strategic_llm:
        # 提取模型名称（第一个冒号之后的部分）
        model_name = strategic_llm.split(":", 1)[1]
        task["model"] = model_name
    elif strategic_llm:
        task["model"] = strategic_llm

    return task

async def run_research_task(query, websocket=None, stream_output=None, tone=Tone.Objective, headers=None):
    task = open_task()
    task["query"] = query

    chief_editor = ChiefEditorAgent(task, websocket, stream_output, tone, headers)
    research_report = await chief_editor.run_research_task()

    if websocket and stream_output:
        await stream_output("logs", "research_report", research_report, websocket)

    return research_report

async def main():
    task = open_task()

    chief_editor = ChiefEditorAgent(task)
    research_report = await chief_editor.run_research_task(task_id=uuid.uuid4())

    return research_report

if __name__ == "__main__":
    asyncio.run(main())
