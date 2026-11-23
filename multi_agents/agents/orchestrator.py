import os
import time
import datetime
from langgraph.graph import StateGraph, END
# from langgraph.checkpoint.memory import MemorySaver
from .utils.views import print_agent_output
from ..memory.research import ResearchState
from .utils.utils import sanitize_filename
from .plan_review import (
    DEFAULT_MAX_PLAN_REVISIONS,
    route_human_feedback,
)
from .fact_review import (
    DEFAULT_MAX_FACT_CHECK_REVISIONS,
    MaxFactCheckRevisionsExceededError,
    route_fact_check,
)

# 导入智能体类
from . import \
    WriterAgent, \
    EditorAgent, \
    PublisherAgent, \
    ResearchAgent, \
    HumanAgent, \
    FactCheckerAgent, \
    VisualizerAgent


class ChiefEditorAgent:
    """负责管理和协调编辑任务的智能体。"""

    def __init__(self, task: dict, websocket=None, stream_output=None, tone=None, headers=None):
        self.task = task
        self.websocket = websocket
        self.stream_output = stream_output
        self.headers = headers or {}
        self.tone = tone
        self.task_id = self._generate_task_id()
        self.output_dir = self._create_output_directory()

    def _generate_task_id(self):
        # 目前基于时间，但可以是任何唯一标识符
        return int(time.time())

    def _create_output_directory(self):
        output_dir = "./outputs/" + \
            sanitize_filename(
                f"run_{self.task_id}_{self.task.get('query')[0:40].strip()}")

        os.makedirs(output_dir, exist_ok=True)
        return output_dir

    def _initialize_agents(self):
        return {
            "writer": WriterAgent(self.websocket, self.stream_output, self.headers),
            "editor": EditorAgent(self.websocket, self.stream_output, self.tone, self.headers),
            "research": ResearchAgent(self.websocket, self.stream_output, self.tone, self.headers),
            "publisher": PublisherAgent(self.output_dir, self.websocket, self.stream_output, self.headers),
            "human": HumanAgent(self.websocket, self.stream_output, self.headers),
            "fact_checker": FactCheckerAgent(self.websocket, self.stream_output, self.headers),
            "visualizer": VisualizerAgent(self.websocket, self.stream_output, self.headers)
        }

    def _create_workflow(self, agents):
        workflow = StateGraph(ResearchState)

        # 为每个智能体添加节点
        workflow.add_node("browser", agents["research"].run_initial_research)
        workflow.add_node("planner", agents["editor"].plan_research)
        workflow.add_node("researcher", agents["editor"].run_parallel_research)
        workflow.add_node("writer", agents["writer"].run)
        workflow.add_node("fact_checker", agents["fact_checker"].run)
        workflow.add_node("visualizer", agents["visualizer"].run)
        workflow.add_node("publisher", agents["publisher"].run)
        workflow.add_node("human", agents["human"].review_plan)

        # 添加边
        self._add_workflow_edges(workflow)

        return workflow

    def _add_workflow_edges(self, workflow):
        workflow.add_edge('browser', 'planner')
        workflow.add_edge('planner', 'human')
        workflow.add_edge('researcher', 'writer')
        workflow.add_edge('writer', 'fact_checker')
        workflow.add_edge('visualizer', 'publisher')
        workflow.set_entry_point("browser")
        workflow.add_edge('publisher', END)

        # 人工回路：精确的"否"批准（human 智能体）+ 通过 plan_review.route_human_feedback
        # 有界的计划修订（task.max_plan_revisions）。
        workflow.add_conditional_edges(
            'human',
            self._route_human_feedback,
            {"accept": "researcher", "revise": "planner"},
        )

        # 事实核查器回路 —— 通过 task.max_fact_check_revisions 有界
        workflow.add_conditional_edges(
            'fact_checker',
            self._route_fact_check,
            {"accept": "visualizer", "revise": "writer"}
        )

    def _route_human_feedback(self, review):
        """路由人工计划反馈；超过 max_plan_revisions 后强制接受。

        ``route_human_feedback`` 在超过配置的上限时抛出异常。
        对于图的边，我们将其视为接受，使运行进行到研究
        而不是死在 LangGraph 默认的 recursion_limit 上。
        """
        from .plan_review import MaxPlanRevisionsExceededError

        max_plan_revisions = self.task.get(
            "max_plan_revisions", DEFAULT_MAX_PLAN_REVISIONS)
        try:
            return route_human_feedback(review, max_plan_revisions)
        except MaxPlanRevisionsExceededError:
            return "accept"

    def _route_fact_check(self, state):
        """路由事实核查结果；超过上限后强制接受。

        ``route_fact_check`` 在超过 max_fact_check_revisions 时抛出异常。
        与人工回路一样，边将其视为接受，使运行
        继续而不是死在 LangGraph 的 recursion_limit 上。
        """
        max_fact_check_revisions = self.task.get(
            "max_fact_check_revisions", DEFAULT_MAX_FACT_CHECK_REVISIONS)
        try:
            return route_fact_check(state, max_fact_check_revisions)
        except MaxFactCheckRevisionsExceededError:
            return "accept"

    def init_research_team(self):
        """初始化并为研究团队创建工作流。"""
        agents = self._initialize_agents()
        return self._create_workflow(agents)

    async def _log_research_start(self):
        message = f"Starting the research process for query '{self.task.get('query')}'..."
        if self.websocket and self.stream_output:
            await self.stream_output("logs", "starting_research", message, self.websocket)
        else:
            print_agent_output(message, "MASTER")

    async def run_research_task(self, task_id=None):
        """
        用已初始化的研究团队运行研究任务。

        参数：
            task_id (可选): 要运行的任务 ID。

        返回：
            研究任务的结果。
        """
        research_team = self.init_research_team()
        chain = research_team.compile()

        await self._log_research_start()

        config = {
            "configurable": {
                "thread_id": task_id,
                "thread_ts": datetime.datetime.utcnow()
            }
        }

        result = await chain.ainvoke({"task": self.task}, config=config)
        return result
