import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from gpt_researcher import GPTResearcher


def build_research_tools(task: dict, cost_tracker: dict | None = None):
    """构建研究工具，并把任务的 source 配置绑定进去。

    报告来源（"web"、"local" 或 "hybrid"）是任务配置，
    不是智能体每次调用应决定的东西，因此这里把它闭包捕获，
    而不是作为工具参数暴露。对 "local" 和 "hybrid"，把
    DOC_PATH 环境变量设为你的文档目录。

    传入一个字典作为 cost_tracker，即可在其 "total" 键下累积
    GPT Researcher API 成本。
    """
    report_source = task.get("source", "web")

    def _track_cost(researcher: GPTResearcher):
        if cost_tracker is not None:
            cost_tracker["total"] = cost_tracker.get("total", 0.0) + researcher.get_costs()

    async def quick_search(query: str) -> str:
        """执行快速网页搜索，返回带摘要和 URL 的顶部结果。

        用于确定话题范围、核实一个事实，或收集刚好够规划报告大纲的
        上下文。结果是原始摘要 —— 如果首次搜索没有浮现所需内容，
        用细化查询迭代。如需对某章节做深入的、引用级别的研究，
        改用 deep_research。
        """
        researcher = GPTResearcher(query=query, report_type="research_report", verbose=False)
        results = await researcher.quick_search(query)
        _track_cost(researcher)
        if isinstance(results, list):
            formatted = "\n\n".join(
                f"[{i}] {r.get('title') or ''}\n{r.get('body') or r.get('content') or ''}\n"
                f"URL: {r.get('href') or r.get('url') or ''}".strip()
                for i, r in enumerate(results, 1)
            )
            return formatted or "No results found."
        return results or "No results found."

    async def deep_research(query: str, parent_query: str = "") -> str:
        """对查询运行完整的 GPT Researcher 管线。

        规划子查询、收集并校验数十个来源（网页和/或本地文档，
        取决于任务配置），并返回一份带行内引用和参考文献列表的
        详细 Markdown 研究报告。此过程较慢（几分钟）但彻底 ——
        每个报告章节用一次，不要用于快速查找。

        参数：
            query: 要调查的研究问题或章节主题。
            parent_query: 当研究更大报告的某个章节时的整体报告主题。
                使研究在上下文中保持聚焦。
        """
        report_type = "subtopic_report" if parent_query else "research_report"
        researcher = GPTResearcher(
            query=query,
            report_type=report_type,
            report_source=report_source,
            parent_query=parent_query,
            verbose=False,
        )
        await researcher.conduct_research()
        report = await researcher.write_report()
        _track_cost(researcher)

        sources = researcher.get_source_urls()
        if sources:
            report += "\n\n### Sources\n\n" + "\n".join(f"- {url}" for url in sources)
        return report

    return quick_search, deep_research
