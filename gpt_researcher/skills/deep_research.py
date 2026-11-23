from typing import List, Dict, Any, Optional, Set
import asyncio
import logging
import re
import time
from datetime import datetime, timedelta

import json_repair

from gpt_researcher.llm_provider.generic.base import ReasoningEfforts
from ..utils.llm import create_chat_completion
from ..utils.enum import ReportType, ReportSource, Tone
from ..actions.query_processing import get_search_results

logger = logging.getLogger(__name__)

# 上下文中允许的最大词数（25k 词，留安全余量）
MAX_CONTEXT_WORDS = 25000

JSON_BLOCK_PATTERNS = [
    re.compile(
        r"```(?:json)?\s*(?P<payload>[\s\S]*?)```",
        re.IGNORECASE,
    ),
    re.compile(r"(?P<payload>\[[\s\S]*\])"),
    re.compile(r"(?P<payload>\{[\s\S]*\})"),
]

QUERY_LINE_PATTERN = re.compile(
    r"^(?:[-*]|\d+[.)])?\s*Query:\s*(?P<query>.+)$",
    re.IGNORECASE,
)
GOAL_LINE_PATTERN = re.compile(
    r"^(?:[-*]|\d+[.)])?\s*(?:Goal|Research Goal):\s*(?P<goal>.+)$",
    re.IGNORECASE,
)
QUESTION_LINE_PATTERN = re.compile(
    r"^(?:[-*]|\d+[.)])?\s*(?:Question:\s*)?(?P<question>.+\?)$",
    re.IGNORECASE,
)
LEARNING_LINE_PATTERN = re.compile(
    r"^(?:[-*]|\d+[.)])?\s*Learning(?:\s*\[(?P<citation>[^\]]+)\])?:\s*(?P<learning>.+)$",
    re.IGNORECASE,
)
URL_PATTERN = re.compile(r"https?://[^\s\]\)>\",;]+")


def _extract_json_payloads(response: str) -> list[str]:
    candidates: list[str] = []
    seen: set[str] = set()

    for pattern in JSON_BLOCK_PATTERNS:
        for match in pattern.finditer(response):
            candidate = match.group("payload").strip()
            if candidate and candidate not in seen:
                candidates.append(candidate)
                seen.add(candidate)

    return candidates


def _load_repaired_json(response: str) -> Any:
    for candidate in [response.strip(), *_extract_json_payloads(response)]:
        if not candidate:
            continue
        try:
            return json_repair.loads(candidate)
        except Exception as exc:
            logger.debug(
                "json_repair failed on candidate (%d chars): %s",
                len(candidate), exc,
            )
            continue
    return None


def parse_search_queries_response(response: str, num_queries: int) -> List[Dict[str, str]]:
    parsed = _load_repaired_json(response)
    candidate_queries = parsed
    if isinstance(parsed, dict):
        candidate_queries = parsed.get("queries") or parsed.get("searchQueries") or parsed.get("items")

    if isinstance(candidate_queries, list):
        queries = []
        for item in candidate_queries:
            if not isinstance(item, dict):
                continue
            query = item.get("query")
            research_goal = item.get("researchGoal")
            if not isinstance(query, str) or not isinstance(research_goal, str):
                continue
            # 先规范化再校验，使只含空白的值被拒绝，而不是被接受后
            # 又被裁剪成 ""。
            query = query.strip()
            research_goal = research_goal.strip()
            if query and research_goal:
                queries.append({"query": query, "researchGoal": research_goal})
        if queries:
            return queries[:num_queries]

    queries: List[Dict[str, str]] = []
    current_query: Dict[str, str] = {}

    for raw_line in response.replace("```json", "").replace("```", "").splitlines():
        line = raw_line.strip()
        if not line:
            continue

        query_match = QUERY_LINE_PATTERN.match(line)
        goal_match = GOAL_LINE_PATTERN.match(line)

        if query_match:
            if current_query.get("query") and current_query.get("researchGoal"):
                queries.append(current_query)
            current_query = {"query": query_match.group("query").strip()}
        elif goal_match and current_query.get("query"):
            current_query["researchGoal"] = goal_match.group("goal").strip()

    if current_query.get("query") and current_query.get("researchGoal"):
        queries.append(current_query)

    return queries[:num_queries]


def _question_strings(value: Any) -> List[str]:
    """读取提示词指定的字符串列表形状，不强制转换无效条目。"""
    if not isinstance(value, list):
        return []
    return [item.strip() for item in value if isinstance(item, str) and item.strip()]


def parse_follow_up_questions_response(response: str, num_questions: int) -> List[str]:
    parsed = _load_repaired_json(response)
    candidate_questions = parsed
    if isinstance(parsed, dict):
        candidate_questions = parsed.get("questions") or parsed.get("followUpQuestions") or parsed.get("items")

    if isinstance(candidate_questions, list):
        questions = _question_strings(candidate_questions)
        if questions:
            return questions[:num_questions]

    questions: List[str] = []
    for raw_line in response.replace("```json", "").replace("```", "").splitlines():
        line = raw_line.strip()
        if not line:
            continue

        question_match = QUESTION_LINE_PATTERN.match(line)
        if question_match:
            questions.append(question_match.group("question").strip())

    return questions[:num_questions]


def parse_research_results_response(response: str, num_learnings: int) -> Dict[str, Any]:
    parsed = _load_repaired_json(response)

    if isinstance(parsed, dict):
        learnings_payload = parsed.get("learnings", [])
        follow_up_payload = parsed.get("followUpQuestions") or parsed.get("questions") or []
        learnings: List[str] = []
        citations: Dict[str, str] = {}

        if isinstance(learnings_payload, list):
            for item in learnings_payload:
                if isinstance(item, dict):
                    learning = str(item.get("insight") or item.get("learning") or "").strip()
                    citation = str(item.get("sourceUrl") or item.get("citation") or "").strip()
                else:
                    learning = str(item).strip()
                    citation = ""

                if learning:
                    learnings.append(learning)
                    if citation:
                        citations[learning] = citation

        questions = _question_strings(follow_up_payload)
        if learnings or questions:
            return {
                "learnings": learnings[:num_learnings],
                "followUpQuestions": questions[:num_learnings],
                "citations": citations,
            }

    learnings: List[str] = []
    questions: List[str] = []
    citations: Dict[str, str] = {}

    for raw_line in response.replace("```json", "").replace("```", "").splitlines():
        line = raw_line.strip()
        if not line:
            continue

        learning_match = LEARNING_LINE_PATTERN.match(line)
        question_match = QUESTION_LINE_PATTERN.match(line)

        if learning_match:
            learning = learning_match.group("learning").strip()
            citation = (learning_match.group("citation") or "").strip()
            if not citation:
                url_match = URL_PATTERN.search(learning)
                if url_match:
                    citation = url_match.group(0)
                    learning = learning.replace(citation, "").strip(" -")
            if learning:
                learnings.append(learning)
                if citation:
                    citations[learning] = citation
        elif question_match:
            questions.append(question_match.group("question").strip())

    return {
        "learnings": learnings[:num_learnings],
        "followUpQuestions": questions[:num_learnings],
        "citations": citations,
    }

def count_words(text) -> int:
    """统计文本字符串的词数。同时处理字符串和列表。"""
    if isinstance(text, list):
        text = " ".join(str(item) for item in text)
    return len(str(text).split())

def trim_context_to_word_limit(context_list: List[str], max_words: int = MAX_CONTEXT_WORDS) -> List[str]:
    """在词数限制内裁剪上下文列表，同时保留最近的/最相关的条目"""
    total_words = 0
    trimmed_context = []

    # 逆序处理以保留最近的条目
    for item in reversed(context_list):
        text = " ".join(str(part) for part in item) if isinstance(item, list) else str(item)
        words = count_words(item)
        if total_words + words <= max_words:
            trimmed_context.insert(0, item)  # Insert at start to maintain original order
            total_words += words
        elif not trimmed_context:
            trimmed_context.insert(0, " ".join(text.split()[:max_words]))
            break
        else:
            break

    return trimmed_context

class ResearchProgress:
    def __init__(self, total_depth: int, total_breadth: int):
        self.current_depth = 1  # 从 1 开始，递增到 total_depth
        self.total_depth = total_depth
        self.current_breadth = 0  # 从 0 开始，随查询完成递增到 total_breadth
        self.total_breadth = total_breadth
        self.current_query: Optional[str] = None
        self.total_queries = 0
        self.completed_queries = 0


class DeepResearchSkill:
    def __init__(self, researcher):
        self.researcher = researcher
        self.breadth = getattr(researcher.cfg, 'deep_research_breadth', 4)
        self.depth = getattr(researcher.cfg, 'deep_research_depth', 2)
        self.concurrency_limit = getattr(researcher.cfg, 'deep_research_concurrency', 2)
        self.websocket = researcher.websocket
        self.tone = researcher.tone
        self.config_path = researcher.cfg.config_path if hasattr(researcher.cfg, 'config_path') else None
        self.headers = researcher.headers or {}
        self.visited_urls = researcher.visited_urls
        self.learnings = []
        self.research_sources = []  # 跟踪所有研究来源
        self.context = []  # 跟踪所有上下文

    async def generate_search_queries(self, query: str, num_queries: int = 3) -> List[Dict[str, str]]:
        """为研究生成 SERP 查询"""
        messages = [
            {
                "role": "system",
                "content": (
                    "You are an expert researcher generating search queries. "
                    "Return valid JSON only. Do not include markdown, code fences, bullets, numbering, or prose."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Given the following prompt, generate {num_queries} unique search queries to research the topic thoroughly. "
                    "For each query, provide a research goal.\n\n"
                    "Return ONLY a JSON array of objects using this exact schema:\n"
                    '[{"query": "<search query>", "researchGoal": "<research goal>"}]\n\n'
                    f"Prompt: {query}"
                ),
            },
        ]

        response = await create_chat_completion(
            messages=messages,
            llm_provider=self.researcher.cfg.strategic_llm_provider,
            model=self.researcher.cfg.strategic_llm_model,
            reasoning_effort=self.researcher.cfg.reasoning_effort,
            temperature=0.4,
            llm_kwargs=self.researcher.cfg.llm_kwargs,
            cost_callback=getattr(self.researcher, "add_costs", None),
        )

        return parse_search_queries_response(response, num_queries)

    async def generate_research_plan(self, query: str, num_questions: int = 3) -> List[str]:
        """生成后续问题以澄清研究方向"""
        # 从所有检索器获取初始搜索结果以辅助查询生成
        all_search_results = []
        for retriever in self.researcher.retrievers:
            try:
                results = await get_search_results(
                    query,
                    retriever,
                    researcher=self.researcher
                )
                all_search_results.extend(results)
            except Exception as e:
                logger.warning(f"Error with retriever {retriever.__name__}: {e}")
        search_results = all_search_results
        logger.info(f"Initial web knowledge obtained: {len(search_results)} results")

        # 获取当前时间作为上下文
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        messages = [
            {
                "role": "system",
                "content": (
                    "You are an expert researcher. Your task is to analyze the original query and search results, "
                    "then generate targeted questions that explore different aspects and time periods of the topic. "
                    "Return valid JSON only."
                ),
            },
            {"role": "user",
             "content": f"""Original query: {query}

Current time: {current_time}

Search results:
{search_results}

Based on these results, the original query, and the current time, generate {num_questions} unique questions. Each question should explore a different aspect or time period of the topic, considering recent developments up to {current_time}.

Return ONLY a JSON object using this exact schema:
{{"questions": ["<question 1>", "<question 2>"]}}"""}
        ]

        response = await create_chat_completion(
            messages=messages,
            llm_provider=self.researcher.cfg.strategic_llm_provider,
            model=self.researcher.cfg.strategic_llm_model,
            reasoning_effort=ReasoningEfforts.High.value,
            temperature=0.4,
            llm_kwargs=self.researcher.cfg.llm_kwargs,
            cost_callback=getattr(self.researcher, "add_costs", None),
        )

        return parse_follow_up_questions_response(response, num_questions)

    async def process_research_results(self, query: str, context: str, num_learnings: int = 3) -> Dict[str, List[str]]:
        """处理研究结果以提取经验教训和后续问题"""
        messages = [
            {
                "role": "system",
                "content": (
                    "You are an expert researcher analyzing search results. "
                    "Return valid JSON only."
                ),
            },
            {"role": "user",
             "content": (
                 f"Given the following research results for the query '{query}', extract key learnings and suggest "
                 "follow-up questions. For each learning, include a citation to the source URL if available.\n\n"
                 "Return ONLY a JSON object using this exact schema:\n"
                 '{"learnings": [{"insight": "<insight>", "sourceUrl": "<url or empty string>"}], '
                 '"followUpQuestions": ["<question 1>", "<question 2>"]}\n\n'
                 f"Research results:\n{context}"
             )}
        ]

        response = await create_chat_completion(
            messages=messages,
            llm_provider=self.researcher.cfg.strategic_llm_provider,
            model=self.researcher.cfg.strategic_llm_model,
            temperature=0.4,
            reasoning_effort=ReasoningEfforts.High.value,
            # 为推理模型的推理 token 预留余量
            max_tokens=getattr(self.researcher.cfg, "strategic_token_limit", 8000),
            llm_kwargs=self.researcher.cfg.llm_kwargs,
            cost_callback=getattr(self.researcher, "add_costs", None),
        )

        return parse_research_results_response(response, num_learnings)

    async def deep_research(
            self,
            query: str,
            breadth: int,
            depth: int,
            learnings: List[str] = None,
            citations: Dict[str, str] = None,
            visited_urls: Set[str] = None,
            on_progress=None
    ) -> Dict[str, Any]:
        """执行深度迭代研究"""
        print(f"\n📊 DEEP RESEARCH: depth={depth}, breadth={breadth}, query={query[:100]}...", flush=True)
        if learnings is None:
            learnings = []
        if citations is None:
            citations = {}
        if visited_urls is None:
            visited_urls = set()

        all_learnings = learnings.copy()
        all_citations = citations.copy()
        all_visited_urls = visited_urls.copy()
        all_context = []
        all_sources = []

        progress = ResearchProgress(depth, breadth)

        if on_progress:
            on_progress(progress)

        # 生成搜索查询
        print(f"🔎 Generating {breadth} search queries...", flush=True)
        serp_queries = await self.generate_search_queries(query, num_queries=breadth)
        print(f"✅ Generated {len(serp_queries)} queries: {[q['query'] for q in serp_queries]}", flush=True)
        progress.total_queries = len(serp_queries)
        if not serp_queries:
            logger.warning("Deep research generated zero search queries; stopping descent.")
            return {
                'learnings': all_learnings,
                'visited_urls': all_visited_urls,
                'citations': all_citations,
                'context': all_context,
                'sources': all_sources,
            }

        # 带并发限制地处理查询
        semaphore = asyncio.Semaphore(self.concurrency_limit)

        async def process_query(serp_query: Dict[str, str]) -> Optional[Dict[str, Any]]:
            async with semaphore:
                try:
                    progress.current_query = serp_query['query']
                    if on_progress:
                        on_progress(progress)

                    from .. import GPTResearcher
                    researcher = GPTResearcher(
                        query=serp_query['query'],
                        report_type=ReportType.ResearchReport.value,
                        report_source=ReportSource.Web.value,
                        tone=self.tone,
                        websocket=self.websocket,
                        config_path=self.config_path,
                        headers=self.headers,
                        visited_urls=self.visited_urls,
                        # 将 MCP 配置传播给嵌套研究员
                        mcp_configs=self.researcher.mcp_configs,
                        mcp_strategy=self.researcher.mcp_strategy
                    )

                    # 执行研究
                    context = await researcher.conduct_research()
                    self.researcher.add_costs(researcher.get_costs())

                    # 获取结果和已访问 URL
                    visited = researcher.visited_urls
                    sources = researcher.research_sources

                    # 处理结果以提取经验教训和引用
                    results = await self.process_research_results(
                        query=serp_query['query'],
                        context=context
                    )

                    # 更新进度
                    progress.completed_queries += 1
                    progress.current_breadth += 1
                    if on_progress:
                        on_progress(progress)

                    return {
                        'learnings': results['learnings'],
                        'visited_urls': list(visited),
                        'followUpQuestions': results['followUpQuestions'],
                        'researchGoal': serp_query['researchGoal'],
                        'citations': results['citations'],
                        'context': "\n".join(context) if isinstance(context, list) else (context or ""),
                        'sources': sources if sources else []
                    }

                except Exception as e:
                    import traceback
                    error_details = traceback.format_exc()
                    logger.error(f"Error processing query '{serp_query['query']}': {str(e)}")
                    print(f"\n❌ DEEP RESEARCH ERROR: {str(e)}\n{error_details}", flush=True)
                    return None

        # 带并发限制地并发处理查询
        tasks = [process_query(query) for query in serp_queries]
        results = await asyncio.gather(*tasks)
        results = [r for r in results if r is not None]

        # 根据成功的查询更新广度进度
        progress.current_breadth = len(results)
        if on_progress:
            on_progress(progress)

        # #1579: 如果该层的每个分支都失败（密钥损坏、检索器离线等），
        # 就停止，而不是从空目标/空经验教训中无限生成后续问题。
        if not results:
            logger.warning(
                "Deep research produced no successful query results at depth=%s; stopping descent.",
                depth,
            )
            print(
                f"\nDEEP RESEARCH: no successful results at depth={depth}; stopping to avoid infinite work.",
                flush=True,
            )
            return {
                'learnings': all_learnings,
                'visited_urls': all_visited_urls,
                'citations': all_citations,
                'context': all_context,
                'sources': all_sources,
            }

        # 收集所有结果
        for result in results:
            all_learnings.extend(result['learnings'])
            all_visited_urls.update(result['visited_urls'])
            all_citations.update(result['citations'])
            if result['context']:
                # 使用 extend 而非 append：当 CURATE_SOURCES=True 时，
                # result['context'] 是 List[dict]。append() 会把它嵌套为
                # 单个条目，导致后续 "\n".join() 崩溃并抛出
                # "expected str instance, dict found"。
                ctx = result['context']
                if isinstance(ctx, list):
                    all_context.extend(ctx)
                else:
                    all_context.append(ctx)
            if result['sources']:
                all_sources.extend(result['sources'])

            # 如需要则继续深入
            if depth > 1:
                new_breadth = max(2, breadth // 2)
                new_depth = depth - 1
                progress.current_depth += 1

                # 从研究目标和后续问题创建下一查询
                next_query = f"""
                Previous research goal: {result['researchGoal']}
                Follow-up questions: {' '.join(result['followUpQuestions'])}
                """

                # 递归研究
                deeper_results = await self.deep_research(
                    query=next_query,
                    breadth=new_breadth,
                    depth=new_depth,
                    learnings=all_learnings,
                    citations=all_citations,
                    visited_urls=all_visited_urls,
                    on_progress=on_progress
                )

                all_learnings = deeper_results['learnings']
                all_visited_urls.update(deeper_results['visited_urls'])
                all_citations.update(deeper_results['citations'])
                if deeper_results.get('context'):
                    all_context.extend(deeper_results['context'])
                if deeper_results.get('sources'):
                    all_sources.extend(deeper_results['sources'])

        # 更新类级跟踪
        self.context.extend(all_context)
        self.research_sources.extend(all_sources)

        # 裁剪上下文以保持在词数限制内
        trimmed_context = trim_context_to_word_limit(all_context)
        logger.info(f"Trimmed context from {len(all_context)} items to {len(trimmed_context)} items to stay within word limit")

        return {
            'learnings': list(set(all_learnings)),
            'visited_urls': list(all_visited_urls),
            'citations': all_citations,
            'context': trimmed_context,
            'sources': all_sources
        }

    async def run(self, on_progress=None) -> str:
        """运行深度研究过程并生成最终报告"""
        print(f"\n🔍 DEEP RESEARCH: Starting with breadth={self.breadth}, depth={self.depth}, concurrency={self.concurrency_limit}", flush=True)
        start_time = time.time()

        # 记录初始成本
        initial_costs = self.researcher.get_costs()

        follow_up_questions = await self.generate_research_plan(self.researcher.query)
        answers = ["Automatically proceeding with research"] * len(follow_up_questions)

        qa_pairs = [f"Q: {q}\nA: {a}" for q, a in zip(follow_up_questions, answers)]
        combined_query = f"""
        Initial Query: {self.researcher.query}\nFollow - up Questions and Answers:\n
        """ + "\n".join(qa_pairs)

        results = await self.deep_research(
            query=combined_query,
            breadth=self.breadth,
            depth=self.depth,
            on_progress=on_progress
        )

        # 深度研究后的成本
        research_costs = self.researcher.get_costs() - initial_costs

        # 如果有日志处理器则记录研究成本
        if self.researcher.log_handler:
            await self.researcher._log_event("research", step="deep_research_costs", details={
                "research_costs": research_costs,
                "total_costs": self.researcher.get_costs()
            })

        # 准备带引用的上下文
        context_with_citations = []
        for learning in results['learnings']:
            citation = results['citations'].get(learning, '')
            if citation:
                context_with_citations.append(f"{learning} [Source: {citation}]")
            else:
                context_with_citations.append(learning)

        # 将研究上下文加到经验教训让出的空间中，因为
        # 最终裁剪会保留最新的条目，会先丢弃经验教训
        if results.get('context'):
            context_budget = MAX_CONTEXT_WORDS - count_words(context_with_citations)
            if context_budget > 0:
                context_with_citations.extend(
                    trim_context_to_word_limit(results['context'], context_budget)
                )

        # 将最终上下文裁剪到词数限制
        final_context = trim_context_to_word_limit(context_with_citations)
        
        # 设置增强的上下文和已访问 URL
        self.researcher.context = "\n".join(
            item if isinstance(item, str)
            else item.get("Content", str(item)) if isinstance(item, dict)
            else str(item)
            for item in final_context
        )
        self.researcher.visited_urls = results['visited_urls']

        # 设置研究来源
        self.researcher.research_sources = results.get('sources') or []

        # 记录总执行时间
        end_time = time.time()
        execution_time = timedelta(seconds=end_time - start_time)
        logger.info(f"Total research execution time: {execution_time}")
        logger.info(f"Total research costs: ${research_costs:.2f}")

        # 返回上下文 —— 不在这里生成报告，因为将由主智能体完成
        return self.researcher.context
