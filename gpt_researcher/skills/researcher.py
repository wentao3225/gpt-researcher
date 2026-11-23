"""GPT Researcher 的研究协调技能。

此模块提供 ResearchConductor 类，管理和协调研究过程，
包括查询规划、网络搜索和上下文收集。
"""

import asyncio
import logging
import os
import random

from ..actions.agent_creator import choose_agent
from ..actions.query_processing import get_search_results, plan_research_outline
from ..actions.utils import stream_output
from ..document import DocumentLoader, LangChainDocumentLoader, OnlineDocumentLoader
from ..observability import emit_research_event
from ..utils.enum import ReportSource, ReportType
from ..utils.logging_config import get_json_handler


class ResearchConductor:
    """管理和协调研究过程。

    此类处理主要研究工作流，包括规划研究查询、
    执行网络搜索、管理 MCP 检索器，以及从各种来源
    收集上下文。

    属性：
        researcher: 上级 GPTResearcher 实例。
        logger: 研究事件的日志器。
        json_handler: JSON 日志处理器。
    """

    def __init__(self, researcher):
        """初始化 ResearchConductor。

        参数：
            researcher: 拥有此协调器的 GPTResearcher 实例。
        """
        self.researcher = researcher
        self.logger = logging.getLogger('research')
        self.json_handler = get_json_handler()
        # 为 MCP 结果添加缓存以避免冗余调用
        self._mcp_results_cache = None
        # 保护缓存填充，防止研究过程并发运行时竞争
        self._mcp_cache_lock = asyncio.Lock()
        # 跟踪平衡模式的 MCP 查询次数
        self._mcp_query_count = 0

    async def _emit_research_event(self, event: str, **fields) -> None:
        """通过可选的观察者发出事件，不要求 GPTResearcher。"""
        await emit_research_event(
            getattr(self.researcher, "research_observer", None),
            event,
            **fields,
        )

    async def _get_initial_search_results(self, query, query_domains=None):
        await stream_output(
            "logs",
            "planning_research",
            f"🌐 Browsing the web to learn more about the task: {query}...",
            self.researcher.websocket,
        )

        try:
            search_results = await get_search_results(
                query,
                self.researcher.retrievers[0],
                query_domains,
                researcher=self.researcher,
                max_results=self.researcher.cfg.max_search_results_per_query,
            )
        except Exception as exc:
            await self._emit_research_event(
                "search_failed",
                query=query,
                provider=self.researcher.retrievers[0].__name__,
                reason_code=type(exc).__name__,
            )
            raise
        self.logger.info(f"Initial search results obtained: {len(search_results)} results")
        return search_results

    async def plan_research(self, query, query_domains=None, *, search_results=None):
        """规划子查询，可选使用本轮已获取的结果。"""
        if search_results is None:
            search_results = await self._get_initial_search_results(query, query_domains)

        await stream_output(
            "logs",
            "planning_research",
            f"🤔 Planning the research strategy and subtasks...",
            self.researcher.websocket,
        )

        retriever_names = [r.__name__ for r in self.researcher.retrievers]
        # 移除重复日志 —— 改为在 conduct_research 中记录一次

        outline = await plan_research_outline(
            query=query,
            search_results=search_results,
            agent_role_prompt=self.researcher.role,
            cfg=self.researcher.cfg,
            parent_query=self.researcher.parent_query,
            report_type=self.researcher.report_type,
            cost_callback=self.researcher.add_costs,
            retriever_names=retriever_names,  # 传入检索器名称以优化 MCP
            **self.researcher.kwargs
        )
        for position, planned_query in enumerate(outline, start=1):
            await self._emit_research_event(
                "query_planned",
                query=planned_query,
                position=position,
            )
        self.logger.info(f"Research outline planned: {outline}")
        return outline

    async def conduct_research(self):
        """运行 GPT Researcher 来执行研究"""
        if self.json_handler:
            self.json_handler.update_content("query", self.researcher.query)
        
        self.logger.info(f"Starting research for query: {self.researcher.query}")
        
        # 在研究开始时记录一次活动的检索器
        retriever_names = [r.__name__ for r in self.researcher.retrievers]
        self.logger.info(f"Active retrievers: {retriever_names}")
        
        # 注意：这里刻意不清除 visited_urls。它可能与父研究员共享
        # （如详细报告把累积的 URL 传给每个子主题研究员），
        # 从而使已抓取的 URL 不会被再次抓取。
        research_data = []

        if self.researcher.verbose:
            await stream_output(
                "logs",
                "starting_research",
                f"🔍 Starting the research task for '{self.researcher.query}'...",
                self.researcher.websocket,
            )
            await stream_output(
                "logs",
                "agent_generated",
                self.researcher.agent,
                self.researcher.websocket
            )

        # 如果尚未定义则选择智能体和角色
        if not (self.researcher.agent and self.researcher.role):
            self.researcher.agent, self.researcher.role = await choose_agent(
                query=self.researcher.query,
                cfg=self.researcher.cfg,
                parent_query=self.researcher.parent_query,
                cost_callback=self.researcher.add_costs,
                headers=self.researcher.headers,
                prompt_family=self.researcher.prompt_family
            )
                
        # 检查是否配置了 MCP 检索器
        has_mcp_retriever = any("mcpretriever" in r.__name__.lower() for r in self.researcher.retrievers)
        if has_mcp_retriever:
            self.logger.info("MCP retrievers configured and will be used with standard research flow")

        # 根据来源类型执行研究
        if self.researcher.source_urls:
            self.logger.info("Using provided source URLs")
            research_data = await self._get_context_by_urls(self.researcher.source_urls)
            # `research_data and len(research_data) == 0` 永远不可能为真 ——
            # 真值永远不为空 —— 因此这个通知从未触发过。
            if not research_data and self.researcher.verbose:
                await stream_output(
                    "logs",
                    "answering_from_memory",
                    f"🧐 I was unable to find relevant context in the provided sources...",
                    self.researcher.websocket,
                )
            if self.researcher.complement_source_urls:
                self.logger.info("Complementing with web search")
                additional_research = await self._get_context_by_web_search(self.researcher.query, [], self.researcher.query_domains)
                research_data = " ".join(
                    part for part in (research_data, additional_research) if part
                )
        elif self.researcher.report_source == ReportSource.Web.value:
            self.logger.info("Using web search with all configured retrievers")
            research_data = await self._get_context_by_web_search(self.researcher.query, [], self.researcher.query_domains)
        elif self.researcher.report_source == ReportSource.Local.value:
            self.logger.info("Using local search")
            document_data = await DocumentLoader(self.researcher.cfg.doc_path).load()
            self.logger.info(f"Loaded {len(document_data)} documents")
            if self.researcher.vector_store:
                self.researcher.vector_store.load(document_data)

            research_data = await self._get_context_by_web_search(self.researcher.query, document_data, self.researcher.query_domains)
        # 混合搜索：同时包含本地文档和网络来源
        elif self.researcher.report_source == ReportSource.Hybrid.value:
            if self.researcher.document_urls:
                document_data = await OnlineDocumentLoader(self.researcher.document_urls).load()
            else:
                document_data = await DocumentLoader(self.researcher.cfg.doc_path).load()
            if self.researcher.vector_store:
                self.researcher.vector_store.load(document_data)
            # 本地文档分支与网络分支互不依赖，因此并发运行；
            # visited_urls 仍会在两者之间去重。
            docs_context, web_context = await asyncio.gather(
                self._get_context_by_web_search(self.researcher.query, document_data, self.researcher.query_domains),
                self._get_context_by_web_search(self.researcher.query, [], self.researcher.query_domains),
            )
            research_data = self.researcher.prompt_family.join_local_web_documents(docs_context, web_context)
        elif self.researcher.report_source == ReportSource.Azure.value:
            from ..document.azure_document_loader import AzureDocumentLoader
            azure_loader = AzureDocumentLoader(
                container_name=os.getenv("AZURE_CONTAINER_NAME"),
                connection_string=os.getenv("AZURE_CONNECTION_STRING")
            )
            azure_files = await azure_loader.load()
            document_data = await DocumentLoader(azure_files).load()  # 复用现有加载器
            research_data = await self._get_context_by_web_search(self.researcher.query, document_data)
            
        elif self.researcher.report_source == ReportSource.LangChainDocuments.value:
            langchain_documents_data = await LangChainDocumentLoader(
                self.researcher.documents
            ).load()
            if self.researcher.vector_store:
                self.researcher.vector_store.load(langchain_documents_data)
            research_data = await self._get_context_by_web_search(
                self.researcher.query, langchain_documents_data, self.researcher.query_domains
            )
        elif self.researcher.report_source == ReportSource.LangChainVectorStore.value:
            research_data = await self._get_context_by_vectorstore(self.researcher.query, self.researcher.vector_store_filter)

        # 对来源排序和策展
        self.researcher.context = research_data
        if self.researcher.cfg.curate_sources:
            self.logger.info("Curating sources")
            curated = await self.researcher.source_curator.curate_sources(research_data)
            # curate_sources() 返回带 Title/Content/Source 键的 List[dict]。
            # 规范化为 str，使期望 researcher.context 是字符串的下游代码
            # （如 "\n".join、.split()、len()）不会崩溃。
            if isinstance(curated, list):
                self.researcher.context = "\n\n".join(
                    "Title: {title}\nContent: {content}\nSource: {source}".format(
                        title=s.get("Title", ""),
                        content=s.get("Content", ""),
                        source=s.get("Source", ""),
                    ) if isinstance(s, dict) else str(s)
                    for s in curated
                )
            else:
                self.researcher.context = curated

        if self.researcher.verbose:
            await stream_output(
                "logs",
                "research_step_finalized",
                f"Finalized research step.\n💸 Total Research Costs: ${self.researcher.get_costs()}",
                self.researcher.websocket,
            )
            if self.json_handler:
                self.json_handler.update_content("costs", self.researcher.get_costs())
                self.json_handler.update_content("context", self.researcher.context)

        self.logger.info(f"Research completed. Context size: {len(str(self.researcher.context))}")
        return self.researcher.context

    async def _get_context_by_urls(self, urls):
        """抓取给定 URL 的上下文并压缩"""
        self.logger.info(f"Getting context from URLs: {urls}")
        
        new_search_urls = await self._get_new_urls(urls)
        self.logger.info(f"New URLs to process: {new_search_urls}")

        scraped_content = await self.researcher.scraper_manager.browse_urls(new_search_urls)
        self.logger.info(f"Scraped content from {len(scraped_content)} URLs")

        if self.researcher.vector_store:
            self.researcher.vector_store.load(scraped_content)

        context = await self.researcher.context_manager.get_similar_content_by_query(
            self.researcher.query, scraped_content
        )
        return context

    # 类似地为其他方法添加日志...

    async def _get_context_by_vectorstore(self, query, filter: dict | None = None):
        """
        通过搜索向量库为研究任务生成上下文
        返回：
            context: 上下文列表
        """
        self.logger.info(f"Starting vectorstore search for query: {query}")
        context = []
        # 生成包含原查询的子查询
        sub_queries = await self.plan_research(query)
        # 如果不是子研究员的一部分，将原始查询加入研究以获得更好结果
        if self.researcher.report_type != "subtopic_report":
            sub_queries.append(query)

        if self.researcher.verbose:
            await stream_output(
                "logs",
                "subqueries",
                f"🗂️  I will conduct my research based on the following queries: {sub_queries}...",
                self.researcher.websocket,
                True,
                sub_queries,
            )

        # 使用 asyncio.gather 异步处理子查询
        context = await asyncio.gather(
            *[
                self._process_sub_query_with_vectorstore(sub_query, filter)
                for sub_query in sub_queries
            ]
        )
        return context

    async def _get_context_by_web_search(self, query, scraped_data: list | None = None, query_domains: list | None = None):
        """
        通过搜索查询并抓取结果为研究任务生成上下文
        返回：
            context: 上下文列表
        """
        self.logger.info(f"Starting web search for query: {query}")
        
        if scraped_data is None:
            scraped_data = []
        if query_domains is None:
            query_domains = []

        # **可配置的 MCP 优化：控制 MCP 策略**
        mcp_retrievers = [r for r in self.researcher.retrievers if "mcpretriever" in r.__name__.lower()]
        
        # 获取 MCP 策略配置
        mcp_strategy = self._get_mcp_strategy()
        
        # 加锁使并发研究过程（如混合模式）只填充一次 MCP 缓存，
        # 而不是竞争运行两次相同的 MCP 研究。
        async with self._mcp_cache_lock:
            if mcp_retrievers and self._mcp_results_cache is None:
                if mcp_strategy == "disabled":
                    # MCP 已禁用 - 完全跳过 MCP 研究
                    self.logger.info("MCP disabled by strategy, skipping MCP research")
                    if self.researcher.verbose:
                        await stream_output(
                            "logs",
                            "mcp_disabled",
                            f"⚡ MCP research disabled by configuration",
                            self.researcher.websocket,
                        )
                elif mcp_strategy == "fast":
                    # 快速：用原始查询运行一次 MCP
                    self.logger.info("MCP fast strategy: Running once with original query")
                    if self.researcher.verbose:
                        await stream_output(
                            "logs",
                            "mcp_optimization",
                            f"🚀 MCP Fast: Running once for main query (performance mode)",
                            self.researcher.websocket,
                        )

                    # 用原始查询执行一次 MCP 研究
                    mcp_context = await self._execute_mcp_research_for_queries([query], mcp_retrievers)
                    self._mcp_results_cache = mcp_context
                    self.logger.info(f"MCP results cached: {len(mcp_context)} total context entries")
                elif mcp_strategy == "deep":
                    # 深度：将对所有查询运行 MCP（原始行为）- 延迟到按查询执行
                    self.logger.info("MCP deep strategy: Will run for all queries")
                    if self.researcher.verbose:
                        await stream_output(
                            "logs",
                            "mcp_comprehensive",
                            f"🔍 MCP Deep: Will run for each sub-query (thorough mode)",
                            self.researcher.websocket,
                        )
                    # 不缓存 - 让每个子查询单独运行 MCP
                else:
                    # 未知策略 - 默认为快速
                    self.logger.warning(f"Unknown MCP strategy '{mcp_strategy}', defaulting to fast")
                    mcp_context = await self._execute_mcp_research_for_queries([query], mcp_retrievers)
                    self._mcp_results_cache = mcp_context
                    self.logger.info(f"MCP results cached: {len(mcp_context)} total context entries")

        # 生成包含原查询的子查询
        initial_results = await self._get_initial_search_results(query, query_domains)
        sub_queries = await self.plan_research(query, query_domains, search_results=initial_results)
        self.logger.info(f"Generated sub-queries: {sub_queries}")
        
        # 如果不是子研究员的一部分，将原始查询加入研究以获得更好结果
        if self.researcher.report_type != "subtopic_report":
            sub_queries.append(query)

        if self.researcher.verbose:
            await stream_output(
                "logs",
                "subqueries",
                f"🗂️ I will conduct my research based on the following queries: {sub_queries}...",
                self.researcher.websocket,
                True,
                sub_queries,
            )

        # 使用 asyncio.gather 异步处理子查询
        try:
            # 在后续搜索可能丢失规划来源之前读取它们。
            # 保持传入文档路径和仅 MCP 路径不变。
            initial_context = ""
            if not scraped_data and initial_results:
                initial_context = await self._get_context_from_initial_results(
                    query, query_domains, initial_results
                )
            context = await asyncio.gather(
                *[
                    self._process_sub_query(sub_query, scraped_data, query_domains)
                    for sub_query in sub_queries
                ]
            )
            self.logger.info(f"Gathered context from {len(context)} sub-queries")
            # 过滤空结果并拼接上下文
            context = [c for c in [initial_context, *context] if c]
            if context:
                combined_context = " ".join(context)
                self.logger.info(f"Combined context size: {len(combined_context)}")
                return combined_context
            return []
        except Exception as e:
            self.logger.error(f"Error during web search: {e}", exc_info=True)
            return []

    def _get_mcp_strategy(self) -> str:
        """
        获取 MCP 策略配置。
        
        优先级：
        1. 实例级设置（self.researcher.mcp_strategy）
        2. 配置文件设置（self.researcher.cfg.mcp_strategy）
        3. 默认值（"fast"）
        
        返回：
            str: MCP 策略
                "disabled" = 完全跳过 MCP
                "fast" = 用原始查询运行一次 MCP（默认）
                "deep" = 对所有子查询运行 MCP
        """
        # 先检查实例级设置
        if hasattr(self.researcher, 'mcp_strategy') and self.researcher.mcp_strategy is not None:
            return self.researcher.mcp_strategy
        
        # 检查配置设置
        if hasattr(self.researcher.cfg, 'mcp_strategy'):
            return self.researcher.cfg.mcp_strategy
        
        # 默认为快速模式
        return "fast"

    async def _execute_mcp_research_for_queries(self, queries: list, mcp_retrievers: list) -> list:
        """
        对一组查询执行 MCP 研究。
        
        参数：
            queries: 要研究的查询列表
            mcp_retrievers: MCP 检索器类列表
            
        返回：
            list: 所有查询合并的 MCP 上下文条目
        """
        all_mcp_context = []
        
        for i, query in enumerate(queries, 1):
            self.logger.info(f"Executing MCP research for query {i}/{len(queries)}: {query}")
            
            for retriever in mcp_retrievers:
                try:
                    mcp_results = await self._execute_mcp_research(retriever, query)
                    if mcp_results:
                        for result in mcp_results:
                            content = result.get("body", "")
                            url = result.get("href", "")
                            title = result.get("title", "")
                            
                            if content:
                                context_entry = {
                                    "content": content,
                                    "url": url,
                                    "title": title,
                                    "query": query,
                                    "source_type": "mcp"
                                }
                                all_mcp_context.append(context_entry)
                        
                        self.logger.info(f"Added {len(mcp_results)} MCP results for query: {query}")
                        
                        if self.researcher.verbose:
                            await stream_output(
                                "logs",
                                "mcp_results_cached",
                                f"✅ Cached {len(mcp_results)} MCP results from query {i}/{len(queries)}",
                                self.researcher.websocket,
                            )
                except Exception as e:
                    self.logger.error(f"Error in MCP research for query '{query}': {e}")
                    if self.researcher.verbose:
                        await stream_output(
                            "logs",
                            "mcp_cache_error",
                            f"⚠️ MCP research error for query {i}, continuing with other sources",
                            self.researcher.websocket,
                        )
        
        return all_mcp_context

    def _tavily_mcp_redundant_with_direct(self, mcp_retrievers, non_mcp_retrievers) -> bool:
        """当 MCP 只会重新查询 Tavily 而直接 Tavily 已启用时为 True。

        前端的 Tavily Web Search MCP 预设命中与 `TavilySearch` 相同的 API，
        两者同时运行时只为没有新数据的额外 LLM 工具选择成本买单（#1875）。
        """
        if not mcp_retrievers or not non_mcp_retrievers:
            return False
        has_direct_tavily = any(
            getattr(r, "__name__", "").lower() == "tavilysearch" for r in non_mcp_retrievers
        )
        if not has_direct_tavily:
            return False
        configs = getattr(self.researcher, "mcp_configs", None) or []
        if not configs:
            return False
        # 如果每个已配置的 MCP 服务器都是 Tavily MCP 包，则视为冗余。
        def _is_tavily_mcp(cfg: dict) -> bool:
            name = str(cfg.get("name", "")).lower()
            args = " ".join(str(a) for a in (cfg.get("args") or [])).lower()
            command = str(cfg.get("command", "")).lower()
            blob = f"{name} {args} {command}"
            return "tavily" in blob

        return all(isinstance(c, dict) and _is_tavily_mcp(c) for c in configs)


    async def _process_sub_query(self, sub_query: str, scraped_data: list = [], query_domains: list = []):
        """接收子查询，据此抓取 URL 并收集上下文。"""
        if self.json_handler:
            self.json_handler.log_event("sub_query", {
                "query": sub_query,
                "scraped_data_size": len(scraped_data)
            })
        
        if self.researcher.verbose:
            await stream_output(
                "logs",
                "running_subquery_research",
                f"\n🔍 Running research for '{sub_query}'...",
                self.researcher.websocket,
            )

        try:
            # 识别 MCP 检索器
            mcp_retrievers = [r for r in self.researcher.retrievers if "mcpretriever" in r.__name__.lower()]
            non_mcp_retrievers = [r for r in self.researcher.retrievers if "mcpretriever" not in r.__name__.lower()]

            # 在默认 RETRIEVER=tavily 下避免双重 Tavily 路径（直接检索器 + tavily-mcp）。
            if self._tavily_mcp_redundant_with_direct(mcp_retrievers, non_mcp_retrievers):
                self.logger.warning(
                    "Skipping LLM MCP Tavily path because TavilySearch is already configured as a direct retriever; set RETRIEVER without tavily or use non-Tavily MCP servers to keep MCP."
                )
                if self.researcher.verbose:
                    await stream_output(
                        "logs",
                        "mcp_tavily_deduped",
                        "⚠️ Skipping Tavily MCP (redundant with direct Tavily retriever) to avoid double API cost",
                        self.researcher.websocket,
                    )
                mcp_retrievers = []
            
            # 初始化上下文组件
            mcp_context = []
            web_context = ""
            
            # 获取 MCP 策略配置
            mcp_strategy = self._get_mcp_strategy()
            
            # **可配置的 MCP 处理**
            if mcp_retrievers:
                if mcp_strategy == "disabled":
                    # MCP 已禁用 - 完全跳过
                    self.logger.info(f"MCP disabled for sub-query: {sub_query}")
                elif mcp_strategy == "fast" and self._mcp_results_cache is not None:
                    # 快速：使用缓存结果
                    mcp_context = self._mcp_results_cache.copy()
                    
                    if self.researcher.verbose:
                        await stream_output(
                            "logs",
                            "mcp_cache_reuse",
                            f"♻️ Reusing cached MCP results ({len(mcp_context)} sources) for: {sub_query}",
                            self.researcher.websocket,
                        )
                    
                    self.logger.info(f"Reused {len(mcp_context)} cached MCP results for sub-query: {sub_query}")
                elif mcp_strategy == "deep":
                    # 深度：为每个子查询运行 MCP
                    self.logger.info(f"Running deep MCP research for: {sub_query}")
                    if self.researcher.verbose:
                        await stream_output(
                            "logs",
                            "mcp_comprehensive_run",
                            f"🔍 Running deep MCP research for: {sub_query}",
                            self.researcher.websocket,
                        )
                    
                    mcp_context = await self._execute_mcp_research_for_queries([sub_query], mcp_retrievers)
                else:
                    # 回退：无缓存且非深度模式时，为该查询运行 MCP
                    self.logger.warning("MCP cache not available, falling back to per-sub-query execution")
                    if self.researcher.verbose:
                        await stream_output(
                            "logs",
                            "mcp_fallback",
                            f"🔌 MCP cache unavailable, running MCP research for: {sub_query}",
                            self.researcher.websocket,
                        )
                    
                    mcp_context = await self._execute_mcp_research_for_queries([sub_query], mcp_retrievers)
            
            # 使用非 MCP 检索器获取网络搜索上下文（未提供抓取数据时）
            if not scraped_data:
                scraped_data = await self._scrape_data_by_urls(sub_query, query_domains)
                self.logger.info(f"Scraped data size: {len(scraped_data)}")

            # 基于抓取数据获取相似内容
            if scraped_data:
                web_context = await self.researcher.context_manager.get_similar_content_by_query(sub_query, scraped_data)
                self.logger.info(f"Web content found for sub-query: {len(str(web_context)) if web_context else 0} chars")

            # 智能结合 MCP 上下文与网络上下文
            combined_context = self._combine_mcp_and_web_context(mcp_context, web_context, sub_query)
            
            # 记录上下文结合结果
            if combined_context:
                context_length = len(str(combined_context))
                self.logger.info(f"Combined context for '{sub_query}': {context_length} chars")
                
                if self.researcher.verbose:
                    mcp_count = len(mcp_context)
                    web_available = bool(web_context)
                    cache_used = self._mcp_results_cache is not None and mcp_retrievers and mcp_strategy != "deep"
                    cache_status = " (cached)" if cache_used else ""
                    await stream_output(
                        "logs",
                        "context_combined",
                        f"📚 Combined research context: {mcp_count} MCP sources{cache_status}, {'web content' if web_available else 'no web content'}",
                        self.researcher.websocket,
                    )
            else:
                self.logger.warning(f"No combined context found for sub-query: {sub_query}")
                if self.researcher.verbose:
                    await stream_output(
                        "logs",
                        "subquery_context_not_found",
                        f"🤷 No content found for '{sub_query}'...",
                        self.researcher.websocket,
                    )
            
            if combined_context and self.json_handler:
                self.json_handler.log_event("content_found", {
                    "sub_query": sub_query,
                    "content_size": len(str(combined_context)),
                    "mcp_sources": len(mcp_context),
                    "web_content": bool(web_context)
                })
                
            return combined_context
            
        except Exception as e:
            self.logger.error(f"Error processing sub-query {sub_query}: {e}", exc_info=True)
            if self.researcher.verbose:
                await stream_output(
                    "logs",
                    "subquery_error",
                    f"❌ Error processing '{sub_query}': {str(e)}",
                    self.researcher.websocket,
                )
            return ""

    async def _execute_mcp_research(self, retriever, query):
        """
        使用新的两阶段方法执行 MCP 研究。
        
        参数：
            retriever: MCP 检索器类
            query: 搜索查询
            
        返回：
            list: MCP 研究结果
        """
        retriever_name = retriever.__name__
        
        self.logger.info(f"Executing MCP research with {retriever_name} for query: {query}")
        
        try:
            # 用适当参数实例化 MCP 检索器
            # 传入研究员实例（self.researcher），其中包含 cfg 和 mcp_configs
            retriever_instance = retriever(
                query=query, 
                headers=self.researcher.headers,
                query_domains=self.researcher.query_domains,
                websocket=self.researcher.websocket,
                researcher=self.researcher  # 传入整个研究员实例
            )
            
            if self.researcher.verbose:
                await stream_output(
                    "logs",
                    "mcp_retrieval_stage1",
                    f"🧠 Stage 1: Selecting optimal MCP tools for: {query}",
                    self.researcher.websocket,
                )
            
            # MCPRetriever.search() 是同步接口：它启动自己的线程并等待
            # （future.result(timeout=300)），因此整个 MCP 轮次中它都会阻塞
            # 调用方 —— LLM 工具选择加上每个远程工具调用。像其他检索器调用点
            # 一样在工作线程上运行它；如果从这里直接调用，它会冻结事件
            # 循环整轮：websocket 日志流无法刷新，且
            # _get_context_by_web_search 汇聚的兄弟子查询无法推进。
            results = await asyncio.to_thread(
                retriever_instance.search,
                max_results=self.researcher.cfg.max_search_results_per_query,
            )
            
            if results:
                result_count = len(results)
                self.logger.info(f"MCP research completed: {result_count} results from {retriever_name}")
                
                if self.researcher.verbose:
                    await stream_output(
                        "logs",
                        "mcp_research_complete",
                        f"🎯 MCP research completed: {result_count} intelligent results obtained",
                        self.researcher.websocket,
                    )
                
                return results
            else:
                self.logger.info(f"No results returned from MCP research with {retriever_name}")
                if self.researcher.verbose:
                    await stream_output(
                        "logs",
                        "mcp_no_results",
                        f"ℹ️ No relevant information found via MCP for: {query}",
                        self.researcher.websocket,
                    )
                return []
                
        except Exception as e:
            self.logger.error(f"Error in MCP research with {retriever_name}: {str(e)}")
            if self.researcher.verbose:
                await stream_output(
                    "logs",
                    "mcp_research_error",
                    f"⚠️ MCP research error: {str(e)} - continuing with other sources",
                    self.researcher.websocket,
                )
            return []

    def _combine_mcp_and_web_context(self, mcp_context: list, web_context: str, sub_query: str) -> str:
        """
        智能结合 MCP 和网络研究上下文。
        
        参数：
            mcp_context: MCP 上下文条目列表
            web_context: 网络研究上下文字符串
            sub_query: 正在处理的子查询
            
        返回：
            str: 结合后的上下文字符串
        """
        combined_parts = []
        
        # 如果网络上下文可用则先添加
        if web_context and web_context.strip():
            combined_parts.append(web_context.strip())
            self.logger.debug(f"Added web context: {len(web_context)} chars")
        
        # 以适当格式添加 MCP 上下文
        if mcp_context:
            mcp_formatted = []
            
            for i, item in enumerate(mcp_context):
                content = item.get("content", "")
                url = item.get("url", "")
                title = item.get("title", f"MCP Result {i+1}")
                
                if content and content.strip():
                    # 创建格式良好的上下文条目
                    if url and url != f"mcp://llm_analysis":
                        citation = f"\n\n*Source: {title} ({url})*"
                    else:
                        citation = f"\n\n*Source: {title}*"
                    
                    formatted_content = f"{content.strip()}{citation}"
                    mcp_formatted.append(formatted_content)
            
            if mcp_formatted:
                # 以清晰分隔拼接 MCP 结果
                mcp_section = "\n\n---\n\n".join(mcp_formatted)
                combined_parts.append(mcp_section)
                self.logger.debug(f"Added {len(mcp_context)} MCP context entries")
        
        # 拼接所有部分
        if combined_parts:
            final_context = "\n\n".join(combined_parts)
            self.logger.info(f"Combined context for '{sub_query}': {len(final_context)} total chars")
            return final_context
        else:
            self.logger.warning(f"No context to combine for sub-query: {sub_query}")
            return ""

    async def _process_sub_query_with_vectorstore(self, sub_query: str, filter: dict | None = None):
        """接收子查询，并从用户提供的向量库收集上下文

        参数：
            sub_query (str): 从原始查询生成的子查询

        返回：
            str: 从搜索中收集的上下文
        """
        if self.researcher.verbose:
            await stream_output(
                "logs",
                "running_subquery_with_vectorstore_research",
                f"\n🔍 Running research for '{sub_query}'...",
                self.researcher.websocket,
            )

        context = await self.researcher.context_manager.get_similar_content_by_query_with_vectorstore(sub_query, filter)

        return context

    async def _get_new_urls(self, url_set_input):
        """从给定 URL 集合获取新的 URL。
        参数：url_set_input (set[str]): 要获取新 URL 的 URL 集合
        返回：list[str]: 从给定 URL 集合获取的新 URL
        """

        new_urls = []
        for url in url_set_input:
            if url not in self.researcher.visited_urls:
                self.researcher.visited_urls.add(url)
                new_urls.append(url)
                if self.researcher.verbose:
                    await stream_output(
                        "logs",
                        "added_source_url",
                        f"✅ Added source url to research: {url}\n",
                        self.researcher.websocket,
                        True,
                        url,
                    )

        return new_urls

    async def _get_context_from_initial_results(self, query, query_domains, initial_results):
        """通过正常检索获取规划来源；绝不把摘要当作页面。"""
        if "mcpretriever" in self.researcher.retrievers[0].__name__.lower():
            return ""
        try:
            data = await self._scrape_data_by_urls(
                query, query_domains, initial_results=initial_results
            )
            if data:
                return await self.researcher.context_manager.get_similar_content_by_query(query, data)
        except Exception as e:
            # 规划来源不可用不应妨碍后续研究。
            self.logger.warning(f"Error reading initial search results: {e}")
        return ""

    async def _search_relevant_source_urls(self, query, query_domains: list | None = None, *, initial_results=None):
        new_search_urls = []
        prefetched_content = []
        if query_domains is None:
            query_domains = []

        # 遍历当前设置的检索器
        # 这使方法在检索器被临时修改时也能工作
        # 初始结果只来自第一个检索器。复用其内容契约，
        # 而不发出另一次搜索或把结果归给其他检索器。
        retrievers = self.researcher.retrievers if initial_results is None else self.researcher.retrievers[:1]
        for retriever_class in retrievers:
            # 跳过 MCP 检索器，因为它们不提供供抓取的 URL
            if "mcpretriever" in retriever_class.__name__.lower():
                continue

            try:
                # 用子查询实例化检索器
                retriever = retriever_class(query, query_domains=query_domains)

                # 使用当前检索器执行搜索
                if initial_results is None:
                    search_results = await asyncio.to_thread(
                        retriever.search, max_results=self.researcher.cfg.max_search_results_per_query
                    )
                else:
                    search_results = initial_results

                if not search_results:
                    await self._emit_research_event(
                        "search_empty",
                        query=query,
                        provider=retriever_class.__name__,
                    )
                    continue

                # 该检索器返回的是待抓取的 URL，还是它已取回的
                # 内容？优先使用显式声明（BaseRetriever.requires_scraping）；
                # 检索器未声明时回退到旧的基于长度的启发式，使
                # 第三方和用户自定义检索器不受影响。
                requires_scraping = getattr(retriever, "requires_scraping", None)

                # 将已有内容的结果与需要抓取的结果分开
                for rank, result in enumerate(search_results, start=1):
                    url = result.get("href") or result.get("url")
                    raw_content = result.get("raw_content")

                    if not url:
                        continue

                    await self._emit_research_event(
                        "candidate_discovered",
                        query=query,
                        provider=retriever_class.__name__,
                        url=url,
                        rank=rank,
                    )

                    if requires_scraping is True:
                        # 已声明：URL 之外的任何内容都是预览，无论多长，
                        # 因此仍会抓取页面。这就是阻止长摘要被误认为
                        # 文章正文而丢失引用的机制。
                        new_search_urls.append(url)
                    elif requires_scraping is False:
                        # 已声明：检索器自己抓取了内容。
                        if raw_content:
                            prefetched_content.append({
                                "url": url,
                                "raw_content": raw_content,
                            })
                            self.researcher.add_research_sources([{"url": url}])
                            # #2100: 预取 URL 必须像 _get_new_urls 中抓取的
                            # URL 一样加入 visited_urls，否则重复项会
                            # 重新合并，引用消失。
                            self.researcher.visited_urls.add(url)
                            await self._emit_research_event(
                                "content_prefetched",
                                query=query,
                                provider=retriever_class.__name__,
                                url=url,
                                rank=rank,
                                content_length=len(raw_content),
                            )
                        else:
                            new_search_urls.append(url)
                    elif raw_content and len(raw_content) > 100:
                        # 未声明：旧行为，保持不变。
                        prefetched_content.append({
                            "url": url,
                            "raw_content": raw_content,
                        })
                        self.researcher.add_research_sources([{"url": url}])
                        # #2100: 与上面相同的 visited_urls 记账。
                        self.researcher.visited_urls.add(url)
                        await self._emit_research_event(
                            "content_prefetched",
                            query=query,
                            provider=retriever_class.__name__,
                            url=url,
                            rank=rank,
                            content_length=len(raw_content),
                        )
                    else:
                        new_search_urls.append(url)
            except Exception as e:
                self.logger.error(f"Error searching with {retriever_class.__name__}: {e}")
                await self._emit_research_event(
                    "search_failed",
                    query=query,
                    provider=retriever_class.__name__,
                    reason_code=type(e).__name__,
                )

        # 获取唯一的 URL
        requested_urls = list(dict.fromkeys(new_search_urls))
        new_search_urls = await self._get_new_urls(new_search_urls)
        selected = set(new_search_urls)
        for url in requested_urls:
            if url in selected:
                await self._emit_research_event(
                    "selected_for_fetch",
                    query=query,
                    url=url,
                )
            else:
                await self._emit_research_event(
                    "candidate_deduplicated",
                    query=query,
                    url=url,
                    reason_code="ALREADY_VISITED",
                )
        random.shuffle(new_search_urls)

        return new_search_urls, prefetched_content

    async def _scrape_data_by_urls(self, sub_query, query_domains: list | None = None, *, initial_results=None):
        """
        对多个检索器运行子查询并抓取结果 URL。
        已提供全文的检索器（如 PubMed Central）的内容直接
        通过，不再重新抓取。

        参数：
            sub_query (str): 要搜索的子查询。

        返回：
            list: 抓取内容结果列表。
        """
        if query_domains is None:
            query_domains = []

        new_search_urls, prefetched_content = await self._search_relevant_source_urls(
            sub_query, query_domains, initial_results=initial_results
        )

        # 如果启用了详细模式则记录研究过程
        if self.researcher.verbose:
            await stream_output(
                "logs",
                "researching",
                f"🤔 Researching for relevant information across multiple sources...\n",
                self.researcher.websocket,
            )

        # 抓取需要获取的 URL（跳过检索器已提供的）
        scraped_content = await self.researcher.scraper_manager.browse_urls(new_search_urls)

        # 合并不重新抓取就已提供全文的检索器的预取内容
        scraped_content.extend(prefetched_content)

        if self.researcher.vector_store:
            self.researcher.vector_store.load(scraped_content)

        return scraped_content

    async def _search(self, retriever, query):
        """
        使用指定的检索器执行搜索。
        
        参数：
            retriever: 要使用的检索器类
            query: 搜索查询
            
        返回：
            list: 搜索结果
        """
        retriever_name = retriever.__name__
        is_mcp_retriever = "mcpretriever" in retriever_name.lower()
        
        self.logger.info(f"Searching with {retriever_name} for query: {query}")
        
        try:
            # 实例化检索器
            retriever_instance = retriever(
                query=query, 
                headers=self.researcher.headers,
                query_domains=self.researcher.query_domains,
                websocket=self.researcher.websocket if is_mcp_retriever else None,
                researcher=self.researcher if is_mcp_retriever else None
            )
            
            # 如果使用 MCP 检索器则记录 MCP 服务器配置
            if is_mcp_retriever and self.researcher.verbose:
                await stream_output(
                    "logs",
                    "mcp_retrieval",
                    f"🔌 Consulting MCP server(s) for information on: {query}",
                    self.researcher.websocket,
                )
            
            # 执行搜索
            if hasattr(retriever_instance, 'search'):
                results = retriever_instance.search(
                    max_results=self.researcher.cfg.max_search_results_per_query
                )
                
                # 记录结果信息
                if results:
                    result_count = len(results)
                    self.logger.info(f"Received {result_count} results from {retriever_name}")
                    
                    # MCP 检索器的特殊日志
                    if is_mcp_retriever:
                        if self.researcher.verbose:
                            await stream_output(
                                "logs",
                                "mcp_results",
                                f"✓ Retrieved {result_count} results from MCP server",
                                self.researcher.websocket,
                            )
                        
                        # 记录结果详情
                        for i, result in enumerate(results[:3]):  # 记录前 3 个结果
                            title = result.get("title", "No title")
                            url = result.get("href", "No URL")
                            content_length = len(result.get("body", "")) if result.get("body") else 0
                            self.logger.info(f"MCP result {i+1}: '{title}' from {url} ({content_length} chars)")
                            
                        if result_count > 3:
                            self.logger.info(f"... and {result_count - 3} more MCP results")
                else:
                    self.logger.info(f"No results returned from {retriever_name}")
                    if is_mcp_retriever and self.researcher.verbose:
                        await stream_output(
                            "logs",
                            "mcp_no_results",
                            f"ℹ️ No relevant information found from MCP server for: {query}",
                            self.researcher.websocket,
                        )
                
                return results
            else:
                self.logger.error(f"Retriever {retriever_name} does not have a search method")
                return []
        except Exception as e:
            self.logger.error(f"Error searching with {retriever_name}: {str(e)}")
            if is_mcp_retriever and self.researcher.verbose:
                await stream_output(
                    "logs",
                    "mcp_error",
                    f"❌ Error retrieving information from MCP server: {str(e)}",
                    self.researcher.websocket,
                )
            return []
            
    async def _extract_content(self, results):
        """
        使用浏览器管理器从搜索结果中提取内容。
        
        参数：
            results: 搜索结果
            
        返回：
            list: 提取的内容
        """
        self.logger.info(f"Extracting content from {len(results)} search results")
        
        # 从搜索结果获取 URL
        urls = []
        for result in results:
            if isinstance(result, dict) and "href" in result:
                urls.append(result["href"])
        
        # 未找到 URL 则跳过
        if not urls:
            return []
            
        # 确保不访问已访问过的 URL
        new_urls = [url for url in urls if url not in self.researcher.visited_urls]
        
        # 没有新 URL 则返回空
        if not new_urls:
            return []
            
        # 抓取 URL 的内容
        scraped_content = await self.researcher.scraper_manager.browse_urls(new_urls)
        
        # 将 URL 加入 visited_urls
        self.researcher.visited_urls.update(new_urls)
        
        return scraped_content
        
    async def _summarize_content(self, query, content):
        """
        摘要提取的内容。
        
        参数：
            query: 搜索查询
            content: 提取的内容
            
        返回：
            str: 摘要后的内容
        """
        self.logger.info(f"Summarizing content for query: {query}")
        
        # 无内容则跳过
        if not content:
            return ""
            
        # 使用上下文管理器摘要内容
        summary = await self.researcher.context_manager.get_similar_content_by_query(
            query, content
        )
        
        return summary
        
    async def _update_search_progress(self, current, total):
        """
        更新搜索进度。
        
        参数：
            current: 已处理的子查询数
            total: 子查询总数
        """
        if self.researcher.verbose and self.researcher.websocket:
            progress = int((current / total) * 100)
            await stream_output(
                "logs",
                "research_progress",
                f"📊 Research Progress: {progress}%",
                self.researcher.websocket,
                True,
                {
                    "current": current,
                    "total": total,
                    "progress": progress
                }
            )
