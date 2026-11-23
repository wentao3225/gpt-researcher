"""
质量评估指标
------------
所有报告质量指标都集中在本模块中，按衡量对象分组：

来源质量 —— 输入是否优质？
  source_diversity()      —— 通过比率 + 香农熵衡量域名多样性        （$0）
  source_authority()      —— 权威性评分；未知域名在提供 LLM 时使用   （$0 / LLM）

写作行为 —— 报告是如何撰写的？
  citation_faithfulness() —— 正文中引用的来源是否与 References 列表一致（$0）
  subtopic_coverage()     —— 报告对预期子主题的覆盖程度           （LLM）

幻觉 —— 内容是否有依据？
  unsupported_claim()     —— 断言级可信度（supported/inferred/unsupported）（LLM）
"""

import asyncio
import json
import math
import re
from collections import Counter
from urllib.parse import urlparse


# ---------------------------------------------------------------------------
# 通用辅助函数
# ---------------------------------------------------------------------------

def _extract_domains(urls: list[str]) -> set[str]:
    """从 URL 列表返回归一化后的域名集合（去掉 www.）。"""
    return {urlparse(u).netloc.lower().removeprefix("www.") for u in urls if u}


def _extract_domains_from_text(text: str) -> set[str]:
    """扫描原始文本中的 URL 并提取其域名。"""
    urls = re.findall(r'https?://[^\s\)\]\,\"\'<>]+', text)
    return _extract_domains(urls)


def _parse_json(text: str):
    """从 LLM 响应中解析 JSON，容忍 Markdown 代码围栏。"""
    text = text.strip()
    match = re.search(r'```(?:json)?\s*([\s\S]+?)\s*```', text)
    if match:
        text = match.group(1)
    return json.loads(text)


def _split_at_references(report: str) -> tuple[str, str]:
    """把报告切分为（正文, 参考文献部分）。

    单智能体和多智能体报告都以 `## References`（或 Sources/Citations/Bibliography）
    章节结尾，列出所有来源。该列表并不能证明正文是依据这些来源撰写的，因此
    引用类指标只应考察正文。返回 (body, refs)；若不存在则 refs 为 ""。
    """
    m = re.search(r'(?im)^#+\s*(references|sources|citations|bibliography)\s*$', report)
    if m:
        return report[:m.start()], report[m.start():]
    return report, ""


# ---------------------------------------------------------------------------
# 1. 引用忠实度（写作行为）
# ---------------------------------------------------------------------------

def citation_faithfulness(report: str, source_urls: list[str], context=None) -> dict:
    """
    报告“声称”使用的来源与它正文中实际引用的来源之间的一致程度。

        citation_faithfulness = |正文中引用 ∩ 参考文献中列出| / |参考文献中列出|

      1.0  = 末尾 `## References` 列表中的每个来源都确实在正文某处被引用
      低   = 许多来源被列在参考文献里却从未在正文中引用
             （装饰性 / 未使用的引用 —— 属于自洽性失败）
      None = 报告没有可供比较的 References 章节

    本指标衡量 LLM 的引用纪律（是否用了它声称使用的内容），而非内容是否正确
    —— 后者由 unsupported_claim 负责。

    另外返回 `citation_coverage` 作为**描述性**统计量（不是质量分数）：
    研究中实际用到的来源里，有多少在正文中被引用。覆盖率低可能只是说明智能体
    合理地丢弃了一些来源，因此不应单独当作质量信号解读。

    Args:
        report:      完整的 Markdown 报告文本。
        source_urls: researcher.get_source_urls() 的 URL 列表。
        context:     researcher.conduct_research() 的原始上下文（str 或 list）；
                     提供时，覆盖率的分母 = 进入上下文（经相关性过滤）的来源。
    """
    source_domains = _extract_domains(source_urls)

    body, refs     = _split_at_references(report)
    cited_domains  = _extract_domains_from_text(body)   # 仅正文中的引用
    listed_domains = _extract_domains_from_text(refs)   # 仅报告末尾的列表

    # --- 忠实度：列在参考文献里的内容有多少在正文中被引用 ---
    if listed_domains:
        aligned      = cited_domains & listed_domains
        faithfulness = round(len(aligned) / len(listed_domains), 3)
        listed_only  = sorted(listed_domains - cited_domains)
    else:
        faithfulness = None        # 没有参考文献章节 → 不适用
        listed_only  = []

    # --- 覆盖率（仅描述性） ---
    if context:
        context_text = context if isinstance(context, str) else json.dumps(context)
        used_domains = _extract_domains_from_text(context_text) & source_domains
    else:
        used_domains = source_domains
    covered  = cited_domains & used_domains
    coverage = round(len(covered) / len(used_domains), 3) if used_domains else None

    return {
        "citation_faithfulness": faithfulness,
        "listed_ref_domains":    len(listed_domains),
        "cited_body_domains":    len(cited_domains),
        "listed_only_domains":   listed_only,    # 被列为参考文献但正文从未引用
        "citation_coverage":     coverage,       # 描述性指标，不是质量分数
        "uncited_used_domains":  sorted(used_domains - cited_domains),
    }


# ---------------------------------------------------------------------------
# 2. 来源多样性
# ---------------------------------------------------------------------------

def source_diversity(source_urls: list[str]) -> dict:
    """
    用比率和香农熵衡量域名多样性。

    diversity_ratio = 唯一域名数 / 总来源数  (0–1)
    domain_entropy  = 域名分布上的香农熵
                      （越高 = 分布越均匀，没有单一域名占据主导）
    """
    if not source_urls:
        return {
            "total_sources":   0,
            "unique_domains":  0,
            "diversity_ratio": 0.0,
            "domain_entropy":  0.0,
            "top_domains":     [],
        }

    domains = [urlparse(u).netloc.lower().removeprefix("www.") for u in source_urls if u]
    counts  = Counter(domains)
    total   = len(domains)
    unique  = len(counts)

    entropy = -sum((c / total) * math.log2(c / total) for c in counts.values())

    # diversity_ratio = 唯一域名数 / 总来源数
    # 香农熵 = -sum((c/total) * log2(c/total) for c in counts.values())
    return {
        "total_sources":   total,
        "unique_domains":  unique,
        "diversity_ratio": round(unique / total, 3),
        "domain_entropy":  round(entropy, 3),
        "top_domains":     counts.most_common(3),
    }


# ---------------------------------------------------------------------------
# 3. 来源权威性
# ---------------------------------------------------------------------------

# 高权威性域名（学术、政府、国际组织等）
_HIGH_AUTHORITY_DOMAINS = {
    "nature.com", "science.org", "cell.com", "thelancet.com",
    "nejm.org", "pubmed.ncbi.nlm.nih.gov", "ncbi.nlm.nih.gov",
    "arxiv.org", "scholar.google.com", "jstor.org", "ssrn.com",
    "plos.org", "biorxiv.org", "medrxiv.org",
    "who.int", "un.org", "imf.org", "worldbank.org",
    "oecd.org", "wto.org", "europa.eu",
    "britannica.com",
}

# 主流新闻媒体域名
_MAJOR_NEWS_DOMAINS = {
    "bbc.com", "bbc.co.uk", "reuters.com", "apnews.com",
    "nytimes.com", "theguardian.com", "washingtonpost.com",
    "wsj.com", "economist.com", "ft.com", "bloomberg.com",
    "npr.org", "pbs.org", "aljazeera.com",
}

# 供 LLM 评定域名权威性的提示词，属于提示词内容，保留英文
_LLM_AUTHORITY_PROMPT = """
You are a research source evaluator.

Rate the authority of each domain below as a research source on a scale of 0.0 to 1.0.
Consider whether it is a known academic journal, government body, established news outlet,
professional organization, or specialized expert source.

Domains:
{domains}

Return ONLY a JSON array in this exact format, one entry per domain:
[{{"domain": "example.com", "score": 0.7, "reason": "established tech news outlet"}}]
""".strip()


def _score_domain(domain: str):
    """只做高置信度的规则匹配。

    对可以靠规则可靠评分的域名返回 (score, tier)
    （.gov、.edu、已知学术/新闻白名单、wikipedia），否则返回 None
    以表示该域名应推迟交给 LLM 判断。

    注意：这里故意不对 .com / .org 评分。许多高权威性来源都在
    .com/.org 上（例如 sciencedirect.com、bmj.com、cureus.com），
    一刀切的 “.com → 0.5” 规则会把它们误判为低权威，更关键的是
    使它们永远无法进入 LLM 评分环节。这里把所有此类域名都推迟给
    LLM 处理（批量调用，因此成本约为每次查询 1 次调用）。
    """
    d = domain.lower().removeprefix("www.")
    if d.endswith(".gov"):           return 1.0, "government"
    if d.endswith(".edu"):           return 0.9, "academic-edu"
    if d in _HIGH_AUTHORITY_DOMAINS: return 0.9, "high-authority"
    if d in _MAJOR_NEWS_DOMAINS:     return 0.8, "major-news"
    if "wikipedia.org" in d:         return 0.75, "wikipedia"
    return None   # 把 .com / .org / 其他一切推迟给 LLM


def _fallback_score(domain: str) -> tuple[float, str]:
    """没有 grader_model 时，对推迟域名使用纯规则评分。"""
    d = domain.lower().removeprefix("www.")
    if d.endswith(".org"): return 0.65, "nonprofit-org"
    if d.endswith(".com"): return 0.5,  "commercial"
    return 0.4, "unknown"


async def _llm_score_domains(domains: list[str], grader_model) -> list[dict]:
    """用一次 LLM 调用批量评定域名权威性，失败时降级为 unknown。"""
    prompt = _LLM_AUTHORITY_PROMPT.format(
        domains="\n".join(f"- {d}" for d in domains)
    )
    try:
        resp    = await grader_model.ainvoke(prompt)
        results = _parse_json(resp.content)
        return [
            {"domain": r["domain"], "score": float(r["score"]), "tier": f"llm:{r.get('reason','')}"}
            for r in results
        ]
    except Exception:
        return [{"domain": d, "score": 0.4, "tier": "unknown"} for d in domains]


async def source_authority(source_urls: list[str], grader_model=None) -> dict:
    """
    每个来源域名的平均权威性评分（0.0–1.0）。

    高置信度规则层级：.gov=1.0、.edu=0.9、已知学术/新闻=0.8-0.9、
    wikipedia=0.75。其余全部（.com、.org、未知 TLD）在提供 grader_model 时
    推迟到一次批量 LLM 调用中处理 —— 正因如此，像 sciencedirect.com 或
    bmj.com 这样的来源才能被识别为高权威。
    若没有 grader_model，推迟域名回退为 .org=0.65 / .com=0.5 / 0.4。
    """
    if not source_urls:
        return {"avg_authority_score": 0.0, "breakdown": []}

    domains = [urlparse(u).netloc.lower().removeprefix("www.") for u in source_urls if u]

    breakdown = []
    deferred  = []
    for domain in domains:
        match = _score_domain(domain)
        if match is not None:
            score, tier = match
            breakdown.append({"domain": domain, "score": score, "tier": tier})
        else:
            deferred.append(domain)

    if deferred:
        score_map = {}
        if grader_model:
            # 用一次 LLM 调用为去重后的推迟域名批量评分
            scored    = await _llm_score_domains(sorted(set(deferred)), grader_model)
            score_map = {s["domain"]: s for s in scored}
        for domain in deferred:
            entry = score_map.get(domain)
            if entry is None:
                fb_score, fb_tier = _fallback_score(domain)
                entry = {"domain": domain, "score": fb_score, "tier": fb_tier}
            breakdown.append(dict(entry))

    if not breakdown:
        return {"avg_authority_score": 0.0, "breakdown": []}

    avg = sum(b["score"] for b in breakdown) / len(breakdown)

    return {
        "avg_authority_score": round(avg, 3),
        "breakdown": sorted(breakdown, key=lambda x: -x["score"]),
    }


# ---------------------------------------------------------------------------
# 4. 子主题覆盖度
# ---------------------------------------------------------------------------

# 下面两组提示词会发送给 LLM，属于提示词内容，保留英文
_SUBTOPICS_PROMPT = """
You are a research quality evaluator.

List the key subtopics that a comprehensive research report on the following
question must cover to be considered thorough.

Guidelines for choosing how many subtopics to list:
- Narrow or specific questions (e.g. a single fact, date, or person): 2-3 subtopics
- Moderate questions (e.g. a process, event, or comparison): 4-5 subtopics
- Broad or complex questions (e.g. a trend, field, or multi-factor topic): 6-7 subtopics

Return ONLY a JSON array of short subtopic strings (3-6 words each).
Do not include any explanation or extra text.

Question: {query}
""".strip()

_COVERAGE_ONE_PROMPT = """
You are a research quality evaluator.

Decide whether the single subtopic below is clearly and substantively addressed
in the research report (not merely mentioned in passing).

Subtopic: {subtopic}

Report:
{report}

Return ONLY a JSON object: {{"covered": true}} or {{"covered": false}}.
""".strip()


async def _check_one_subtopic(subtopic: str, report: str, grader_model) -> bool:
    """独立判断单个子主题的覆盖情况（避免批量效应）。"""
    try:
        resp   = await grader_model.ainvoke(
            _COVERAGE_ONE_PROMPT.format(subtopic=subtopic, report=report[:12000]))
        result = _parse_json(resp.content)
        return bool(result.get("covered", False))
    except Exception:
        return False


async def subtopic_coverage(query: str, report: str, grader_model, subtopics=None) -> dict:
    """
    由 LLM 评判对预期子主题的覆盖情况（约 2 次调用，约 $0.02/查询）。

    第 1 步：为该查询生成预期子主题（数量自适应）。
    第 2 步：检查报告覆盖了哪些子主题。

    `subtopics`：可选的预生成子主题列表。提供时跳过第 1 步，并针对这组
    确切的子主题评判覆盖度 —— 这能在多次调用之间固定子主题集合
    （例如用于扰动测试，避免覆盖度被重新生成的漂移所混淆）。
    """
    if subtopics is None:
        try:
            resp1     = await grader_model.ainvoke(_SUBTOPICS_PROMPT.format(query=query))
            subtopics = _parse_json(resp1.content)
        except Exception:
            return {
                "expected_subtopics":     [],
                "covered":                [],
                "missing":                [],
                "subtopic_coverage_rate": None,
                "error":                  "Failed to parse subtopics from LLM",
            }

    # 独立判断每个子主题（每个一次调用，并发执行），
    # 以免某个子主题的结论被同一批中的其他子主题影响。
    flags   = await asyncio.gather(
        *(_check_one_subtopic(s, report, grader_model) for s in subtopics)
    )
    covered = [s for s, ok in zip(subtopics, flags) if ok]
    missing = [s for s, ok in zip(subtopics, flags) if not ok]

    total = len(subtopics)
    return {
        "expected_subtopics":     subtopics,
        "covered":                covered,
        "missing":                missing,
        "subtopic_coverage_rate": round(len(covered) / total, 3) if total else 0.0,
    }


# ---------------------------------------------------------------------------
# 5. 无支撑断言率
# ---------------------------------------------------------------------------

# 下面两组提示词会发送给 LLM，属于提示词内容，保留英文
_EXTRACT_CLAIMS_PROMPT = """
You are a fact-checking assistant.

Extract specific, verifiable factual claims from the research report below.
Focus on concrete statements: dates, numbers, names, causal relationships, statistics.
Exclude: opinions, vague generalisations, hedged statements, and section summaries.

Guidelines for how many claims to extract:
- Short or narrow reports: 3-5 claims
- Typical research reports: 6-8 claims
- Long or multi-topic reports: up to 10 claims

Report (first 3000 characters):
{report}

Return ONLY a JSON array of claim strings. No explanation or extra text.
""".strip()

_VERIFY_CLAIMS_PROMPT = """
You are a fact-checking assistant.

For each claim below, assign a credibility score (0.0–1.0) based on
how well the source context supports it.

Scoring rules:

score = 1.0  →  "supported"
  The context directly and explicitly states the claim.
  This is the ONLY valid score for claims containing specific data:
  numbers, percentages, dates, statistics, version numbers, or named figures.

score = 0.3–0.9  →  "inferred"
  The claim is a reasonable synthesis drawn from multiple parts of the context,
  but no single source states it explicitly.
  Only applicable to conceptual or summary statements — NEVER to data claims.
  Score higher when more authoritative sources (3+) support the inference (→ 0.75–0.9).
  Score lower when only 1–2 commercial sources loosely imply it (→ 0.3–0.5).

score = 0.0  →  "unsupported"
  No traceable basis in context, contradicted, or specific data not found in context.

Important: do not use your own knowledge — only use the provided context.
Any claim with a specific number, date, or statistic not in the context → score 0.0.

Claims:
{claims}

Source context (first 4000 characters):
{context}

Return ONLY a JSON array, one object per claim:
[
  {{"claim": "...", "category": "supported",   "score": 1.0,  "reason": "..."}},
  {{"claim": "...", "category": "inferred",    "score": 0.75, "reason": "..."}},
  {{"claim": "...", "category": "unsupported", "score": 0.0,  "reason": "..."}}
]
""".strip()


async def _verify_one_claim(claim: str, context_text: str, grader_model) -> dict:
    """在给定上下文下独立评估单条断言（避免批量效应）。"""
    try:
        resp   = await grader_model.ainvoke(
            _VERIFY_CLAIMS_PROMPT.format(
                claims=json.dumps([claim], ensure_ascii=False),
                context=context_text[:4000],
            )
        )
        result = _parse_json(resp.content)
        if isinstance(result, list) and result:
            return result[0]
        if isinstance(result, dict):
            return result
    except Exception:
        pass
    return {"claim": claim, "category": "unsupported", "score": 0.0,
            "reason": "verification failed"}


async def unsupported_claim(report: str, context, grader_model, claims=None) -> dict:
    """
    断言级可信度评分（约 2 次调用，约 $0.02/查询）。

    第 1 步：从报告中抽取 3-10 条具体、可验证的断言。
    第 2 步：把每条断言评为 supported (1.0)、inferred (0.3-0.9) 或 unsupported (0.0)。

    数据类断言（数字/日期/统计）只可能是 supported 或 unsupported，绝不可能是 inferred。
    返回 avg_claim_score（整体可信度）、unsupported_claim_rate（仅硬性失败）、
    inferred_claim_rate（推理占比）以及逐条断言的明细。

    `claims`：可选的预抽取断言列表。提供时跳过抽取步骤，直接为这组确切断言
    评分 —— 这能在多次调用之间固定断言集合
    （例如用于扰动测试，避免评分被重新抽取的漂移所混淆）。
    """
    context_text = context if isinstance(context, str) else json.dumps(context)

    if claims is None:
        try:
            resp1  = await grader_model.ainvoke(_EXTRACT_CLAIMS_PROMPT.format(report=report[:12000]))
            claims = _parse_json(resp1.content)
        except Exception as e:
            return {
                "total_claims": 0, "avg_claim_score": None,
                "unsupported_claim_rate": None, "inferred_claim_rate": None,
                "breakdown": [], "error": f"Claim extraction failed: {e}",
            }

    if not claims:
        return {
            "total_claims": 0, "avg_claim_score": 0.0,
            "unsupported_claim_rate": 0.0, "inferred_claim_rate": 0.0,
            "breakdown": [],
        }

    # 独立评估每条断言（每条一次 LLM 调用，并发执行）。
    # 若批量评分，一条被污染或荒谬的断言会带偏模型对同一批中无关断言的判断
    # —— 于是某条断言的分数取决于它的同批伙伴。独立评分让每条断言的分数自洽。
    scored_claims = await asyncio.gather(
        *(_verify_one_claim(c, context_text, grader_model) for c in claims)
    )

    supported_l   = [c for c in scored_claims if c.get("category") == "supported"]
    inferred_l    = [c for c in scored_claims if c.get("category") == "inferred"]
    unsupported_l = [c for c in scored_claims if c.get("category") == "unsupported"]

    total     = len(scored_claims)
    avg_score = round(sum(float(c.get("score", 0.0)) for c in scored_claims) / total, 3) if total else 0.0

    return {
        "total_claims":           total,
        "supported_count":        len(supported_l),
        "inferred_count":         len(inferred_l),
        "unsupported_count":      len(unsupported_l),
        "avg_claim_score":        avg_score,
        "unsupported_claim_rate": round(len(unsupported_l) / total, 3) if total else 0.0,
        "inferred_claim_rate":    round(len(inferred_l)    / total, 3) if total else 0.0,
        "breakdown": [
            {
                "claim":    c.get("claim"),
                "category": c.get("category"),
                "score":    float(c.get("score", 0.0)),
                "reason":   c.get("reason", ""),
            }
            for c in scored_claims
        ],
    }


# ---------------------------------------------------------------------------
# 编排（由 run_eval.py 和 benchmark.py 共用）
# ---------------------------------------------------------------------------

def skipped(reason: str) -> dict:
    """指标无法运行时返回的哨兵值，避免一次失败导致整体中止。"""
    return {"status": "skipped", "reason": reason}


def is_skipped(value) -> bool:
    return isinstance(value, dict) and value.get("status") == "skipped"


async def evaluate_report(
    report: str,
    sources: list,
    context,
    query: str,
    grader_model,
    *,
    run_subtopic: bool = True,
    run_unsupported: bool = True,
) -> dict:
    """在标准化的 `suite.evaluate()` 之上提供向后兼容的字典视图。

    返回 `{metric_name: 明细字典 | skipped 哨兵}`，而由开关关闭的指标对应 `None`
    —— 正是 run_eval/benchmark 已经期望的形态。
    新代码应优先使用 `suite.evaluate()`，它返回带有 group / score /
    概念对齐元数据的类型化 `MetricResult` 对象。
    """
    # 延迟导入以打破 metrics.py 与 suite.py 之间的循环导入。
    from evals.quality_eval.suite import evaluate as _suite_evaluate, default_metrics
    from evals.quality_eval.base import EvalSample

    sample  = EvalSample(query=query, report=report, sources=sources, context=context)
    results = await _suite_evaluate(
        sample, default_metrics(run_subtopic, run_unsupported), grader_model)

    out = {r.name: (skipped(r.reason) if r.skipped else r.breakdown) for r in results}
    out.setdefault("subtopic_coverage", None)     # 被开关关闭时为 None
    out.setdefault("unsupported_claim", None)
    return out
