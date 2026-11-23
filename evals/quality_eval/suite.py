"""
指标套件 —— 具体的 BaseMetric 实现 + evaluate() 入口。
--------------------------------------------------------------------------------
每个类都是 metrics.py 中某个函数的薄封装，把 I/O 统一为
MetricResult（score 始终为“越高越好”；原始明细保留在
`breakdown` 中）。把原始函数保持独立意味着现有的单元测试
和扰动检查依然有效 —— 本层只负责统一结构，
并附加概念对齐元数据（group / aligned_with）。

概念本体（见 quality_eval/README.md）：
    Faithfulness（忠实度）  → unsupported_claim            （对齐 RAGAS Faithfulness）
    Answer Relevancy（答案相关性） → subtopic_coverage      （对齐 RAGAS answer relevancy）
    Source Quality（来源质量）     → source_authority, diversity（自主智能体扩展）
    Citation（引用）       → citation_faithfulness        （对齐 ALCE 引用精确度）
"""

from __future__ import annotations

from evals.quality_eval import metrics as m
from evals.quality_eval.base import BaseMetric, EvalSample, MetricResult, Group


# ---------------------------------------------------------------------------
# 引用（对齐 ALCE 引用精确度 —— 结构性）
# ---------------------------------------------------------------------------

class CitationFaithfulnessMetric(BaseMetric):
    name         = "citation_faithfulness"
    group        = Group.CITATION
    aligned_with = "ALCE citation precision (structural)"
    needs_llm    = False

    async def measure(self, sample: EvalSample, grader_model=None) -> MetricResult:
        raw   = m.citation_faithfulness(sample.report, sample.sources, context=sample.context)
        score = raw.get("citation_faithfulness")
        return MetricResult(
            name=self.name, group=self.group, aligned_with=self.aligned_with,
            score=score,
            reason=("no References section to compare against" if score is None
                    else f"{len(raw['listed_only_domains'])} listed-only (decorative) refs"),
            breakdown=raw,
        )


# ---------------------------------------------------------------------------
# 来源质量（自主智能体扩展）
# ---------------------------------------------------------------------------

class SourceDiversityMetric(BaseMetric):
    name         = "source_diversity"
    group        = Group.SOURCE_QUALITY
    aligned_with = "extension (autonomous agent)"
    needs_llm    = False

    async def measure(self, sample: EvalSample, grader_model=None) -> MetricResult:
        raw = m.source_diversity(sample.sources)
        return MetricResult(
            name=self.name, group=self.group, aligned_with=self.aligned_with,
            score=raw["diversity_ratio"],
            reason=f"{raw['unique_domains']}/{raw['total_sources']} unique domains, entropy={raw['domain_entropy']}",
            breakdown=raw,
        )


class SourceAuthorityMetric(BaseMetric):
    name         = "source_authority"
    group        = Group.SOURCE_QUALITY
    aligned_with = "extension (autonomous agent)"
    needs_llm    = True   # optional: falls back to the rule table without a grader

    async def measure(self, sample: EvalSample, grader_model=None) -> MetricResult:
        raw = await m.source_authority(sample.sources, grader_model=grader_model)
        return MetricResult(
            name=self.name, group=self.group, aligned_with=self.aligned_with,
            score=raw["avg_authority_score"],
            reason=f"avg authority over {len(raw['breakdown'])} domains",
            breakdown=raw,
        )


# ---------------------------------------------------------------------------
# 答案相关性（对齐 RAGAS answer relevancy —— 实现衡量的是完整度）
# ---------------------------------------------------------------------------

class SubtopicCoverageMetric(BaseMetric):
    name         = "subtopic_coverage"
    group        = Group.ANSWER_RELEVANCY
    aligned_with = "RAGAS answer relevancy (measures completeness)"
    needs_llm    = True

    async def measure(self, sample: EvalSample, grader_model=None) -> MetricResult:
        raw   = await m.subtopic_coverage(sample.query, sample.report, grader_model)
        score = raw.get("subtopic_coverage_rate")
        if score is None:
            return MetricResult.skip(self, raw.get("error", "coverage unavailable"))
        return MetricResult(
            name=self.name, group=self.group, aligned_with=self.aligned_with,
            score=score,
            reason=f"{len(raw['covered'])}/{len(raw['expected_subtopics'])} subtopics covered",
            breakdown=raw,
        )


# ---------------------------------------------------------------------------
# 忠实度（对齐 RAGAS Faithfulness）
# ---------------------------------------------------------------------------

class UnsupportedClaimMetric(BaseMetric):
    name         = "unsupported_claim"
    group        = Group.FAITHFULNESS
    aligned_with = "RAGAS Faithfulness"
    needs_llm    = True

    async def measure(self, sample: EvalSample, grader_model=None) -> MetricResult:
        if not sample.context:
            return MetricResult.skip(self, "no source context available")
        raw   = await m.unsupported_claim(sample.report, sample.context, grader_model)
        score = raw.get("avg_claim_score")
        if score is None:
            return MetricResult.skip(self, raw.get("error", "scoring failed"))
        return MetricResult(
            name=self.name, group=self.group, aligned_with=self.aligned_with,
            score=score,   # higher = more faithful; unsupported_rate is in breakdown
            reason=f"{raw['unsupported_count']}/{raw['total_claims']} claims unsupported",
            breakdown=raw,
        )


# ---------------------------------------------------------------------------
# 注册表 + 入口
# ---------------------------------------------------------------------------

def default_metrics(run_subtopic: bool = True, run_unsupported: bool = True) -> list[BaseMetric]:
    """零成本指标始终运行；两个 LLM 指标由开关控制（按“便宜 → 昂贵”排序）。"""
    suite: list[BaseMetric] = [
        CitationFaithfulnessMetric(),
        SourceDiversityMetric(),
        SourceAuthorityMetric(),
    ]
    if run_subtopic:
        suite.append(SubtopicCoverageMetric())
    if run_unsupported:
        suite.append(UnsupportedClaimMetric())
    return suite


async def evaluate(sample: EvalSample, metrics: list[BaseMetric], grader_model=None) -> list[MetricResult]:
    """Run each metric on the sample. A metric failure becomes a skipped
    MetricResult rather than aborting the whole evaluation (DeepEval-style)."""
    results: list[MetricResult] = []
    for metric in metrics:
        try:
            results.append(await metric.measure(sample, grader_model))
        except Exception as e:
            results.append(MetricResult.skip(metric, f"{type(e).__name__}: {e}"))
    return results
