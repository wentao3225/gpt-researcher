"""
quality_eval 指标的扰动（行为）测试。
--------------------------------------
验证指标的**可靠性**：当报告或其来源被刻意降级时，
相应指标是否会朝预期方向移动 —— 并且随着降级加剧而单调变化？
这能抓住纯逻辑无法发现的实现 bug。例如，如果 `_split_at_references`
被写坏了，即使移除全部行内引用，citation_faithfulness 仍会保持高位；
本测试就能抓住这一点。

重要 —— 可靠性 vs 有效性：
  本测试只证明指标实现正确且在方向上敏感（可靠性）。
  它**不能**证明这些指标在衡量“报告质量”（有效性）。
  尤其要注意，claim_score 已知会偏重抽取式（照抄来源）写作而轻视
  抽象式（综合归纳）写作 —— 这是有效性层面的另一个问题，
  扰动测试无法解决。

在多个真实的 gpt-4o 报告（fixtures）上运行，
使结论成为指标的一致属性，而非某一份报告的偶然产物。

用法：
    python -m evals.quality_eval.perturbation
"""

import asyncio
import json
import os
import random
import re
import sys
from pathlib import Path

from evals.quality_eval import metrics

FIXTURES  = Path(__file__).parent / "fixtures" / "perturbation_fixtures.json"
FRACTIONS = [0.0, 0.25, 0.50, 0.75, 1.0]

# 用于降低 source_authority 的低权威性替换 URL
_LOW_AUTHORITY = [
    "https://medium.com/@user/post-{}",
    "https://www.reddit.com/r/topic/comments/{}",
    "https://someblog.blogspot.com/{}",
]

# 用于破坏事实性断言的极性反转词（_corrupt_text）
_POLARITY = {
    "increase": "decrease", "increases": "decreases", "increased": "decreased",
    "decrease": "increase", "decreases": "increases", "decreased": "increased",
    "effective": "ineffective", "ineffective": "effective",
    "higher": "lower", "lower": "higher", "more": "less", "less": "more",
    "positive": "negative", "negative": "positive",
    "improve": "worsen", "improves": "worsens", "improved": "worsened",
    "reduce": "raise", "reduces": "raises", "reduced": "raised",
    "significant": "negligible", "significantly": "negligibly",
}

# 挑选“关键”实义词进行破坏时跳过的停用词
_STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being", "of", "to",
    "in", "and", "or", "for", "with", "on", "at", "by", "as", "that", "this", "it",
    "its", "from", "which", "their", "they", "these", "those", "has", "have", "had",
    "not", "but", "can", "may", "will", "such", "also", "than", "when", "while",
}

# 用于替换关键实体的固定诱饵词（给定 seed 后结果确定）
_DECOY_WORDS = [
    "wombat", "teapot", "bicycle", "umbrella", "marshmallow", "saxophone",
    "tundra", "pelican", "concrete", "lullaby", "thermostat", "origami",
]


# ---------------------------------------------------------------------------
# 扰动函数 —— 每个函数按 `fraction` 降低**一个**输入维度
# ---------------------------------------------------------------------------

def degrade_citations(report: str, fraction: float) -> str:
    """仅从**正文**中移除 `fraction` 比例的行内 `[text](url)` 链接。

    末尾的 `## References` 列表保持不变，每个被移除的链接
    替换为其锚文本，使正文仍可读。预期效果：citation_faithfulness 下降
    （来源仍列在 References 中，但不再在正文中被引用）。
    """
    body, refs = metrics._split_at_references(report)
    links = list(re.finditer(r'\[([^\]]+)\]\((https?://[^\)]+)\)', body))
    n = round(len(links) * fraction)
    # 从后往前移除最后 n 个链接，使前面的偏移量保持有效。
    for m in sorted(links[len(links) - n:], key=lambda x: -x.start()):
        body = body[:m.start()] + m.group(1) + body[m.end():]
    return body + refs


def degrade_authority(sources: list[str], fraction: float) -> list[str]:
    """把 `fraction` 比例的来源替换为低权威性的博客/论坛 URL。

    预期效果：随着权威域名（.gov / 期刊）被换成博客和论坛，
    avg_authority_score 下降。
    """
    out = list(sources)
    n = round(len(out) * fraction)
    for i in range(n):
        out[i] = _LOW_AUTHORITY[i % len(_LOW_AUTHORITY)].format(i)
    return out


def degrade_diversity(sources: list[str], fraction: float) -> list[str]:
    """把 `fraction` 比例的来源集中到同一个共享域名下。

    预期效果：随着不同域名被替换为同一域名下的路径，
    diversity_ratio 朝 1/N 下降。
    """
    out = list(sources)
    n = round(len(out) * fraction)
    for i in range(n):
        out[i] = f"https://onedomain.com/page{i}"
    return out


def degrade_subtopics(report: str, fraction: float) -> str:
    """移除报告中最后 `fraction` 比例的章节（`##` 标题）。

    正文按 `## ` 标题切分；丢弃末尾 `fraction` 比例的章节
    （引言和 References 保留）。预期效果：随着整节内容
    —— 以及它们所覆盖的子主题 —— 消失，subtopic_coverage 下降。
    """
    body, refs = metrics._split_at_references(report)
    parts = re.split(r'(?=^##\s)', body, flags=re.M)   # keep each header with its content
    n_drop = round(len(parts) * fraction)
    kept = parts[:len(parts) - n_drop] if n_drop else parts
    return "".join(kept) + ("\n\n" + refs if refs else "")


def _corrupt_text(text: str, rng: random.Random) -> str:
    """确定性地破坏一条断言，使其与上下文相矛盾。

    三个基于规则、可复现的操作（给定 seed）：
      1. 翻转极性词（increase<->decrease、effective<->ineffective 等）
      2. 把每个数字替换成另一个随机数字
      3. 把最长的两个实义词（很可能是关键实体/名词）替换为固定诱饵词
         —— 正因有这一步，**定性**断言（例如不含数字的法律定义）
         才能被破坏。
    """
    def _flip(m):
        w = m.group()
        new = _POLARITY[w.lower()]
        return new.capitalize() if w[:1].isupper() else new
    text = re.sub(r'\b(' + '|'.join(_POLARITY) + r')\b', _flip, text, flags=re.I)
    text = re.sub(r'\b\d+(?:\.\d+)?(%?)\b',
                  lambda m: str(rng.randint(1, 99)) + m.group(1), text)

    # 关键实体破坏：把最长的两个实义词替换为诱饵词
    words = text.split()
    ranked = sorted(
        (i for i, w in enumerate(words)
         if w.strip(".,;:()").lower() not in _STOPWORDS and len(w.strip(".,;:()")) > 3),
        key=lambda i: -len(words[i]),
    )
    for i in ranked[:2]:
        words[i] = rng.choice(_DECOY_WORDS)
    return " ".join(words)


def corrupt_claims(claims: list[str], fraction: float, seed: int = 0) -> list[str]:
    """破坏**固定**断言集合中 `fraction` 比例的部分（数字 + 极性反转），
    使被破坏的断言与上下文相矛盾。

    与破坏整份报告相比的关键差别：断言集合只抽取**一次**并保持不变；
    只有不断扩大的子集被破坏。这消除了重新抽取带来的漂移，
    而正是这种漂移让报告级扰动充满噪声。预期效果：
    unsupported_claim_rate 随 `fraction` 单调上升。
    """
    rng = random.Random(seed)
    # 固定洗牌，使被破坏的子集在不同 fraction 之间是嵌套的
    # （25% 被破坏 ⊂ 50% 被破坏 ⊂ ...）；否则单调性就是虚假的。
    order = list(range(len(claims)))
    rng.shuffle(order)
    n = round(len(claims) * fraction)
    corrupt_idx = set(order[:n])
    return [_corrupt_text(c, rng) if i in corrupt_idx else c
            for i, c in enumerate(claims)]


# ---------------------------------------------------------------------------
# 运行器
# ---------------------------------------------------------------------------

def _monotone(vals, increasing: bool = False) -> bool:
    clean = [v for v in vals if v is not None]
    if increasing:
        return all(a <= b + 1e-9 for a, b in zip(clean, clean[1:]))
    return all(a >= b - 1e-9 for a, b in zip(clean, clean[1:]))


def _row(topic: str, vals, increasing: bool = False) -> str:
    cells = "".join(f"{v:>8.2f}" if v is not None else f"{'--':>8}" for v in vals)
    verdict = "PASS" if _monotone(vals, increasing) else "FAIL  <-- not monotone!"
    return f"  {topic:<10}{cells}   {verdict}"


def _header() -> str:
    return f"  {'topic':<10}" + "".join(f"{int(f*100):>7}%" for f in FRACTIONS) + "   monotone?"


async def main():
    if not FIXTURES.exists():
        raise SystemExit(f"Fixtures not found: {FIXTURES}\nGenerate them first (3 gpt-4o reports).")
    fixtures = json.loads(FIXTURES.read_text(encoding="utf-8"))
    print(f"Loaded {len(fixtures)} fixtures: {[f['topic'] for f in fixtures]}\n")

    # --- citation_faithfulness：移除正文中的行内引用（零成本） ---
    print("=== citation_faithfulness  (remove inline body citations) ===")
    print(_header())
    for fx in fixtures:
        vals = [
            metrics.citation_faithfulness(
                degrade_citations(fx["report"], f), fx["sources"], context=fx["context"]
            )["citation_faithfulness"]
            for f in FRACTIONS
        ]
        print(_row(fx["topic"], vals))

    # --- source_diversity：把域名集中到一个（零成本） ---
    print("\n=== source_diversity  (collapse onto one domain) ===")
    print(_header())
    for fx in fixtures:
        vals = [
            metrics.source_diversity(degrade_diversity(fx["sources"], f))["diversity_ratio"]
            for f in FRACTIONS
        ]
        print(_row(fx["topic"], vals))

    # --- source_authority：替换为低权威来源（由 LLM 评分） ---
    from langchain_openai import ChatOpenAI
    grader = ChatOpenAI(temperature=0, model_name="gpt-4-turbo",
                        openai_api_key=os.environ["OPENAI_API_KEY"])
    print("\n=== source_authority  (swap to low-authority, LLM-scored) ===")
    print(_header())
    for fx in fixtures:
        vals = []
        for f in FRACTIONS:
            r = await metrics.source_authority(degrade_authority(fx["sources"], f), grader_model=grader)
            vals.append(r["avg_authority_score"])
        print(_row(fx["topic"], vals))

    # --- subtopic_coverage：移除章节，使用固定的子主题集合 ---
    print("\n=== subtopic_coverage  (remove report sections, fixed subtopics) ===")
    print(_header())
    for fx in fixtures:
        # 只生成一次预期子主题（基于完整报告），然后在每个降级档位复用，
        # 使覆盖度不被重新生成的漂移所混淆。
        base      = await metrics.subtopic_coverage(fx["query"], fx["report"], grader)
        subtopics = base.get("expected_subtopics", [])
        vals = []
        for f in FRACTIONS:
            r = await metrics.subtopic_coverage(
                fx["query"], degrade_subtopics(fx["report"], f), grader, subtopics=subtopics)
            vals.append(r.get("subtopic_coverage_rate"))
        print(_row(fx["topic"], vals))

    # --- unsupported_claim：破坏固定断言集合（预期上升） ---
    print("\n=== unsupported_claim  (corrupt fixed claim set, expect RISE) ===")
    print(_header())
    for fx in fixtures:
        # 只抽取一次断言，然后破坏同一集合中不断扩大的子集。
        base   = await metrics.unsupported_claim(fx["report"], fx["context"], grader)
        claims = [c["claim"] for c in base.get("breakdown", [])]
        vals = []
        for f in FRACTIONS:
            r = await metrics.unsupported_claim(
                fx["report"], fx["context"], grader, claims=corrupt_claims(claims, f))
            vals.append(r.get("unsupported_claim_rate"))
        print(_row(fx["topic"], vals, increasing=True))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    # 若尚未设置 API key，则从文件加载（source_authority 需要）。
    if not os.getenv("OPENAI_API_KEY"):
        kp = Path("D:/chenh/api_keys/gpt_api_key.txt")
        if kp.exists():
            os.environ["OPENAI_API_KEY"] = kp.read_text().strip()
    asyncio.run(main())
