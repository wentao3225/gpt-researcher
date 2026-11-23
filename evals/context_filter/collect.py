"""第 1 步：对每个查询只运行一次，记录进入上下文过滤器的内容。

每个抓取到的页面都通过
``ContextManager.get_similar_content_by_query(query, pages)`` 进入报告。
这里记录每次调用的输入，这样 ``replay.py`` 就能用完全相同的页面、
在每种策略下重建上下文。只有过滤器会变化；搜索结果和抓取这些
一次运行中噪声最大的部分都保持固定。

    python -m evals.context_filter.collect --out runs/ --concurrency 4
"""

import argparse
import asyncio
import json
import time
from pathlib import Path

from gpt_researcher import GPTResearcher
from gpt_researcher.skills.context_manager import ContextManager

HERE = Path(__file__).parent

# researcher -> 已记录的调用。对整个进程只打补丁一次，
# 因为多个运行是并发执行的，按运行打补丁/恢复会发生竞争。
_RECORDINGS: dict[int, list] = {}
_original = ContextManager.get_similar_content_by_query


async def _recording(self, query, pages):
    calls = _RECORDINGS.get(id(self.researcher))
    if calls is not None:
        calls.append({"query": query, "pages": [
            {"url": p.get("url"), "title": p.get("title"), "raw_content": p.get("raw_content")}
            for p in pages
        ]})
    return await _original(self, query, pages)


ContextManager.get_similar_content_by_query = _recording


async def collect_one(item: dict, out: Path, sem: asyncio.Semaphore) -> None:
    path = out / f"{item['id']}.json"
    if path.exists():
        return
    async with sem:
        researcher = GPTResearcher(query=item["query"], report_type="research_report")
        calls = _RECORDINGS[id(researcher)] = []
        start = time.perf_counter()
        try:
            await researcher.conduct_research()
        except Exception as e:
            print(f"{item['id']}: FAILED {e}")
            return
        finally:
            _RECORDINGS.pop(id(researcher), None)
        path.write_text(json.dumps({
            **item,
            "research_seconds": round(time.perf_counter() - start, 1),
            "research_cost": researcher.get_costs(),
            "calls": calls,
        }))
        print(f"{item['id']}: {len(calls)} filter calls, "
              f"{sum(len(c['pages']) for c in calls)} pages, {time.perf_counter() - start:.0f}s", flush=True)


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=HERE / "runs")
    parser.add_argument("--concurrency", type=int, default=4)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    items = json.loads((HERE / "queries.json").read_text())
    sem = asyncio.Semaphore(args.concurrency)
    await asyncio.gather(*[collect_one(item, args.out, sem) for item in items])


if __name__ == "__main__":
    asyncio.run(main())
