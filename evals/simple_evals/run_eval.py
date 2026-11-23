import asyncio
import os
import sys
import argparse
import json
import subprocess

# 确保在 Windows 终端（GBK → UTF-8）下 Unicode 输出正常
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, List, TypeVar
from tqdm import tqdm
from dotenv import load_dotenv
from gpt_researcher.agent import GPTResearcher
from gpt_researcher.utils.enum import ReportType, ReportSource, Tone
from evals.simple_evals.simpleqa_eval import SimpleQAEval
from langchain_openai import ChatOpenAI

GRADER_MODEL_NAME = "gpt-4-turbo"
LOGS_DIR = Path(__file__).parent / "logs"


def get_git_commit() -> str:
    """返回当前 HEAD 的提交哈希；若不在 git 仓库中则返回 'unknown'。"""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, check=True
        )
        return result.stdout.strip()
    except Exception:
        return "unknown"

# 泛型函数用的类型变量
T = TypeVar('T')
R = TypeVar('R')

def map_with_progress(fn: Callable[[T], R], items: List[T]) -> List[R]:
    """对每个元素应用函数，并附带进度条。"""
    return [fn(item) for item in tqdm(items)]

def _check_env_vars():
    """校验必需的环境变量是否已设置。放在 main() 内调用，以便 --help 能正常工作。"""
    load_dotenv()
    required = ["OPENAI_API_KEY", "TAVILY_API_KEY"]
    missing = [v for v in required if not os.getenv(v)]
    if missing:
        raise ValueError(f"Missing required environment variables: {', '.join(missing)}")
    if not os.getenv("LANGCHAIN_API_KEY"):
        print("Note: LANGCHAIN_API_KEY not set — LangSmith tracing disabled (optional)")

async def evaluate_single_query(query: str, evaluator: SimpleQAEval) -> dict:
    """运行单个评估查询并返回结果"""
    print(f"\nEvaluating query: {query}")
    
    t_start = time.perf_counter()   # 每个查询的计时起点

    researcher = GPTResearcher(
        query=query,
        report_type=ReportType.ResearchReport.value,
        report_format="markdown",
        report_source=ReportSource.Web.value,
        tone=Tone.Objective,
        verbose=True
    )
    context = await researcher.conduct_research()
    report = await researcher.write_report()

    latency_seconds = round(time.perf_counter() - t_start, 2)   # 响应延迟

    # 获取正确答案并进行评估
    example = next((ex for ex in evaluator.examples if ex['problem'] == query), None)   # 没有 'problem' 键则为 None
    if example is None:
        raise ValueError(f"No matching example found for query: {query}")
    correct_answer = example['answer']

    eval_result = evaluator.evaluate_example({
        "problem": query,
        "answer": correct_answer,
        "predicted": report
    })

    # 结果收集与格式化
    result = {
        'query': query,
        'context_length': len(context),
        'report_length': len(report),
        'latency_seconds': latency_seconds,
        'cost': researcher.get_costs(),
        'sources': researcher.get_source_urls(),
        'evaluation_score': eval_result["score"],
        'evaluation_grade': eval_result["metrics"]["grade"]
    }

    print(f"✓ Completed research and evaluation")
    print(f"  - Sources found: {len(result['sources'])}")
    print(f"  - Evaluation grade: {result['evaluation_grade']}")
    print(f"  - Latency: {latency_seconds}s")
    print(f"  - Cost: ${result['cost']:.4f}")

    return result

async def main(num_examples: int):
    if num_examples < 1:
        raise ValueError("num_examples must be at least 1")
    # 开始计时
    _check_env_vars()

    run_start = time.perf_counter()
    run_timestamp = datetime.now(timezone.utc).isoformat()

    try:
        # 初始化评估器
        grader_model = ChatOpenAI(
            temperature=0,
            model_name=GRADER_MODEL_NAME,
            openai_api_key=os.getenv("OPENAI_API_KEY")
        )
        evaluator = SimpleQAEval(grader_model=grader_model, num_examples=num_examples)
        
        if not evaluator.examples:
            raise ValueError("No examples loaded in evaluator")
        
        print(f"Starting GPT-Researcher evaluation with {num_examples} test queries...")
        
        # 对每个示例获取评估结果
        results = []
        for example in evaluator.examples:
            if 'problem' not in example:
                print(f"Warning: Skipping example without 'problem' key: {example}")
                continue
                
            query = example['problem']
            print(f"\nEvaluating query: {query}")
            try:
                result = await evaluate_single_query(query, evaluator)
                results.append(result)
                
            except Exception as e:
                print(f"✗ Error evaluating query: {str(e)}")
                results.append({
                    'query': query,
                    'error': str(e)
                })
        
        if not results:
            raise ValueError("No results generated")
            
        # 打印任意示例数下的汇总
        if num_examples > 0:  # Changed from > 1
            print("\n=== Evaluation Summary ===")
            print(f"Total queries tested: {len(evaluator.examples)}")
            successful = len([r for r in results if 'error' not in r])
            print(f"Successful queries: {successful}")
            print(f"Failed queries: {len(evaluator.examples) - successful}")
            
            if successful > 0:
                # 统计不同等级的个数
                correct = sum(1 for r in results if r.get('evaluation_grade') == "CORRECT")
                incorrect = sum(1 for r in results if r.get('evaluation_grade') == "INCORRECT")
                not_attempted = sum(1 for r in results if r.get('evaluation_grade') == "NOT_ATTEMPTED")
                
                print("\n=== AGGREGATE METRICS ===")
                metrics = {
                    "correct_rate": correct / successful,
                    "incorrect_rate": incorrect / successful,
                    "not_attempted_rate": not_attempted / successful,
                    "answer_rate": (correct + incorrect) / successful,
                }
                
                # 调试输出
                print("\nDebug counts:")
                print(f"Total successful: {successful}")
                print(f"CORRECT: {correct}")
                print(f"INCORRECT: {incorrect}")
                print(f"NOT_ATTEMPTED: {not_attempted}")
                
                # 计算准确率与 F1
                metrics["accuracy"] = (
                    correct / (correct + incorrect)  # Accuracy among attempted answers
                    if (correct + incorrect) > 0
                    else 0
                )
                
                # 精确率 = 正确数 / 已作答数
                precision = correct / (correct + incorrect) if (correct + incorrect) > 0 else 0
                
                # 召回率 = 正确数 / 总数
                recall = correct / successful if successful > 0 else 0
                
                # F1 = 2 * (精确率 * 召回率) / (精确率 + 召回率)
                metrics["f1"] = (
                    2 * (precision * recall) / (precision + recall)
                    if (precision + recall) > 0
                    else 0
                )
                
                print(json.dumps(metrics, indent=2))
                print("========================")
                print(f"Accuracy: {metrics['accuracy']:.3f}")
                print(f"F1 Score: {metrics['f1']:.3f}")
                
                total_cost = sum(r['cost'] for r in results if 'error' not in r)
                latencies = sorted(r['latency_seconds'] for r in results if 'error' not in r)
                avg_latency = sum(latencies) / successful
                p50_latency = latencies[int(successful * 0.50)]
                p95_latency = latencies[min(int(successful * 0.95), successful - 1)]
                print(f"\nTotal cost: ${total_cost:.4f}")
                print(f"Average cost per query: ${total_cost/successful:.4f}")
                print(f"Latency — avg: {avg_latency:.1f}s  p50: {p50_latency:.1f}s  p95: {p95_latency:.1f}s")

                # 写入结构化的 JSON 日志
                run_metadata = {
                    "timestamp": run_timestamp,
                    "git_commit": get_git_commit(),
                    "grader_model": GRADER_MODEL_NAME,
                    "researcher_model": os.getenv("SMART_LLM", "openai:gpt-4.1"),
                    "num_examples": num_examples,
                    "total_duration_seconds": round(time.perf_counter() - run_start, 2),
                }
                log_record = {
                    "run_metadata": run_metadata,
                    "aggregate_metrics": {
                        **metrics,
                        "total_cost": round(total_cost, 6),
                        "avg_cost_per_query": round(total_cost / successful, 6),
                        "avg_latency_seconds": round(avg_latency, 2),
                        "p50_latency_seconds": round(p50_latency, 2),
                        "p95_latency_seconds": round(p95_latency, 2),
                        "successful": successful,
                        "failed": len(evaluator.examples) - successful,
                    },
                    "results": [r for r in results],
                }
                LOGS_DIR.mkdir(exist_ok=True)
                log_filename = LOGS_DIR / f"eval_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}_n{num_examples}.jsonl"
                with open(log_filename, "w", encoding="utf-8") as f:
                    f.write(json.dumps(log_record) + "\n")
                print(f"\nStructured log saved → {log_filename}")

    except Exception as e:
        print(f"Fatal error in main: {str(e)}")
        raise

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run GPT-Researcher evaluation')
    parser.add_argument('--num_examples', type=int, default=1,
                      help='Number of examples to evaluate. Default is 1 example.')
    args = parser.parse_args()
    
    try:
        asyncio.run(main(args.num_examples))
    except KeyboardInterrupt:
        print("\nEvaluation interrupted by user")
    except Exception as e:
        print(f"Fatal error: {str(e)}")