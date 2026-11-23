"""
运行 GPT-Researcher 查询并评估其是否存在幻觉的脚本。
"""
import json
import logging
import random
import asyncio
import argparse
import os
from pathlib import Path
from typing import Dict, List, Optional
from dotenv import load_dotenv

from gpt_researcher.agent import GPTResearcher
from gpt_researcher.utils.enum import ReportType, ReportSource, Tone
from gpt_researcher.utils.logging_config import get_json_handler

from .evaluate import HallucinationEvaluator

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 加载环境变量
load_dotenv()

# 默认路径
DEFAULT_OUTPUT_DIR = "evals/hallucination_eval/results"
DEFAULT_QUERIES_FILE = "evals/hallucination_eval/inputs/search_queries.jsonl"

class ResearchEvaluator:
    """运行 GPT-Researcher 查询并评估响应是否存在幻觉。"""

    def __init__(self, queries_file: str = DEFAULT_QUERIES_FILE):
        """
        初始化研究评估器。

        Args:
            queries_file: 包含搜索查询的 JSONL 文件路径
        """

        self.queries_file = Path(queries_file)
        self.hallucination_evaluator = HallucinationEvaluator()
        
    def load_queries(self, num_queries: Optional[int] = None) -> List[str]:
        """
        加载查询，并可选择性地抽样。

        Args:
            num_queries: 可选，随机抽样的查询数量

        Returns:
            查询字符串列表
        """
        queries = []
        with open(self.queries_file) as f:
            for line in f:
                data = json.loads(line.strip())
                queries.append(data["question"])
            
        if num_queries and num_queries < len(queries):
            return random.sample(queries, num_queries)
        return queries
        
    async def run_research(self, query: str) -> Dict:
        """
        用 GPT-Researcher 运行单个查询。

        Args:
            query: 要研究的搜索查询

        Returns:
            包含研究结果与上下文的字典
        """
        researcher = GPTResearcher(
            query=query,
            report_type=ReportType.ResearchReport.value,
            report_format="markdown",
            report_source=ReportSource.Web.value,
            tone=Tone.Objective,
            verbose=True
        )
        
        # 运行研究并获取结果
        research_result = await researcher.conduct_research()
        report = await researcher.write_report()
        
        return {
            "query": query,
            "report": report,
            "context": research_result,
        }
        
    def evaluate_research(
        self,
        research_data: Dict,
        output_dir: Optional[str] = None
    ) -> Dict:
        """
        评估研究结果是否存在幻觉。

        Args:
            research_data: 包含研究结果与上下文的字典
            output_dir: 可选，保存评估结果的目录

        Returns:
            包含评估结果的字典
        """
        # 未提供时则使用默认输出目录
        if output_dir is None:
            output_dir = DEFAULT_OUTPUT_DIR
            
        # 用最终合并后的上下文作为源文本
        source_text = research_data.get("context", "")
        
        if not source_text:
            logger.warning("No source text found in research results - skipping evaluation")
            eval_result = {
                "input": research_data["query"],
                "output": research_data["report"],
                "source": "No source text available",
                "is_hallucination": None,
                "confidence_score": None,
                "reasoning": "Evaluation skipped - no source text available for verification"
            }
        else:
            # 评估研究报告是否存在幻觉
            eval_result = self.hallucination_evaluator.evaluate_response(
                model_output=research_data["report"],
                source_text=source_text
            )
        
        # 保存到输出目录
        os.makedirs(output_dir, exist_ok=True)

        # 追加到评估记录
        records_file = Path(output_dir) / "evaluation_records.jsonl"
        with open(records_file, "a") as f:
            f.write(json.dumps(eval_result) + "\n")
            
        return eval_result

async def main(num_queries: int = 5, output_dir: str = DEFAULT_OUTPUT_DIR):
    """
    在有抽样的一批查询上运行评估。

    Args:
        num_queries: 要评估的查询数量
        output_dir: 保存结果的目录
    """
    evaluator = ResearchEvaluator()

    # 加载并抽取查询
    queries = evaluator.load_queries(num_queries)
    logger.info(f"Selected {len(queries)} queries for evaluation")
    
    # 对每个查询运行研究与评估
    all_results = []
    total_hallucinated = 0
    total_responses = 0
    total_evaluated = 0
    
    for query in queries:
        try:
            logger.info(f"Processing query: {query}")
            
            # 运行研究
            research_data = await evaluator.run_research(query)

            # 评估结果
            eval_results = evaluator.evaluate_research(
                research_data,
                output_dir=output_dir
            )
            
            all_results.append(eval_results)
            
            # 更新计数器
            total_responses += 1
            if eval_results["is_hallucination"] is not None:
                total_evaluated += 1
                if eval_results["is_hallucination"]:
                    total_hallucinated += 1
                    
        except Exception as e:
            logger.error(f"Error processing query '{query}': {str(e)}")
            continue
            
    # 计算幻觉率
    hallucination_rate = (total_hallucinated / total_evaluated) if total_evaluated > 0 else None
    
    # 保存汇总结果
    aggregate_results = {
        "total_queries": len(queries),
        "successful_queries": len(all_results),
        "total_responses": total_responses,
        "total_evaluated": total_evaluated,
        "total_hallucinated": total_hallucinated,
        "hallucination_rate": hallucination_rate,
        "results": all_results
    }
    
    aggregate_file = Path(output_dir) / "aggregate_results.json"
    with open(aggregate_file, "w") as f:
        json.dump(aggregate_results, f, indent=2)
    logger.info(f"Saved aggregate results to {aggregate_file}")
    
    # 打印汇总
    print("\n=== Evaluation Summary ===")
    print(f"Queries processed: {len(queries)}")
    print(f"Responses evaluated: {total_evaluated}")
    print(f"Responses skipped (no source text): {total_responses - total_evaluated}")
    if hallucination_rate is not None:
        print(f"Hallucination rate: {hallucination_rate * 100:.1f}%")
    else:
        print("No responses could be evaluated due to missing source text")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run GPT-Researcher evaluation")
    parser.add_argument("-n", "--num-queries", type=int, default=5,
                      help="Number of queries to evaluate")
    parser.add_argument("-o", "--output-dir", type=str, default=DEFAULT_OUTPUT_DIR,
                      help="Directory to save results")
    args = parser.parse_args()
    
    asyncio.run(main(args.num_queries, args.output_dir)) 