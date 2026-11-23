"""
使用 judges 库评估模型输出是否存在幻觉。
"""
import logging
from pathlib import Path
from typing import Dict, List, Optional

from dotenv import load_dotenv
from judges.classifiers.hallucination import HaluEvalDocumentSummaryNonFactual

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class HallucinationEvaluator:
    """使用 judges 库评估模型输出是否存在幻觉。"""

    def __init__(self, model: str = "openai/gpt-4o"):
        """初始化幻觉评估器。"""
            
        self.summary_judge = HaluEvalDocumentSummaryNonFactual(model=model)
        
    def evaluate_response(self, model_output: str, source_text: str) -> Dict:
        """
        对照源文档，评估单条模型响应是否存在幻觉。

        Args:
            model_output: 待评估的模型响应
            source_text: 用于核对摘要的源文本

        Returns:
            包含评估结果的字典
        """
        try:
            # 使用文档摘要评估
            judgment = self.summary_judge.judge(
                input=source_text,  # 源文档
                output=model_output  # 待评估的摘要
            )
                
            return {
                "output": model_output,
                "source": source_text,
                "is_hallucination": judgment.score,
                "reasoning": judgment.reasoning
            }
            
        except Exception as e:
            logger.error(f"Error evaluating response: {str(e)}")
            raise

def main():
    # 示例测试用例
    model_output = "The capital of France is Paris, a city known for its rich history and culture."
    source_text = "Paris is the capital and largest city of France, located in the northern part of the country."
    
    evaluator = HallucinationEvaluator()
    result = evaluator.evaluate_response(
        model_output=model_output,
        source_text=source_text
    )
    
    # 打印结果
    print("\nEvaluation Results:")
    print(f"Output: {result['output']}")
    print(f"Source: {result['source']}")
    print(f"Hallucination: {'Yes' if result['is_hallucination'] else 'No'}")
    print(f"Reasoning: {result['reasoning']}")

if __name__ == "__main__":
    main() 