import logging
import json
import os
from datetime import datetime
from pathlib import Path

class JSONResearchHandler:
    def __init__(self, json_file):
        self.json_file = json_file
        self.research_data = {
            "timestamp": datetime.now().isoformat(),
            "events": [],
            "content": {
                "query": "",
                "sources": [],
                "context": [],
                "report": "",
                "costs": 0.0
            }
        }

    def log_event(self, event_type: str, data: dict):
        self.research_data["events"].append({
            "timestamp": datetime.now().isoformat(),
            "type": event_type,
            "data": data
        })
        self._save_json()

    def update_content(self, key: str, value):
        self.research_data["content"][key] = value
        self._save_json()

    def _save_json(self):
        with open(self.json_file, 'w') as f:
            json.dump(self.research_data, f, indent=2)

def setup_research_logging():
    # 如果日志目录不存在则创建
    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)
    
    # 为日志文件生成时间戳
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 创建日志文件路径
    log_file = logs_dir / f"research_{timestamp}.log"
    json_file = logs_dir / f"research_{timestamp}.json"
    
    # 为研究日志配置文件处理器
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    
    # 获取研究日志器并配置
    research_logger = logging.getLogger('research')
    research_logger.setLevel(logging.INFO)
    
    # 清除已有处理器，避免重复
    research_logger.handlers.clear()
    
    # 添加文件处理器
    research_logger.addHandler(file_handler)
    
    # 添加控制台输出的流处理器
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    research_logger.addHandler(console_handler)
    
    # 阻止向根日志器传播，避免重复日志
    research_logger.propagate = False
    
    # 创建 JSON 处理器
    json_handler = JSONResearchHandler(json_file)
    
    return str(log_file), str(json_file), research_logger, json_handler

def get_research_logger():
    return logging.getLogger('research')

def get_json_handler():
    return getattr(logging.getLogger('research'), 'json_handler', None)
