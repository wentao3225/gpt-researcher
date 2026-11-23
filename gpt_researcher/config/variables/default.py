from .base import BaseConfig

DEFAULT_CONFIG: BaseConfig = {
    "RETRIEVER": "tavily",
    "EMBEDDING": "openai:text-embedding-3-small",
    "SIMILARITY_THRESHOLD": 0.42,
    "CONTEXT_FILTER": "auto",  # auto | jev | keyword | embeddings | none；auto = 有 TYPESAFE_API_KEY 时用 jev，否则用 keyword
    "FAST_LLM": "openai:gpt-5.4-mini",
    "SMART_LLM": "openai:gpt-6-sol",  # 支持长回复（2k+ 字）。
    "STRATEGIC_LLM": "openai:gpt-5.4",  # 用于规划的推理模型；可通过 REASONING_EFFORT 调节速度与深度的平衡。
    # 输出 token 上限。对推理模型（默认的 gpt-5.x 系列）而言，
    # 这些值映射到 max_completion_tokens，其中也包含推理 token ——
    # 因此在可见输出之上预留了较大的余量。
    "FAST_TOKEN_LIMIT": 6000,
    "SMART_TOKEN_LIMIT": 12000,
    "STRATEGIC_TOKEN_LIMIT": 8000,
    "BROWSE_CHUNK_MAX_LENGTH": 8192,
    "CURATE_SOURCES": False,
    "SUMMARY_TOKEN_LIMIT": 700,
    "TEMPERATURE": 0.4,
    "USER_AGENT": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36 Edg/119.0.0.0",
    "MAX_SEARCH_RESULTS_PER_QUERY": 5,
    "MEMORY_BACKEND": "local",
    "TOTAL_WORDS": 1200,
    "REPORT_FORMAT": "APA",
    "MAX_ITERATIONS": 3,
    "AGENT_ROLE": None,
    "SCRAPER": "bs",
    "MAX_SCRAPER_WORKERS": 15,
    "SCRAPER_RATE_LIMIT_DELAY": 0.0,  # 抓取器请求之间的最小间隔秒数（0 = 不限制，适用于 API 有速率限制的场景）
    "MAX_SUBTOPICS": 3,
    "LANGUAGE": "english",
    "REPORT_SOURCE": "web",
    "DOC_PATH": "./my-docs",
    "PROMPT_FAMILY": "default",
    "LLM_KWARGS": {},
    "EMBEDDING_KWARGS": {},
    "VERBOSE": False,
    # 深度研究专用设置
    "DEEP_RESEARCH_BREADTH": 3,
    "DEEP_RESEARCH_DEPTH": 2,
    "DEEP_RESEARCH_CONCURRENCY": 4,
    
    # MCP 检索器专用设置
    "MCP_SERVERS": [],  # 预定义的 MCP 服务器配置列表
    "MCP_AUTO_TOOL_SELECTION": True,  # 是否为查询自动选择最佳工具
    "MCP_ALLOWED_ROOT_PATHS": [],  # 允许本地文件访问的根路径列表
    "MCP_STRATEGY": "fast",  # MCP 执行策略："fast"、"deep"、"disabled"
    "REASONING_EFFORT": "medium",
    
    # 图片生成设置（可选 - 需要 GOOGLE_API_KEY）
    # 免费层模型：gemini-2.5-flash-image
    # 付费层模型：imagen-4.0-generate-001、imagen-4.0-fast-generate-001
    "IMAGE_GENERATION_MODEL": "models/gemini-2.5-flash-image",
    "IMAGE_GENERATION_MAX_IMAGES": 3,  # 每份报告可生成的最大图片数量
    "IMAGE_GENERATION_ENABLED": False,  # 内联图片生成的总开关
    "IMAGE_GENERATION_STYLE": "dark",  # 图片风格："dark"（匹配应用主题）、"light" 或 "auto"
    "IMAGE_GENERATION_PROVIDER": "google",  # 图片提供者："google" 或 "modelslab"
}
