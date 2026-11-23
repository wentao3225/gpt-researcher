"""GPT Researcher 配置的枚举类型。"""

from enum import Enum


class ReportType(Enum):
    """研究输出可用报告类型的枚举。

    定义 GPT Researcher 智能体可以生成的各种报告类型。

    属性：
        ResearchReport: 包含全面分析的标准研究报告。
        ResourceReport: 侧重于列出和描述资源的报告。
        OutlineReport: 提供主题结构化大纲的报告。
        CustomReport: 用户自定义的报告格式。
        DetailedReport: 深入的详细分析报告。
        SubtopicReport: 聚焦于特定子主题的报告。
        DeepResearch: 具有广泛分析的深度研究模式。
    """
    ResearchReport = "research_report"
    ResourceReport = "resource_report"
    OutlineReport = "outline_report"
    CustomReport = "custom_report"
    DetailedReport = "detailed_report"
    SubtopicReport = "subtopic_report"
    DeepResearch = "deep"


class ReportSource(Enum):
    """研究可用数据来源的枚举。

    定义研究员在生成报告时可以从哪些来源收集信息。

    属性：
        Web: 从网络搜索并抓取内容。
        Local: 使用本地文档和文件。
        Azure: 使用 Azure blob 存储中的文档。
        LangChainDocuments: 使用 LangChain 文档对象。
        LangChainVectorStore: 使用 LangChain 向量库进行检索。
        Static: 使用预定义的静态内容。
        Hybrid: 组合多种来源类型。
    """
    Web = "web"
    Local = "local"
    Azure = "azure"
    LangChainDocuments = "langchain_documents"
    LangChainVectorStore = "langchain_vectorstore"
    Static = "static"
    Hybrid = "hybrid"


class Tone(Enum):
    """报告可用写作语气的枚举。

    定义生成研究报告时可以使用的各种语气，
    以匹配期望的风格和受众。

    每个语气值都包含对其写作风格的描述。
    """
    Objective = "Objective (impartial and unbiased presentation of facts and findings)"
    Formal = "Formal (adheres to academic standards with sophisticated language and structure)"
    Analytical = (
        "Analytical (critical evaluation and detailed examination of data and theories)"
    )
    Persuasive = (
        "Persuasive (convincing the audience of a particular viewpoint or argument)"
    )
    Informative = (
        "Informative (providing clear and comprehensive information on a topic)"
    )
    Explanatory = "Explanatory (clarifying complex concepts and processes)"
    Descriptive = (
        "Descriptive (detailed depiction of phenomena, experiments, or case studies)"
    )
    Critical = "Critical (judging the validity and relevance of the research and its conclusions)"
    Comparative = "Comparative (juxtaposing different theories, data, or methods to highlight differences and similarities)"
    Speculative = "Speculative (exploring hypotheses and potential implications or future research directions)"
    Reflective = "Reflective (considering the research process and personal insights or experiences)"
    Narrative = (
        "Narrative (telling a story to illustrate research findings or methodologies)"
    )
    Humorous = "Humorous (light-hearted and engaging, usually to make the content more relatable)"
    Optimistic = "Optimistic (highlighting positive findings and potential benefits)"
    Pessimistic = (
        "Pessimistic (focusing on limitations, challenges, or negative outcomes)"
    )
    Simple = "Simple (written for young readers, using basic vocabulary and clear explanations)"
    Casual = "Casual (conversational and relaxed style for easy, everyday reading)"


class PromptFamily(Enum):
    """按名称支持的提示词家族"""
    Default = "default"
    Granite = "granite"
    Granite3 = "granite3"
    Granite31 = "granite3.1"
    Granite32 = "granite3.2"
    Granite33 = "granite3.3"
