"""GPT Researcher 的嵌入提供者管理。

此模块提供 Memory 类，处理跨多个提供者（OpenAI、Cohere、Google、
Ollama 等）的嵌入生成。

支持的提供者：
    - openai: OpenAI 嵌入
    - azure_openai: Azure OpenAI 嵌入
    - cohere: Cohere 嵌入
    - google_vertexai: Google Vertex AI 嵌入
    - google_genai: Google Generative AI 嵌入
    - fireworks: Fireworks AI 嵌入
    - ollama: 本地 Ollama 嵌入
    - together: Together AI 嵌入
    - mistralai: Mistral AI 嵌入
    - huggingface: HuggingFace 嵌入
    - nomic: Nomic 嵌入
    - voyageai: Voyage AI 嵌入
    - dashscope: DashScope 嵌入
    - bedrock: AWS Bedrock 嵌入
    - aimlapi: AIML API 嵌入
    - custom: 自定义 OpenAI 兼容 API
"""

import os
from typing import Any

from gpt_researcher.llm_provider.minimax import get_minimax_base_url

OPENAI_EMBEDDING_MODEL = os.environ.get(
    "OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"
)

_SUPPORTED_PROVIDERS = {
    "openai",
    "azure_openai",
    "cohere",
    "gigachat",
    "google_vertexai",
    "google_genai",
    "fireworks",
    "ollama",
    "together",
    "mistralai",
    "huggingface",
    "nomic",
    "voyageai",
    "dashscope",
    "custom",
    "bedrock",
    "aimlapi",
    "netmind",
    "openrouter",
    "minimax",
    "nebius",
}


class Memory:
    """管理用于文档相似度和检索的嵌入生成。

    此类提供统一的接口，使用各种提供者生成嵌入。
    仅在需要时才延迟加载特定提供者的依赖。

    属性：
        _embeddings: 底层的 LangChain 嵌入实例。

    示例：
        ```python
        memory = Memory("openai", "text-embedding-3-small")
        embeddings = memory.get_embeddings()
        ```
    """

    def __init__(self, embedding_provider: str, model: str, **embedding_kwargs: Any):
        """使用指定的嵌入提供者初始化 Memory。

        参数：
            embedding_provider: 要使用的嵌入提供者名称。
                必须是受支持的提供者之一（openai、cohere 等）。
            model: 用于嵌入的模型名称/ID。
            **embedding_kwargs: 传递给嵌入提供者构造函数的
                额外关键字参数。

        抛出：
            Exception: 如果嵌入提供者不受支持。
        """
        _embeddings = None
        match embedding_provider:
            case "custom":
                from langchain_openai import OpenAIEmbeddings

                _embeddings = OpenAIEmbeddings(
                    model=model,
                    openai_api_key=os.getenv("OPENAI_API_KEY", "custom"),
                    openai_api_base=os.getenv(
                        "OPENAI_BASE_URL", "http://localhost:1234/v1"
                    ),  # default for lmstudio
                    check_embedding_ctx_length=False,
                    **embedding_kwargs,
                )  # quick fix for lmstudio
            case "openai":
                from langchain_openai import OpenAIEmbeddings

                # 通过 OPENAI_BASE_URL 支持自定义 OpenAI 兼容 API
                if "openai_api_base" not in embedding_kwargs and os.environ.get("OPENAI_BASE_URL"):
                    embedding_kwargs["openai_api_base"] = os.environ["OPENAI_BASE_URL"]

                _embeddings = OpenAIEmbeddings(model=model, **embedding_kwargs)
            case "azure_openai":
                from langchain_openai import AzureOpenAIEmbeddings

                _embeddings = AzureOpenAIEmbeddings(
                    model=model,
                    azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
                    openai_api_key=os.environ["AZURE_OPENAI_API_KEY"],
                    openai_api_version=os.environ.get(
                        "AZURE_OPENAI_API_VERSION",
                        os.environ.get("OPENAI_API_VERSION"),
                    ),
                    **embedding_kwargs,
                )
            case "cohere":
                from langchain_cohere import CohereEmbeddings

                _embeddings = CohereEmbeddings(model=model, **embedding_kwargs)
            case "google_vertexai":
                from langchain_google_vertexai import VertexAIEmbeddings

                _embeddings = VertexAIEmbeddings(model=model, **embedding_kwargs)
            case "google_genai":
                from langchain_google_genai import GoogleGenerativeAIEmbeddings

                _embeddings = GoogleGenerativeAIEmbeddings(
                    model=model, **embedding_kwargs
                )
            case "fireworks":
                from langchain_fireworks import FireworksEmbeddings

                _embeddings = FireworksEmbeddings(model=model, **embedding_kwargs)
            case "gigachat":
                from langchain_gigachat import GigaChatEmbeddings

                _embeddings = GigaChatEmbeddings(model=model, **embedding_kwargs)
            case "ollama":
                from langchain_ollama import OllamaEmbeddings

                _embeddings = OllamaEmbeddings(
                    model=model,
                    base_url=os.environ["OLLAMA_BASE_URL"],
                    **embedding_kwargs,
                )
            case "together":
                from langchain_together import TogetherEmbeddings

                _embeddings = TogetherEmbeddings(model=model, **embedding_kwargs)
            case "netmind":
                from langchain_netmind import NetmindEmbeddings

                _embeddings = NetmindEmbeddings(model=model, **embedding_kwargs)
            case "mistralai":
                from langchain_mistralai import MistralAIEmbeddings

                _embeddings = MistralAIEmbeddings(model=model, **embedding_kwargs)
            case "huggingface":
                from langchain_huggingface import HuggingFaceEmbeddings

                _embeddings = HuggingFaceEmbeddings(model_name=model, **embedding_kwargs)
            case "nomic":
                from langchain_nomic import NomicEmbeddings

                _embeddings = NomicEmbeddings(model=model, **embedding_kwargs)
            case "voyageai":
                from langchain_voyageai import VoyageAIEmbeddings

                _embeddings = VoyageAIEmbeddings(
                    voyage_api_key=os.environ["VOYAGE_API_KEY"],
                    model=model,
                    **embedding_kwargs,
                )
            case "dashscope":
                from langchain_community.embeddings import DashScopeEmbeddings

                _embeddings = DashScopeEmbeddings(model=model, **embedding_kwargs)
            case "bedrock":
                from langchain_aws.embeddings import BedrockEmbeddings

                _embeddings = BedrockEmbeddings(model_id=model, **embedding_kwargs)
            case "aimlapi":
                from langchain_openai import OpenAIEmbeddings

                _embeddings = OpenAIEmbeddings(
                    model=model,
                    openai_api_key=os.getenv("AIMLAPI_API_KEY"),
                    openai_api_base=os.getenv("AIMLAPI_BASE_URL", "https://api.aimlapi.com/v1"),
                    **embedding_kwargs,
                )
            case "openrouter":
                from langchain_openai import OpenAIEmbeddings

                _embeddings = OpenAIEmbeddings(
                    model=model,
                    openai_api_key=os.getenv("OPENROUTER_API_KEY"),
                    openai_api_base="https://openrouter.ai/api/v1",
                    **embedding_kwargs,
                )
            case "minimax":
                from langchain_openai import OpenAIEmbeddings

                _embeddings = OpenAIEmbeddings(
                    model=model,
                    openai_api_key=os.getenv("MINIMAX_API_KEY"),
                    openai_api_base=get_minimax_base_url("openai"),
                    **embedding_kwargs,
                )
            case "nebius":
                from langchain_openai import OpenAIEmbeddings

                _embeddings = OpenAIEmbeddings(
                    model=model,
                    openai_api_key=os.getenv("NEBIUS_API_KEY"),
                    openai_api_base=os.getenv("NEBIUS_BASE_URL", "https://api.tokenfactory.nebius.com/v1"),
                    **embedding_kwargs,
                )
            case _:
                raise Exception("Embedding not found.")

        self._embeddings = _embeddings

    def get_embeddings(self):
        """获取已配置的嵌入实例。

        返回：
            为此 Memory 配置的 LangChain 嵌入实例。
        """
        return self._embeddings
