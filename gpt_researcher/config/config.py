"""GPT Researcher 配置管理。

此模块提供 Config 类，管理 GPT Researcher 的所有配置项，
包括 LLM 提供者、嵌入、检索器以及各类运行参数。
"""

import json
import json_repair
import os
import warnings
from copy import deepcopy
from typing import Any, Dict, List, Type, Union, get_args, get_origin

from gpt_researcher.llm_provider.generic.base import ReasoningEfforts

from .variables.base import BaseConfig
from .variables.default import DEFAULT_CONFIG


class Config:
    """GPT Researcher 配置管理器。

    负责从文件、环境变量和默认值中加载、解析和管理所有配置项。

    属性：
        CONFIG_DIR: 存放配置文件的目录。
        config_path: 配置文件路径。
        llm_kwargs: LLM 的额外关键字参数。
        embedding_kwargs: 嵌入的额外关键字参数。
    """

    CONFIG_DIR = os.path.join(os.path.dirname(__file__), "variables")

    def __init__(self, config_path: str | None = None):
        """初始化配置类。

        参数：
            config_path: 可选的 JSON 配置文件路径。
        """
        self.config_path = config_path
        self.llm_kwargs: Dict[str, Any] = {}
        self.embedding_kwargs: Dict[str, Any] = {}

        config_to_use = self.load_config(config_path)
        self._set_attributes(config_to_use)
        self._set_embedding_attributes()
        self._set_llm_attributes()
        self._handle_deprecated_attributes()
        if (os.getenv('REPORT_SOURCE') or config_to_use['REPORT_SOURCE']) != 'web':
          self._set_doc_path(config_to_use)

    def _set_attributes(self, config: Dict[str, Any]) -> None:
        """从配置字典设置配置属性。

        将环境变量与配置文件的值合并，环境变量优先。

        参数：
            config: 配置键值对字典。
        """
        for key, value in config.items():
            env_value = os.getenv(key)
            if env_value is not None:
                value = self.convert_env_value(key, env_value, BaseConfig.__annotations__[key])
            setattr(self, key.lower(), value)

        # 处理 RETRIEVER 的默认值
        retriever_env = os.environ.get("RETRIEVER", config.get("RETRIEVER", "tavily"))
        try:
            self.retrievers = self.parse_retrievers(retriever_env)
        except ValueError as e:
            print(f"Warning: {str(e)}. Defaulting to 'tavily' retriever.")
            self.retrievers = ["tavily"]

    def _set_embedding_attributes(self) -> None:
        """解析并设置嵌入提供者和模型属性。"""
        self.embedding_provider, self.embedding_model = self.parse_embedding(
            self.embedding
        )

    def _set_llm_attributes(self) -> None:
        """为所有 LLM 类型解析并设置提供者和模型属性。"""
        self.fast_llm_provider, self.fast_llm_model = self.parse_llm(self.fast_llm)
        self.smart_llm_provider, self.smart_llm_model = self.parse_llm(self.smart_llm)
        self.strategic_llm_provider, self.strategic_llm_model = self.parse_llm(self.strategic_llm)
        self.reasoning_effort = self.parse_reasoning_effort(self.reasoning_effort)

    def _handle_deprecated_attributes(self) -> None:
        """处理已弃用的配置属性并发出警告。"""
        if os.getenv("EMBEDDING_PROVIDER") is not None:
            warnings.warn(
                "EMBEDDING_PROVIDER is deprecated and will be removed soon. Use EMBEDDING instead.",
                FutureWarning,
                stacklevel=2,
            )
            self.embedding_provider = (
                os.environ["EMBEDDING_PROVIDER"] or self.embedding_provider
            )

            embedding_provider = os.environ["EMBEDDING_PROVIDER"]
            if embedding_provider == "ollama":
                self.embedding_model = os.environ["OLLAMA_EMBEDDING_MODEL"]
            elif embedding_provider == "custom":
                self.embedding_model = os.getenv("OPENAI_EMBEDDING_MODEL", "custom")
            elif embedding_provider == "openai":
                self.embedding_model = "text-embedding-3-large"
            elif embedding_provider == "azure_openai":
                self.embedding_model = "text-embedding-3-large"
            elif embedding_provider == "huggingface":
                self.embedding_model = "sentence-transformers/all-MiniLM-L6-v2"
            elif embedding_provider == "gigachat":
                self.embedding_model = "Embeddings"
            elif embedding_provider == "google_genai":
                self.embedding_model = "gemini-embedding-2"
            else:
                raise Exception("Embedding provider not found.")

        _deprecation_warning = (
            "LLM_PROVIDER, FAST_LLM_MODEL and SMART_LLM_MODEL are deprecated and "
            "will be removed soon. Use FAST_LLM and SMART_LLM instead."
        )
        if os.getenv("LLM_PROVIDER") is not None:
            warnings.warn(_deprecation_warning, FutureWarning, stacklevel=2)
            self.fast_llm_provider = (
                os.environ["LLM_PROVIDER"] or self.fast_llm_provider
            )
            self.smart_llm_provider = (
                os.environ["LLM_PROVIDER"] or self.smart_llm_provider
            )
        if os.getenv("FAST_LLM_MODEL") is not None:
            warnings.warn(_deprecation_warning, FutureWarning, stacklevel=2)
            self.fast_llm_model = os.environ["FAST_LLM_MODEL"] or self.fast_llm_model
        if os.getenv("SMART_LLM_MODEL") is not None:
            warnings.warn(_deprecation_warning, FutureWarning, stacklevel=2)
            self.smart_llm_model = os.environ["SMART_LLM_MODEL"] or self.smart_llm_model

    def _set_doc_path(self, config: Dict[str, Any]) -> None:
        if not self.doc_path:
            self.doc_path = config['DOC_PATH']
        if self.doc_path:
            try:
                self.validate_doc_path()
            except Exception as e:
                print(f"Warning: Error validating doc_path: {str(e)}. Using default doc_path.")
                self.doc_path = DEFAULT_CONFIG['DOC_PATH']

    @classmethod
    def load_config(cls, config_path: str | None) -> Dict[str, Any]:
        """按名称加载配置。

        总是返回全新副本，因此 ``LLM_KWARGS``、``MCP_SERVERS`` 等可变默认值
        不会在多个 ``Config`` 实例之间共享。
        """
        config_path = config_path or os.environ.get("CONFIG_PATH")
        if not config_path:
            return deepcopy(DEFAULT_CONFIG)

        # config_path = os.path.join(cls.CONFIG_DIR, config_path)
        if not os.path.exists(config_path):
            if config_path and config_path != "default":
                print(f"Warning: Configuration not found at '{config_path}'. Using default configuration.")
                if not config_path.endswith(".json"):
                    print(f"Do you mean '{config_path}.json'?")
            return deepcopy(DEFAULT_CONFIG)

        with open(config_path, "r", encoding="utf-8") as f:
            custom_config = json.load(f)

        # 与默认配置合并，确保所有键都存在
        merged_config = deepcopy(DEFAULT_CONFIG)
        merged_config.update(custom_config)
        return merged_config

    @classmethod
    def list_available_configs(cls) -> List[str]:
        """列出所有可用的配置名称。"""
        configs = ["default"]
        for file in os.listdir(cls.CONFIG_DIR):
            if file.endswith(".json"):
                configs.append(file[:-5])  # 去掉 .json 扩展名
        return configs

    def parse_retrievers(self, retriever_str: str) -> List[str]:
        """将检索器字符串解析为列表并校验其有效性。"""
        from ..retrievers.utils import get_all_retriever_names
        
        retrievers = [retriever.strip()
                      for retriever in retriever_str.split(",")]
        valid_retrievers = get_all_retriever_names() or []
        invalid_retrievers = [r for r in retrievers if r not in valid_retrievers]
        if invalid_retrievers:
            raise ValueError(
                f"Invalid retriever(s) found: {', '.join(invalid_retrievers)}. "
                f"Valid options are: {', '.join(valid_retrievers)}."
            )
        return retrievers

    @staticmethod
    def parse_llm(llm_str: str | None) -> tuple[str | None, str | None]:
        """将 LLM 字符串解析为 (llm_provider, llm_model)。"""
        from gpt_researcher.llm_provider.generic.base import _SUPPORTED_PROVIDERS

        if llm_str is None:
            return None, None
        try:
            llm_provider, llm_model = llm_str.split(":", 1)
            assert llm_provider in _SUPPORTED_PROVIDERS, (
                f"Unsupported {llm_provider}.\nSupported llm providers are: "
                + ", ".join(_SUPPORTED_PROVIDERS)
            )
            return llm_provider, llm_model
        except ValueError:
            raise ValueError(
                "Set SMART_LLM or FAST_LLM = '<llm_provider>:<llm_model>' "
                "Eg 'openai:gpt-4o-mini'"
            )

    @staticmethod
    def parse_reasoning_effort(reasoning_effort_str: str | None) -> str | None:
        """将推理力度字符串解析为 (reasoning_effort)。"""
        if reasoning_effort_str is None:
            return ReasoningEfforts.Medium.value
        if reasoning_effort_str not in [effort.value for effort in ReasoningEfforts]:
            raise ValueError(f"Invalid reasoning effort: {reasoning_effort_str}. Valid options are: {', '.join([effort.value for effort in ReasoningEfforts])}")
        return reasoning_effort_str

    @staticmethod
    def parse_embedding(embedding_str: str | None) -> tuple[str | None, str | None]:
        """将嵌入字符串解析为 (embedding_provider, embedding_model)。"""
        from gpt_researcher.memory.embeddings import _SUPPORTED_PROVIDERS

        if embedding_str is None:
            return None, None
        try:
            embedding_provider, embedding_model = embedding_str.split(":", 1)
            assert embedding_provider in _SUPPORTED_PROVIDERS, (
                f"Unsupported {embedding_provider}.\nSupported embedding providers are: "
                + ", ".join(_SUPPORTED_PROVIDERS)
            )
            return embedding_provider, embedding_model
        except ValueError:
            raise ValueError(
                "Set EMBEDDING = '<embedding_provider>:<embedding_model>' "
                "Eg 'openai:text-embedding-3-large'"
            )

    def validate_doc_path(self):
        """确保文档路径对应的文件夹存在"""
        os.makedirs(self.doc_path, exist_ok=True)

    @staticmethod
    def convert_env_value(key: str, env_value: str, type_hint: Type) -> Any:
        """根据类型提示将环境变量转换为相应的类型。"""
        origin = get_origin(type_hint)
        args = get_args(type_hint)

        if origin is Union:
            # 处理 Union 类型（如 Union[str, None] / Optional[str]）。
            # 先检查 None 哨兵值再检查非 None 参数：对 Optional[str] 而言，
            # str 转换永远不会抛异常，因此先按 str 处理会永久遮蔽
            # none/null/"" → None 的分支（见 issue #1899）。
            if type(None) in args and env_value.lower() in ("none", "null", ""):
                return None
            for arg in args:
                if arg is type(None):
                    continue
                try:
                    return Config.convert_env_value(key, env_value, arg)
                except ValueError:
                    continue
            raise ValueError(f"Cannot convert {env_value} to any of {args}")

        if type_hint is bool:
            return env_value.lower() in ("true", "1", "yes", "on")
        elif type_hint is int:
            return int(env_value)
        elif type_hint is float:
            return float(env_value)
        elif type_hint in (str, Any):
            return env_value
        elif type_hint is list or origin is list or origin is List:
            # 环境变量值通常是手工编辑的（尾随逗号、单引号等）。
            # 裸 `list` 的 get_origin 为 None；typing.List[...] 的 origin 为 list。
            try:
                value = json_repair.loads(env_value)
            except Exception as exc:
                raise ValueError(f"Cannot convert {env_value} to list") from exc
            if not isinstance(value, list):
                raise ValueError(f"Cannot convert {env_value} to list")
            return value
        elif type_hint is dict:
            try:
                value = json_repair.loads(env_value)
            except Exception as exc:
                raise ValueError(f"Cannot convert {env_value} to dict") from exc
            if not isinstance(value, dict):
                raise ValueError(f"Cannot convert {env_value} to dict")
            return value
        else:
            raise ValueError(f"Unsupported type {type_hint} for key {key}")


    def set_verbose(self, verbose: bool) -> None:
        """设置日志详细程度。"""
        self.llm_kwargs["verbose"] = verbose

    def get_mcp_server_config(self, name: str) -> dict:
        """
        获取指定 MCP 服务器的配置。
        
        参数：
            name (str): 要获取配置的 MCP 服务器名称。
                
        返回：
            dict: 服务器配置；如果未找到该服务器则返回空字典。
        """
        if not name or not self.mcp_servers:
            return {}
        
        for server in self.mcp_servers:
            if isinstance(server, dict) and server.get("name") == name:
                return server
            
        return {}
