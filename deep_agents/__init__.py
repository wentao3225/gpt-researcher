"""Deep Agents x GPT Researcher 示例。

在 LangChain Deep Agent 框架内把 GPT Researcher 作为研究引擎运行：
主智能体用 todo 规划、把章节委托给带隔离上下文的 researcher
子智能体、把草稿卸载到文件系统，并组装一份最终带引用的报告。
"""
