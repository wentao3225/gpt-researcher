import asyncio
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from .rate_limiter import get_global_rate_limiter


class WorkerPool:
    def __init__(self, max_workers: int, rate_limit_delay: float = 0.0):
        """
        初始化带并发限制和速率限制的 WorkerPool。

        参数：
            max_workers: 最大并发工作进程数
            rate_limit_delay: 全局请求之间的最小间隔秒数（0 = 不限制）
                             此延迟在所有 WorkerPool 之间强制执行，防止
                             压垮有速率限制的 API。
                             例如：10 请求/分钟（Firecrawl 免费层）用 6.0

        注意：
            rate_limit_delay 通过单例速率限制器全局强制执行。
            这意味着如果有多个 GPTResearcher 实例（例如深度研究中），
            它们将共享同一个速率限制，防止 API 过载。
        """
        self.max_workers = max_workers
        self.rate_limit_delay = rate_limit_delay
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.semaphore = asyncio.Semaphore(max_workers)

        # 配置全局速率限制器
        # 所有 WorkerPool 共享同一个速率限制器实例
        global_limiter = get_global_rate_limiter()
        global_limiter.configure(rate_limit_delay)

    @asynccontextmanager
    async def throttle(self):
        """
        结合并发限制和全局速率限制来节流请求。

        - 信号量控制本池内的并发操作数（同时多少个）
        - 全局速率限制器控制跨所有池的请求频率（全局时序）

        这确保即使有多个并发的 GPTResearcher 实例
        （例如深度研究中），总请求速率也保持在限制之内。
        """
        async with self.semaphore:
            # 使用全局速率限制器（所有 WorkerPool 共享）
            global_limiter = get_global_rate_limiter()
            await global_limiter.wait_if_needed()
            yield
