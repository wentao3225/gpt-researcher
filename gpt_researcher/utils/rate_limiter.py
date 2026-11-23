"""
抓取器请求的全局速率限制器。

确保 SCRAPER_RATE_LIMIT_DELAY 在所有 WorkerPool 之间全局生效，
而不只是每个池各自生效。防止多个并发研究员压垮
Firecrawl 等有速率限制的 API。
"""
import asyncio
import time
from typing import ClassVar


class GlobalRateLimiter:
    """
    单例全局速率限制器。

    确保整个应用中任何抓取器请求之间都有最小延迟，
    无论有多少个 WorkerPool 或 GPTResearcher 实例处于活动状态。
    """

    _instance: ClassVar['GlobalRateLimiter'] = None
    _lock: ClassVar[asyncio.Lock] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        """初始化全局速率限制器（仅一次）。"""
        if self._initialized:
            return

        self.last_request_time = 0.0
        self.rate_limit_delay = 0.0
        self._initialized = True

        # 在类级别创建锁，确保所有实例共享
        if GlobalRateLimiter._lock is None:
            # 注意：首次在异步上下文中访问时才会正确初始化
            GlobalRateLimiter._lock = None

    @classmethod
    def get_lock(cls):
        """获取或创建异步锁（必须在异步上下文中调用）。"""
        if cls._lock is None:
            cls._lock = asyncio.Lock()
        return cls._lock

    def configure(self, rate_limit_delay: float):
        """
        配置全局速率限制延迟。

        参数：
            rate_limit_delay: 请求之间的最小间隔秒数（0 = 不限制）
        """
        # 环境/配置可能传入字符串；wait_if_needed 按浮点数比较。
        if rate_limit_delay is None:
            self.rate_limit_delay = 0.0
            return
        try:
            delay = float(rate_limit_delay)
        except (TypeError, ValueError) as e:
            raise ValueError(
                f"rate_limit_delay must be a number, got {rate_limit_delay!r}"
            ) from e
        if delay < 0:
            raise ValueError(f"rate_limit_delay must be non-negative, got {delay}")
        self.rate_limit_delay = delay

    async def wait_if_needed(self):
        """
        在需要时等待，以强制执行全局速率限制。

        此方法确保无论有多少个 WorkerPool 处于活动状态，
        SCRAPER_RATE_LIMIT_DELAY 都在全局得到遵守。
        """
        if self.rate_limit_delay <= 0:
            return  # 不限速

        lock = self.get_lock()
        async with lock:
            current_time = time.time()
            time_since_last = current_time - self.last_request_time

            if time_since_last < self.rate_limit_delay:
                sleep_time = self.rate_limit_delay - time_since_last
                await asyncio.sleep(sleep_time)

            self.last_request_time = time.time()

    def reset(self):
        """重置速率限制器状态（测试时有用）。"""
        self.last_request_time = 0.0


# 单例实例
_global_rate_limiter = GlobalRateLimiter()


def get_global_rate_limiter() -> GlobalRateLimiter:
    """获取全局速率限制器单例实例。"""
    return _global_rate_limiter
