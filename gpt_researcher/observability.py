"""可选的、故障隔离的研究生命周期事件。

核心研究员刻意不解释也不持久化这些事件。消费者可以提供
一个可调用对象，或一个暴露 ``on_event`` 方法的对象。
观察者的失败绝不能改变研究结果。
"""

import inspect
import logging
from collections.abc import Callable
from typing import Any, Protocol

logger = logging.getLogger(__name__)


class ResearchObserver(Protocol):
    """:class:`GPTResearcher` 接受的对象式观察者。"""

    def on_event(self, event: str, **fields: Any) -> Any: ...


async def emit_research_event(
    observer: ResearchObserver | Callable[..., Any] | None,
    event: str,
    **fields: Any,
) -> None:
    """通知观察者，且不允许遥测影响研究。"""
    if observer is None:
        return

    try:
        handler = getattr(observer, "on_event", observer)
        result = handler(event, **fields)
        if inspect.isawaitable(result):
            await result
    except Exception as exc:  # noqa: BLE001 - 观察者绝不能影响研究
        logger.debug("Research observer failed for %s: %s", event, type(exc).__name__)
