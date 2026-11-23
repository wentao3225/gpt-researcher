import logging
import sys
from copy import copy
from typing import Literal

import click

TRACE_LOG_LEVEL = 5


def get_formatted_logger():
    """返回一个带格式的日志器。"""
    logger = logging.getLogger("scraper")
    # 设置日志级别
    logger.setLevel(logging.INFO)

    # 检查日志器是否已有处理器，避免重复
    if not logger.handlers:
        # 创建处理器
        handler = logging.StreamHandler()

        # 使用 DefaultFormatter 创建格式化器
        formatter = DefaultFormatter(
            "%(levelprefix)s [%(asctime)s] %(message)s",
            datefmt="%H:%M:%S"
        )

        # 为处理器设置格式化器
        handler.setFormatter(formatter)

        # 将处理器添加到日志器
        logger.addHandler(handler)

    # 禁止传播，防止父日志器重复记录
    logger.propagate = False

    return logger


class ColourizedFormatter(logging.Formatter):
    """
    自定义日志格式化器类，功能：

    * 以适当的颜色输出 LOG_LEVEL。
    * 如果日志调用包含 `extras={"color_message": ...}`，则用它
      格式化输出，替代纯文本消息。
    """

    level_name_colors = {
        TRACE_LOG_LEVEL: lambda level_name: click.style(str(level_name), fg="blue"),
        logging.DEBUG: lambda level_name: click.style(str(level_name), fg="cyan"),
        logging.INFO: lambda level_name: click.style(str(level_name), fg="green"),
        logging.WARNING: lambda level_name: click.style(str(level_name), fg="yellow"),
        logging.ERROR: lambda level_name: click.style(str(level_name), fg="red"),
        logging.CRITICAL: lambda level_name: click.style(str(level_name), fg="bright_red"),
    }

    def __init__(
        self,
        fmt: str | None = None,
        datefmt: str | None = None,
        style: Literal["%", "{", "$"] = "%",
        use_colors: bool | None = None,
    ):
        if use_colors in (True, False):
            self.use_colors = use_colors
        else:
            self.use_colors = sys.stdout.isatty()
        super().__init__(fmt=fmt, datefmt=datefmt, style=style)

    def color_level_name(self, level_name: str, level_no: int) -> str:
        def default(level_name: str) -> str:
            return str(level_name)  # pragma: no cover

        func = self.level_name_colors.get(level_no, default)
        return func(level_name)

    def should_use_colors(self) -> bool:
        return True  # pragma: no cover

    def formatMessage(self, record: logging.LogRecord) -> str:
        recordcopy = copy(record)
        levelname = recordcopy.levelname
        seperator = " " * (8 - len(recordcopy.levelname))
        if self.use_colors:
            levelname = self.color_level_name(levelname, recordcopy.levelno)
            if "color_message" in recordcopy.__dict__:
                recordcopy.msg = recordcopy.__dict__["color_message"]
                recordcopy.__dict__["message"] = recordcopy.getMessage()
        recordcopy.__dict__["levelprefix"] = levelname + ":" + seperator
        return super().formatMessage(recordcopy)


class DefaultFormatter(ColourizedFormatter):
    def should_use_colors(self) -> bool:
        return sys.stderr.isatty()  # pragma: no cover
