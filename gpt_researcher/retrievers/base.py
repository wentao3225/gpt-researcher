"""检索器的共享契约。

检索器分为两类，区别很重要：

* 大多数返回**仍需抓取的 URL** —— 搜索 API 返回链接加简短摘要，
  真正的页面文本稍后抓取；
* 少数返回**已经抓取的内容** —— PubMed Central 返回完整文章文本，
  自定义检索器的文档化契约是 ``list[{url, raw_content}]``。

历史上没有任何地方记录检索器属于哪类，因此
``_search_relevant_source_urls`` 从任何 ``raw_content`` 字段的长度
推断::

    if url and raw_content and len(raw_content) > 100:

只要摘要碰巧超过阈值，这个猜测就会出错：结果被当作已抓取，
其 URL 永远不会被抓取，报告最终没有可验证的引用
（#1846、#1892）。当时的变通办法是在各个检索器内部限制摘要长度，
使其保持在 100 字符以内 —— 为了满足启发式规则而逐个检索器调参。

``requires_scraping`` 用声明取代猜测。默认为 ``True``，
这是绝大多数检索器的正确答案。

声明是可选的。未定义 ``requires_scraping`` 的检索器 —— 包括任何
第三方或用户自定义的 —— 完全保留旧的基于长度的行为，
因此本仓库之外的代码无需更改。
"""
from abc import ABC, abstractmethod
from typing import Any, Dict, List


class BaseRetriever(ABC):
    """记录检索器契约的可选基类。

    不要求继承；``_search_relevant_source_urls`` 只查找
    ``requires_scraping`` 属性。继承只是让声明显式化，
    并给新检索器一个阅读契约的地方。
    """

    #: 结果是仍需抓取的 URL（``True``，默认），还是检索器已取回的
    #: 内容（``False``）。
    #:
    #: 答案固定时重写为普通类属性；取决于检索器配置时重写为
    #: ``property``。
    requires_scraping: bool = True

    @abstractmethod
    def search(self, max_results: int = 7) -> List[Dict[str, Any]]:
        """返回搜索结果。

        ``requires_scraping = True`` 时，每项应在 ``url`` 或 ``href``
        下携带 URL；任何 ``body``/``snippet`` 都被视为预览，
        页面会被抓取以获得真实内容。

        ``requires_scraping = False`` 时，每项应携带 URL 以及
        ``raw_content`` 下已取回的文本；不执行抓取。
        """
        raise NotImplementedError
