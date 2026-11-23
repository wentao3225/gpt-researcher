"""网页抓取的工具函数。

此模块提供从网页提取内容、图片和处理 HTML 的辅助函数。
"""

import hashlib
import logging
import re
from urllib.parse import parse_qs, urljoin, urlparse

import bs4
from bs4 import BeautifulSoup


def get_relevant_images(soup: BeautifulSoup, url: str) -> list:
    """从页面提取相关图片"""
    image_urls = []
    
    try:
        if soup is None:
            return []
        # 查找所有带 src 属性的 img 标签
        all_images = soup.find_all('img', src=True)
        
        for img in all_images:
            img_src = urljoin(url, img['src'])
            if img_src.startswith(('http://', 'https://')):
                score = 0
                # 检查相关类名
                classes = img.get("class") or []
                if isinstance(classes, str):
                    classes = classes.split()
                elif not isinstance(classes, (list, tuple, set)):
                    classes = []
                if any(cls in classes for cls in ['header', 'featured', 'hero', 'thumbnail', 'main', 'content']):
                    score = 4  # 更高分数
                # 检查尺寸属性
                elif img.get('width') and img.get('height'):
                    width = parse_dimension(img['width'])
                    height = parse_dimension(img['height'])
                    if width and height:
                        if width >= 2000 and height >= 1000:
                            score = 3  # 中等分数（超大图片）
                        elif width >= 1600 or height >= 800:
                            score = 2  # 较低分数
                        elif width >= 800 or height >= 500:
                            score = 1  # 最低分数
                        elif width >= 500 or height >= 300:
                            score = 0  # 最低分数
                        else:
                            continue  # 跳过小图片
                
                image_urls.append({'url': img_src, 'score': score})
        
        # 按分数排序图片（最高在前）
        sorted_images = sorted(image_urls, key=lambda x: x['score'], reverse=True)
        
        return sorted_images[:10]  # 确保总共返回不超过 10 张图片
    
    except Exception as e:
        logging.error(f"Error in get_relevant_images: {e}")
        return []

def parse_dimension(value: str) -> int:
    """解析尺寸值，处理 px 单位"""
    # HTML 的 width/height 属性经常缺失或非字符串；调用方传入的
    # img.get('width') 可能为 None。
    if value is None:
        return None
    if not isinstance(value, str):
        try:
            return int(float(value))
        except (ValueError, TypeError) as e:
            logging.debug("Could not parse dimension value %r: %s", value, e)
            return None
    if value.lower().endswith('px'):
        value = value[:-2]  # 去掉 'px' 后缀
    try:
        # 先转为浮点数以处理 '409.12' 这类小数值
        return int(float(value))
    except (ValueError, TypeError) as e:
        # 非数字尺寸（如 '100%'、'auto'、'50em'）在真实页面上很常见；
        # 以 debug 级别记录而不是刷屏 stdout。
        logging.debug("Could not parse dimension value %r: %s", value, e)
        return None

def extract_title(soup: BeautifulSoup) -> str:
    """从 BeautifulSoup 对象提取标题文本。

    始终返回字符串。空的 ``<title></title>`` 产生 ``""``（而非
    ``None``），含嵌套标记的标题（如 ``<title><span>x</span></title>``）
    产生其文本内容而非原始内部 HTML。
    """
    title_tag = soup.title
    if not title_tag:
        return ""
    return title_tag.get_text(strip=True)

def get_image_hash(image_url: str) -> str:
    """基于图片文件名和关键查询参数计算简单哈希。

    始终返回字符串。出错时返回空字符串而非 None，使把结果用作
    字典键或字符串拼接的调用方不会因 TypeError 崩溃。
    """
    try:
        parsed_url = urlparse(image_url)

        # 提取文件名
        filename = parsed_url.path.split('/')[-1]

        # 提取关键查询参数（如 CDN 提供图片的 'url'）
        query_params = parse_qs(parsed_url.query)
        essential_params = query_params.get('url', [])

        # 合并文件名和关键参数
        image_identifier = filename + ''.join(essential_params)

        # 计算哈希
        return hashlib.md5(image_identifier.encode()).hexdigest()
    except Exception as e:
        logging.error(f"Error calculating image hash for {image_url}: {e}")
        return ""


def clean_soup(soup: BeautifulSoup) -> BeautifulSoup:
    """通过移除不需要的标签来清理 soup"""
    for tag in soup.find_all(
        [
            "script",
            "style",
            "footer",
            "header",
            "nav",
            "menu",
            "sidebar",
            "svg",
        ]
    ):
        tag.decompose()

    disallowed_class_set = {"nav", "menu", "sidebar", "footer"}

    # 清理带特定类名的标签
    def does_tag_have_disallowed_class(elem) -> bool:
        if not isinstance(elem, bs4.Tag):
            return False

        return any(
            cls_name in disallowed_class_set for cls_name in elem.get("class", [])
        )

    for tag in soup.find_all(does_tag_have_disallowed_class):
        tag.decompose()

    return soup


# 浏览器按块布局的元素。每个元素都开启提取文本的新一行，
# 而其他元素的文本保持在围绕它的行内。
_BLOCK_TAGS = frozenset(
    "address article aside blockquote body caption center dd details dialog dir div dl dt"
    " fieldset figcaption figure footer form h1 h2 h3 h4 h5 h6 head header hgroup hr html"
    " legend li listing main menu nav ol optgroup option p plaintext pre search section"
    " summary table tbody td textarea tfoot th thead title tr ul xmp".split()
)
# 这些元素内部的空白是内容，因此其文本按原样保留。
_PREFORMATTED_TAGS = frozenset(("listing", "plaintext", "pre", "textarea", "xmp"))
# 文本从不与相邻文本连在一起的行内盒子。
_BOX_TAGS = frozenset(("button", "select"))
# HTML 中只有这些字符是空白。不换行空格是内容。
_HTML_WHITESPACE = re.compile(r"[ \t\n\r\f]+")


def _block_lines(soup: BeautifulSoup) -> list[str]:
    """以每个块元素一行的方式返回 soup 的文本。

    ``soup.get_text(separator="\\n")`` 把每个文本节点放在单独一行，
    因此一个链接或强调会拆开它所在的句子，甚至单词。
    """
    lines: list[str] = []
    line: list[str] = []

    def end_line() -> None:
        text = _HTML_WHITESPACE.sub(" ", "".join(line)).strip(" ")
        line.clear()
        if text.strip():
            lines.append(text)

    # 显式栈，使深层嵌套的标记不会耗尽递归限制。
    stack: list[tuple[bs4.PageElement, str]] = [(soup, "visit")]
    while stack:
        node, action = stack.pop()
        if action == "end_line":
            end_line()
        elif action == "space":
            line.append(" ")
        elif isinstance(node, bs4.Tag):
            if node.name == "br":
                end_line()
            elif node.name in _PREFORMATTED_TAGS:
                end_line()
                text = node.get_text().strip("\n")
                if text.strip():
                    lines.append(text)
            else:
                if node.name in _BLOCK_TAGS:
                    end_line()
                    stack.append((node, "end_line"))
                elif node.name in _BOX_TAGS:
                    line.append(" ")
                    stack.append((node, "space"))
                stack.extend((child, "visit") for child in reversed(node.contents))
        # Script、style、template 和 comment 字符串是子类，不是页面文本，
        # 与 get_text() 默认应用的过滤器相同。
        elif type(node) in (bs4.NavigableString, bs4.CData):
            line.append(str(node))
    end_line()
    return lines


def get_text_from_soup(soup: BeautifulSoup) -> str:
    """以改进的过滤从 soup 获取相关文本"""
    if soup is None:
        return ""
    text = "\n".join(_block_lines(soup))
    # 移除多余空白
    text = re.sub(r"\s{2,}", " ", text)
    return text
