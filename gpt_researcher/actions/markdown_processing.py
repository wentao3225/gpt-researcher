import re
from html import unescape
import markdown
from typing import List, Dict

def extract_headers(markdown_text: str) -> List[Dict]:
    """
    从 Markdown 文本中提取标题。

    参数：
        markdown_text (str): 要处理的 Markdown 文本。

    返回：
        List[Dict]: 表示标题结构的字典列表。
    """
    if not isinstance(markdown_text, str) or not markdown_text:
        return []
    headers = []
    parsed_md = markdown.markdown(markdown_text)
    lines = parsed_md.split("\n")

    stack = []
    for line in lines:
        if line.startswith("<h") and len(line) > 2 and line[2].isdigit():
            level = int(line[2])
            header_text = line[line.index(">") + 1 : line.rindex("<")]

            while stack and stack[-1]["level"] >= level:
                stack.pop()

            header = {
                "level": level,
                "text": header_text,
            }
            if stack:
                stack[-1].setdefault("children", []).append(header)
            else:
                headers.append(header)

            stack.append(header)

    return headers

def extract_sections(markdown_text: str) -> List[Dict[str, str]]:
    """
    从子主题报告中提取所有已写的章节。

    参数：
        markdown_text (str): 子主题报告文本。

    返回：
        List[Dict[str, str]]: 章节列表，每个章节是包含
        'section_title' 和 'written_content' 的字典。
    """
    if not isinstance(markdown_text, str) or not markdown_text:
        return []
    sections = []
    parsed_md = markdown.markdown(markdown_text)
    
    pattern = r'<h\d>(.*?)</h\d>(.*?)(?=<h\d>|$)'
    matches = re.findall(pattern, parsed_md, re.DOTALL)
    
    for title, content in matches:
        # 在去除标签之后再解码，使 &lt;tag&gt; 这类转义代码
        # 保持为可见文本，而不会被误认为 HTML 标记。
        clean_content = unescape(re.sub(r'<.*?>', '', content)).strip()
        if clean_content:
            sections.append({
                "section_title": unescape(re.sub(r'<.*?>', '', title)).strip(),
                "written_content": clean_content
            })
    
    return sections

def table_of_contents(markdown_text: str) -> str:
    """
    为给定的 Markdown 文本生成目录。

    参数：
        markdown_text (str): 要处理的 Markdown 文本。

    返回：
        str: 生成的目录。
    """
    if not isinstance(markdown_text, str):
        return ""

    def generate_table_of_contents(headers, indent_level=0):
        toc = ""
        for header in headers:
            toc += " " * (indent_level * 4) + "- " + header["text"] + "\n"
            if "children" in header:
                toc += generate_table_of_contents(header["children"], indent_level + 1)
        return toc

    try:
        headers = extract_headers(markdown_text)
        toc = "## Table of Contents\n\n" + generate_table_of_contents(headers)
        return toc
    except Exception as e:
        print("table_of_contents Exception : ", e)
        return markdown_text

def add_references(report_markdown: str, visited_urls: set) -> str:
    """
    为 Markdown 报告添加参考文献。

    参数：
        report_markdown (str): 已有的 Markdown 报告。
        visited_urls (set): 研究期间访问过的 URL 集合。

    返回：
        str: 添加了参考文献的更新版 Markdown 报告。
    """
    if not isinstance(report_markdown, str):
        report_markdown = "" if report_markdown is None else str(report_markdown)
    if visited_urls is None:
        return report_markdown
    try:
        url_markdown = "\n\n\n## References\n\n"
        # 排序以保证参考文献列表是确定性的。``visited_urls`` 是集合，
        # 其迭代顺序每次运行（乃至跨进程）都不同，这曾导致相同输入下
        # 最终报告的 References 部分不可复现。
        url_markdown += "".join(f"- [{url}]({url})\n" for url in sorted(visited_urls))
        updated_markdown_report = report_markdown + url_markdown
        return updated_markdown_report
    except Exception as e:
        print(f"Encountered exception in adding source urls : {e}")
        return report_markdown
