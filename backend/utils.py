import aiofiles
import urllib
import mistune
import os

async def write_to_file(filename: str, text: str) -> None:
    """以 UTF-8 编码异步将文本写入文件。

    参数：
        filename (str): 要写入的文件名。
        text (str): 要写入的文本。
    """
    # 确保 text 是字符串
    if not isinstance(text, str):
        text = str(text)

    # 将文本转换为 UTF-8，替换任何有问题的字符
    text_utf8 = text.encode('utf-8', errors='replace').decode('utf-8')

    async with aiofiles.open(filename, "w", encoding='utf-8') as file:
        await file.write(text_utf8)

async def write_text_to_md(text: str, filename: str = "") -> str:
    """将文本写入 Markdown 文件并返回文件路径。

    参数：
        text (str): 要写入 Markdown 文件的文本。

    返回：
        str: 生成的 Markdown 文件的路径。
    """
    import uuid

    safe_name = (filename or "").strip()[:60] or f"report-{uuid.uuid4().hex[:12]}"
    safe_name = safe_name.replace("/", "-").replace("\\", "-")
    os.makedirs("outputs", exist_ok=True)
    file_path = f"outputs/{safe_name}.md"
    await write_to_file(file_path, text)
    return urllib.parse.quote(file_path)

def _preprocess_images_for_pdf(text: str) -> str:
    """将网页图片 URL 转换为绝对文件路径以用于 PDF 生成。
    
    把 /outputs/images/... URL 转换为 weasyprint 可解析的
    绝对 file:// 路径。
    """
    import re
    
    base_path = os.path.abspath(".")
    
    # 查找带 /outputs/ URL 的 markdown 图片的模式
    def replace_image_url(match):
        alt_text = match.group(1)
        url = match.group(2)
        
        # 将 /outputs/... 转换为绝对路径
        if url.startswith("/outputs/"):
            abs_path = os.path.join(base_path, url.lstrip("/"))
            return f"![{alt_text}]({abs_path})"
        return match.group(0)
    
    # 匹配 ![alt text](/outputs/images/...)
    pattern = r'!\[([^\]]*)\]\((/outputs/[^)]+)\)'
    return re.sub(pattern, replace_image_url, text)


async def write_md_to_pdf(text: str, filename: str = "") -> str:
    """将 Markdown 文本转换为 PDF 文件并返回文件路径。

    参数：
        text (str): 要转换的 Markdown 文本。

    返回：
        str: 生成的 PDF 的编码后文件路径。
    """
    import uuid

    # 以前空/纯空白文件名会写出 "outputs/.pdf"，让下载 UI 困惑
    #（#1718）。优先使用稳定的非空基础名。
    safe_name = (filename or "").strip()[:60] or f"report-{uuid.uuid4().hex[:12]}"
    # 替换路径分隔符，使 PDF 保持在 outputs/ 下。
    safe_name = safe_name.replace("/", "-").replace("\\", "-")
    os.makedirs("outputs", exist_ok=True)
    file_path = f"outputs/{safe_name}.pdf"

    try:
        # 相对于此后端模块解析 css 路径，避免依赖当前工作目录。
        current_dir = os.path.dirname(os.path.abspath(__file__))
        css_path = os.path.join(current_dir, "styles", "pdf_styles.css")
        
        # 预处理图片 URL 以兼容 PDF
        processed_text = _preprocess_images_for_pdf(text)
        
        # 将 base_url 设为当前目录，以解析任何剩余的相对路径
        base_url = os.path.abspath(".")
        from md2pdf.core import md2pdf
        md2pdf(
               file_path,
               raw=processed_text,
               css=css_path,
               base_url=base_url,
            )
        print(f"Report written to {file_path}")
    except Exception as e:
        print(f"Error in converting Markdown to PDF: {e}")
        return ""

    encoded_file_path = urllib.parse.quote(file_path)
    return encoded_file_path

async def write_md_to_word(text: str, filename: str = "") -> str:
    """将 Markdown 文本转换为 DOCX 文件并返回文件路径。

    参数：
        text (str): 要转换的 Markdown 文本。

    返回：
        str: 生成的 DOCX 的编码后文件路径。
    """
    import uuid

    safe_name = (filename or "").strip()[:60] or f"report-{uuid.uuid4().hex[:12]}"
    safe_name = safe_name.replace("/", "-").replace("\\", "-")
    os.makedirs("outputs", exist_ok=True)
    file_path = f"outputs/{safe_name}.docx"

    try:
        from docx import Document
        from htmldocx import HtmlToDocx
        # 将报告 markdown 转换为 HTML
        html = mistune.html(text)
        # 创建文档对象
        doc = Document()
        # 将报告生成的 html 转换为文档格式
        HtmlToDocx().add_html_to_document(html, doc)

        # 将 docx 文档保存到 file_path
        doc.save(file_path)

        print(f"Report written to {file_path}")

        encoded_file_path = urllib.parse.quote(file_path)
        return encoded_file_path

    except Exception as e:
        print(f"Error in converting Markdown to DOCX: {e}")
        return ""