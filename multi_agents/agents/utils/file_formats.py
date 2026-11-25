import aiofiles
import urllib
import uuid
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

async def write_text_to_md(text: str, path: str) -> str:
    """将文本写入 Markdown 文件并返回文件路径。

    参数：
        text (str): 要写入 Markdown 文件的文本。

    返回：
        str: 生成的 Markdown 文件的路径。
    """
    task = uuid.uuid4().hex
    file_path = f"{path}/{task}.md"
    await write_to_file(file_path, text)
    print(f"Report written to {file_path}")
    return file_path


async def write_md_to_pdf(text: str, path: str) -> str:
    """将 Markdown 文本转换为 PDF 文件并返回文件路径。

    参数：
        text (str): 要转换的 Markdown 文本。

    返回：
        str: 生成的 PDF 的编码后文件路径。
    """
    task = uuid.uuid4().hex
    file_path = f"{path}/{task}.pdf"

    try:
        # 获取当前文件所在目录
        current_dir = os.path.dirname(os.path.abspath(__file__))
        css_path = os.path.join(current_dir, "pdf_styles.css")
        
        # Moved imports to inner function to avoid known import errors with gobject-2.0
        from md2pdf.core import md2pdf
        md2pdf(file_path,
               md_content=text,
               css_file_path=css_path,
               base_url=None)
        print(f"Report written to {file_path}")
    except Exception as e:
        print(f"Error in converting Markdown to PDF: {e}")
        return ""

    encoded_file_path = urllib.parse.quote(file_path)
    return encoded_file_path


async def write_md_to_word(text: str, path: str) -> str:
    """将 Markdown 文本转换为 DOCX 文件并返回文件路径。

    参数：
        text (str): 要转换的 Markdown 文本。

    返回：
        str: 生成的 DOCX 的编码后文件路径。
    """
    task = uuid.uuid4().hex
    file_path = f"{path}/{task}.docx"

    try:
        from htmldocx import HtmlToDocx
        from docx import Document
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
