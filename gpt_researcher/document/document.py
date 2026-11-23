import asyncio
import os
from typing import List, Union
from langchain_community.document_loaders import (
    PyMuPDFLoader,
    TextLoader,
    UnstructuredCSVLoader,
UnstructuredEPubLoader,
    UnstructuredExcelLoader,
    UnstructuredMarkdownLoader,
    UnstructuredPowerPointLoader,
    UnstructuredWordDocumentLoader
)
from langchain_community.document_loaders import BSHTMLLoader


def detect_utf8(file_path: str) -> str | None:
    """如果文件能按 UTF-8 解码则返回 "utf-8-sig"，否则返回 None。

    TextLoader 和 BSHTMLLoader 否则会用区域设置编码打开文件，
    在 Windows 上是 cp1252 之类的代码页，会拒绝或弄乱 UTF-8。
    非 UTF-8 文件保持该默认值；"utf-8-sig" 还会去掉 BOM。
    """
    with open(file_path, "rb") as f:
        try:
            f.read().decode("utf-8")
        except UnicodeDecodeError:
            return None
    return "utf-8-sig"


class DocumentLoader:

    def __init__(self, path: Union[str, List[str]]):
        self.path = path

    async def load(self) -> list:
        tasks = []
        if isinstance(self.path, list):
            for file_path in self.path:
                if os.path.isfile(file_path):  # 确保是有效文件
                    filename = os.path.basename(file_path)
                    file_name, file_extension_with_dot = os.path.splitext(filename)
                    file_extension = file_extension_with_dot.strip(".").lower()
                    tasks.append(self._load_document(file_path, file_extension))
                    
        elif isinstance(self.path, (str, bytes, os.PathLike)):
            for root, dirs, files in os.walk(self.path):
                for file in files:
                    file_path = os.path.join(root, file)
                    file_name, file_extension_with_dot = os.path.splitext(file)
                    file_extension = file_extension_with_dot.strip(".").lower()
                    tasks.append(self._load_document(file_path, file_extension))
                    
        else:
            raise ValueError("Invalid type for path. Expected str, bytes, os.PathLike, or list thereof.")

        # for root, dirs, files in os.walk(self.path):
        #     for file in files:
        #         file_path = os.path.join(root, file)
        #         file_name, file_extension_with_dot = os.path.splitext(file_path)
        #         file_extension = file_extension_with_dot.strip(".")
        #         tasks.append(self._load_document(file_path, file_extension))

        docs = []
        for pages in await asyncio.gather(*tasks):
            for page in pages:
                if page.page_content:
                    # 加载器偶尔会省略 metadata["source"]（自定义加载器、
                    # 某些 HTML 分区）。优先用 source，然后回退到路径。
                    meta = getattr(page, "metadata", None) or {}
                    source = meta.get("source") or meta.get("file_path") or ""
                    url = os.path.basename(source) if source else ""
                    docs.append({
                        "raw_content": page.page_content,
                        "url": url,
                    })
                    
        if not docs:
            raise ValueError("🤷 Failed to load any documents!")

        return docs

    async def _load_document(self, file_path: str, file_extension: str) -> list:
        ret_data = []
        try:
            encoding = detect_utf8(file_path) if file_extension in ("txt", "html", "htm") else None
            loader_dict = {
                "pdf": PyMuPDFLoader(file_path),
"epub": UnstructuredEPubLoader(file_path),
                "txt": TextLoader(file_path, encoding=encoding),
                "doc": UnstructuredWordDocumentLoader(file_path),
                "docx": UnstructuredWordDocumentLoader(file_path),
                "pptx": UnstructuredPowerPointLoader(file_path),
                "csv": UnstructuredCSVLoader(file_path, mode="elements"),
                "xls": UnstructuredExcelLoader(file_path, mode="elements"),
                "xlsx": UnstructuredExcelLoader(file_path, mode="elements"),
                "md": UnstructuredMarkdownLoader(file_path),
                "html": BSHTMLLoader(file_path, open_encoding=encoding),
                "htm": BSHTMLLoader(file_path, open_encoding=encoding)
            }

            loader = loader_dict.get(file_extension, None)
            if loader:
                try:
                    ret_data = loader.load()
                except Exception as e:
                    print(
                        f"Failed to load {file_extension or 'unknown'} document: {file_path}"
                    )
                    print(e)

        except Exception as e:
            print(f"Failed to load document : {file_path}")
            print(e)

        return ret_data
