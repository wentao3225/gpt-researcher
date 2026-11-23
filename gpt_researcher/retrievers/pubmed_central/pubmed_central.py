from typing import List, Dict, Any, Optional
import os
import xml.etree.ElementTree as ET
import requests


# 自成一块的 JATS 元素。其他元素（如 <italic>、<sub> 或 <xref>）
# 内部的文本属于其所在的单词或句子。
_BLOCK_TAGS = {
    "p", "sec", "title", "label", "caption", "list-item", "fig", "table-wrap",
    "tr", "th", "td", "disp-formula", "disp-quote", "def-item", "term", "def",
}


def _element_text(element: Optional[ET.Element]) -> str:
    """`element` 的文本，每个块一行，并保留其内联标记。

    用空格连接 `itertext()` 会在内联标记处拆分单词（`CO<sub>2</sub>`
    读作 `CO 2`），而什么都不加又会把段落连在一起。
    """
    if element is None:
        return ""
    parts: List[str] = []
    # 用栈而非递归遍历，使深层嵌套的文档不会耗尽解释器的递归限制
    stack: List[tuple] = [("start", element)]
    while stack:
        step, node = stack.pop()
        if step == "start":
            if node.tag in _BLOCK_TAGS:
                parts.append("\n")
            parts.append(node.text or "")
            stack.append(("end", node))
            for child in reversed(list(node)):
                stack.append(("tail", child))
                stack.append(("start", child))
        elif step == "end":
            if node.tag in _BLOCK_TAGS:
                parts.append("\n")
        else:
            parts.append(node.tail or "")
    lines = (" ".join(line.split()) for line in "".join(parts).split("\n"))
    return "\n".join(line for line in lines if line)


class PubMedCentralSearch:
    """
    PubMed Central 全文搜索
    """

    # PubMed Central 内联返回完整文章文本，因此无需再抓取。
    requires_scraping = False

    def __init__(self, query: str, query_domains=None):
        self.base_search_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
        self.base_fetch_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
        self.base_link_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/elink.fcgi"
        
        # 从环境获取 API 密钥
        self.api_key = os.getenv('NCBI_API_KEY')
        if not self.api_key:
            print("Warning: NCBI_API_KEY not set. Requests will be rate-limited.")
        
        self.query = query
        self.db_type = os.getenv('PUBMED_DB', 'pmc')  # 全文默认使用 PMC
        
        # 来自环境的可选参数
        self.params = self._populate_params()

    def _populate_params(self) -> Dict[str, Any]:
        """
        从以 'PUBMED_ARG_' 为前缀的环境变量填充参数
        """
        params = {
            key[len('PUBMED_ARG_'):].lower(): value
            for key, value in os.environ.items()
            if key.startswith('PUBMED_ARG_')
        }
        
        # 未提供时设置默认值
        params.setdefault('sort', 'relevance')
        params.setdefault('retmode', 'json')
        return params

    def _search_articles(self, max_results: int) -> Optional[List[str]]:
        """
        根据查询搜索文章 ID
        """
        # 构建带全文过滤器的搜索查询
        if self.db_type == 'pubmed':
            search_term = f"{self.query} AND (ffrft[filter] OR pmc[filter])"
        else:  # PMC 总是有全文
            search_term = self.query
        
        search_params = {
            "db": self.db_type,
            "term": search_term,
            "retmax": max_results,
            "api_key": self.api_key,
            **self.params  # 包含自定义参数
        }
        
        try:
            response = requests.get(self.base_search_url, params=search_params, timeout=20)
            response.raise_for_status()
            data = response.json()
            if not isinstance(data, dict):
                return []

            # 错误/意外的信封可能把 esearchresult 设为 null、列表，
            # 或省略 idlist。只有真正的列表才视为成功。
            esearch = data.get("esearchresult")
            if not isinstance(esearch, dict):
                return []
            id_list = esearch.get("idlist") or []
            if not isinstance(id_list, list):
                return []
            print(f"Found {len(id_list)} articles with full text available")
            return id_list
            
        except requests.RequestException as e:
            print(f"Failed to search articles: {e}")
            return None

    def _pmid_to_pmcid(self, pmid: str) -> Optional[str]:
        """将 PubMed ID 解析为 EFetch 使用的数字 PMC ID。"""
        link_params = {
            "dbfrom": "pubmed",
            "db": "pmc",
            "linkname": "pubmed_pmc",
            "id": pmid,
            "retmode": "xml",
            "api_key": self.api_key,
        }
        try:
            response = requests.get(self.base_link_url, params=link_params, timeout=20)
            response.raise_for_status()
            root = ET.fromstring(response.text)
            if root.tag == "ERROR" or root.find(".//ERROR") is not None:
                return None
            for link_set in root.findall(".//LinkSetDb"):
                if link_set.findtext("LinkName") != "pubmed_pmc":
                    continue
                pmc_id = link_set.findtext("Link/Id")
                if pmc_id and pmc_id.strip().isdigit():
                    return pmc_id.strip()
        except (requests.RequestException, ET.ParseError):
            return None
        return None

    def _fetch_full_text(self, article_id: str) -> Optional[Dict[str, str]]:
        """
        获取单篇文章的全文内容
        """
        if self.db_type == "pubmed":
            article_id = self._pmid_to_pmcid(article_id)
            if article_id is None:
                return None

        fetch_params = {
            "db": "pmc",  # 全文始终从 PMC 获取
            "id": article_id,
            "rettype": "full",
            "retmode": "xml",
            "api_key": self.api_key
        }
        
        try:
            response = requests.get(self.base_fetch_url, params=fetch_params, timeout=20)
            response.raise_for_status()
            
            # 解析 XML 内容
            try:
                root = ET.fromstring(response.text)
                if root.tag == "ERROR" or root.find(".//ERROR") is not None:
                    return None
                article = root if root.tag == "article" else root.find(".//article")
                if article is None:
                    return None

                # 错误信封和仅元数据的响应不是全文。
                body_text = _element_text(article.find('.//body'))
                if not body_text:
                    return None

                # 提取标题和摘要，包括嵌套的格式化标签
                title_text = _element_text(article.find('.//article-title'))
                abstract_text = _element_text(article.find('.//abstract'))
                
                # 合并所有文本内容
                full_content = f"Title: {title_text}\n\nAbstract: {abstract_text}\n\nBody: {body_text}"
                
                # 构建 URL
                if self.db_type == "pmc" or article_id.startswith("PMC"):
                    url = f"https://www.ncbi.nlm.nih.gov/pmc/articles/{article_id}/"
                else:
                    url = f"https://www.ncbi.nlm.nih.gov/pmc/articles/PMC{article_id}/"
                
                return {
                    "href": url,
                    "url": url,
                    "body": full_content,
                    "raw_content": full_content,
                    "title": title_text
                }
                
            except ET.ParseError as e:
                return None
                
        except requests.RequestException as e:
            return None

    def search(self, max_results: int = 5) -> List[Dict[str, Any]]:
        """
        执行搜索并获取全文内容。

        :param max_results: 返回结果的最大数量
        :return: JSON 响应，格式如下：
            [
              {
                "url": "https://www.ncbi.nlm.nih.gov/pmc/articles/PMC1234567/",
                "raw_content": "文章的全文内容..."
              },
              ...
            ]
        """
        # 步骤 1：搜索文章 ID。始终返回列表（绝不返回 None）：
        # actions.query_processing.get_search_results 等调用方的类型是
        # -> List[Dict]，而 skills.researcher 会执行 len(search_results)，
        # 对 None 会抛出 TypeError。
        article_ids = self._search_articles(max_results)
        if not article_ids:
            return []
        
        # 步骤 2：为每篇文章获取全文
        results = []
        for article_id in article_ids:
            article_content = self._fetch_full_text(article_id)
            if article_content:
                results.append(article_content)
        
        return results
