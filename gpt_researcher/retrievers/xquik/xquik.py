# Xquik X/Twitter 检索器
#
# 在 X（Twitter）上搜索实时观点、开发者讨论、产品反馈、突发新闻
# 和专家意见。
# 每条推文 $0.00015 —— 比官方 X API 便宜 33 倍。

import json
import os
import urllib.parse
import urllib.request


class XquikSearch:
    """
    Xquik X/Twitter 搜索检索器。

    通过 Xquik REST API 搜索推文，并以所有 GPT Researcher 检索器
    使用的标准 {title, href, body} 格式返回结果。

    在环境中设置 XQUIK_API_KEY。可在 https://xquik.com 获取密钥。
    """

    def __init__(self, query, query_domains=None, **kwargs):
        self.query = query
        self.query_domains = query_domains
        self.api_key = self.get_api_key()

    def get_api_key(self):
        try:
            api_key = os.environ["XQUIK_API_KEY"]
        except KeyError:
            raise Exception(
                "Xquik API key not found. Please set the XQUIK_API_KEY "
                "environment variable. Get a key at https://xquik.com"
            )
        return api_key

    def search(self, max_results=10):
        """
        通过 Xquik API 搜索 X/Twitter。

        返回：
            list: 搜索结果 [{title, href, body}, ...]
        """
        print(f"Searching X/Twitter with query: {self.query}...")

        try:
            results = self._search_tweets(max_results)
            return results
        except Exception as e:
            print(f"Error: {e}. Failed fetching X/Twitter sources. Resulting in empty response.")
            return []

    def _search_tweets(self, max_results):
        params = urllib.parse.urlencode({
            "q": self.query,
            "limit": min(max_results, 200),
            "queryType": "Top",
        })
        url = f"https://xquik.com/api/v1/x/tweets/search?{params}"

        req = urllib.request.Request(url, headers={
            "X-API-Key": self.api_key,
            "Accept": "application/json",
            "User-Agent": "gpt-researcher/1.0",
        })

        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read().decode("utf-8"))

        # 使用 `or` 回退，使显式的 JSON null（"tweets": null、
        # "author": null、"text": null）不会绕过 `.get()` 的默认值，
        # 导致下方切片/属性访问崩溃 —— 那会被 search() 的宽泛 except
        # 吞掉，静默丢弃所有结果。与同目录的 GetXAPI 检索器一致。
        # 同时拒绝非列表的 `tweets` 和非字典的推文行：API 可能返回
        # 字符串/对象信封或稀疏元组，前者会按字符迭代，后者会在 .get
        # 上抛出 AttributeError —— 同样是静默丢弃。
        tweets = data.get("tweets") or []
        if not isinstance(tweets, list):
            return []
        search_results = []

        for tweet in tweets[:max_results]:
            if not isinstance(tweet, dict):
                continue
            author = tweet.get("author") or {}
            username = author.get("username") or "unknown"
            text = tweet.get("text") or ""
            tweet_id = tweet.get("id") or ""

            likes = tweet.get("likeCount", 0)
            retweets = tweet.get("retweetCount", 0)
            views = tweet.get("viewCount", 0)

            engagement = f"{likes} likes, {retweets} RTs"
            if views:
                engagement += f", {views} views"

            search_results.append({
                "title": f"@{username}: {text[:120]}{'...' if len(text) > 120 else ''}",
                "href": f"https://x.com/{username}/status/{tweet_id}",
                "body": f"{text}\n\n[{engagement}]",
            })

        return search_results
