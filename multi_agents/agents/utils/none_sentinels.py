"""对 LLM / 人工 "none" / "no" 接受哨兵值的严格解析器。

子串检查（``"None" in response``、``"no" in feedback``）会误接受
类似 "None of the criteria are met" 的审阅意见或人工计划反馈
"not enough cost data"。保持哨兵值精确（可选的周围引号/空白），
使真实的批评意见绝不被当作批准。
"""

from __future__ import annotations


def is_none_accept_response(response: str | None) -> bool:
    """仅当 *response* 恰好是接受哨兵 ``None`` 时返回 True。

    接受可选的周围空白和一对匹配的 ``'`` / ``"`` 引号（模型常给
    令牌加引号）。空 / ``None`` 输入不被视为接受 —— 意指
    "无审阅" 的调用方应显式传递该状态，而不是依赖空字符串。
    """
    if response is None:
        return False
    if not isinstance(response, str):
        response = str(response)
    text = response.strip()
    if len(text) >= 2 and text[0] == text[-1] and text[0] in "\"'":
        text = text[1:-1].strip()
    return text.lower() == "none"


def is_human_plan_approval(feedback: str | None) -> bool:
    """仅当 *feedback* 恰好是取消/批准哨兵 ``no`` 时返回 True。

    人工反馈提示词在计划无需更改时请用户回复 ``no``。匹配
    整个（规范化后的）字符串避免丢弃仅包含字母 ``no`` 的
    真实修订请求（如 "not enough on evaluation"、"novel methods"）。
    """
    if feedback is None:
        return False
    if not isinstance(feedback, str):
        feedback = str(feedback)
    return feedback.strip().lower() == "no"
