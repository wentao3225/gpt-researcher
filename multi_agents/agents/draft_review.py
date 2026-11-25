DEFAULT_MAX_DRAFT_REVISIONS = 3


class MaxDraftRevisionsExceededError(RuntimeError):
    """当审阅者/修订者轮次超过配置的草稿限制时抛出。"""


def route_draft_review(draft, max_draft_revisions=DEFAULT_MAX_DRAFT_REVISIONS):
    """为编辑审阅回路返回 accept | revise。

    * ``review is None`` → accept（包括调用方清空 review 后的硬上限强制接受）
    * ``max_draft_revisions is None`` → 永不强制接受（运维人员选择退出）
    * ``draft_revision_count > max`` → 抛出，使边可以优雅地强制接受
      而不是撞上 LangGraph 的 recursion_limit
    """
    if draft.get("review") is None:
        return "accept"

    if max_draft_revisions is None:
        return "revise"

    count = draft.get("draft_revision_count", 0)
    if count > max_draft_revisions:
        raise MaxDraftRevisionsExceededError(
            "Draft revision limit exceeded. "
            f"Received {count} revision rounds; "
            f"max_draft_revisions is {max_draft_revisions}."
        )

    return "revise"
