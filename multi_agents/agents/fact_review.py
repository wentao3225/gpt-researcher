DEFAULT_MAX_FACT_CHECK_REVISIONS = 3


class MaxFactCheckRevisionsExceededError(RuntimeError):
    """当撰写者/事实核查器轮次超过配置的限制时抛出。"""


def route_fact_check(state, max_fact_check_revisions=DEFAULT_MAX_FACT_CHECK_REVISIONS):
    if state.get("fact_check_notes") is None:
        return "accept"

    if max_fact_check_revisions is None:
        return "revise"

    count = state.get("fact_check_revision_count", 0)
    if count > max_fact_check_revisions:
        raise MaxFactCheckRevisionsExceededError(
            "Fact-check revision limit exceeded. "
            f"Received {count} revision rounds; "
            f"max_fact_check_revisions is {max_fact_check_revisions}."
        )

    return "revise"
