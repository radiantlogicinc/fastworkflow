"""Bound model-facing memory summaries without mutating their source evidence."""
import json


def _size(value):
    # ASCII serialization is conservative even for heavily escaped Unicode.
    return len(json.dumps(value, ensure_ascii=True).encode("utf-8"))


def _excerpt(text, budget):
    if _size(text) <= budget:
        return text
    marker = "\n[... omitted from memory-summary input; full evidence retained in traces ...]\n"
    lo, hi = 0, len(text)
    while lo < hi:
        count = (lo + hi + 1) // 2
        head = (count + 1) // 2
        tail = count // 2
        candidate = text[:head] + marker + (text[-tail:] if tail else "")
        if _size(candidate) <= budget:
            lo = count
        else:
            hi = count - 1
    head, tail = (lo + 1) // 2, lo // 2
    return text[:head] + marker + (text[-tail:] if tail else "")


def bounded_summary_inputs(user_query, workflow_actions, final_agent_response):
    """At most 64KB of serialized values, leaving prompt/output headroom.

    Small turns are unchanged. For large turns the request and answer get
    separate quotas, so a large action log cannot crowd either out. The excerpt
    is explicitly incomplete and used only to summarize conversation memory.
    """
    inputs = dict(user_query=user_query, workflow_actions=workflow_actions,
                  final_agent_response=final_agent_response)
    if _size(inputs) <= 64000:
        return inputs
    return dict(
        user_query=_excerpt(user_query, 8000),
        workflow_actions=[{"history_excerpt": _excerpt(
            json.dumps(workflow_actions, ensure_ascii=True), 28000)}],
        final_agent_response=_excerpt(final_agent_response, 24000),
    )
