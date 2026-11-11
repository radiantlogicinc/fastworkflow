"""Parameter extraction: attempt 1 runs at temperature 0.0, retries run at temperature 1.0.

Real dspy objects only: a tiny ``dspy.LM`` subclass records the kwargs of every call.
"""
import litellm  # noqa: F401  (must precede dspy)
import dspy

from fastworkflow.utils.extraction_attempts import FirstAttemptGreedy


class RecordingLM(dspy.LM):
    """A dspy.LM that answers from a script and records the temperature and rollout id of each call."""

    def __init__(self, replies, **kwargs):
        super().__init__("openai/recording", cache=False, **kwargs)
        self.replies = replies
        self.calls = []  # shared with copies

    def copy(self, **kwargs):
        clone = super().copy(**kwargs)
        clone.replies, clone.calls = self.replies, self.calls
        return clone

    def forward(self, prompt=None, messages=None, **kwargs):
        self.calls.append({"temperature": self.kwargs.get("temperature"),
                           "rollout_id": self.kwargs.get("rollout_id")})
        text = self.replies[min(len(self.calls) - 1, len(self.replies) - 1)]
        return _response(text)


def _response(text):
    from types import SimpleNamespace
    msg = SimpleNamespace(content=text, tool_calls=None)
    return SimpleNamespace(choices=[SimpleNamespace(message=msg, finish_reason="stop")],
                           usage={}, model="recording", _hidden_params={})


def _run(replies, temperature=0.2, n=3):
    lm = RecordingLM(replies, temperature=temperature)
    module = dspy.Predict("question -> answer")
    best = FirstAttemptGreedy(
        module=module, N=n, threshold=1.0,
        reward_fn=lambda args, pred: 0.0 if pred.answer == "bad" else 1.0)
    with dspy.context(lm=lm, adapter=dspy.ChatAdapter()):
        pred = best(question="q")
    return pred, lm.calls


def _reply(answer):
    return f"[[ ## answer ## ]]\n{answer}\n\n[[ ## completed ## ]]"


def test_first_attempt_is_greedy_whatever_the_configured_temperature_and_returns_when_it_passes():
    pred, calls = _run([_reply("good")])
    assert pred.answer == "good"
    assert len(calls) == 1
    assert calls[0]["temperature"] == 0.0
    assert calls[0]["rollout_id"] is None


def test_retries_run_at_temperature_one_with_distinct_rollout_ids():
    pred, calls = _run([_reply("bad"), _reply("bad"), _reply("good")])
    assert pred.answer == "good"
    assert [c["temperature"] for c in calls] == [0.0, 1.0, 1.0]
    assert calls[1]["rollout_id"] != calls[2]["rollout_id"]
    assert None not in (calls[1]["rollout_id"], calls[2]["rollout_id"])


def test_all_attempts_failing_returns_the_best_and_stops_at_n():
    pred, calls = _run([_reply("bad")])
    assert pred.answer == "bad"
    assert len(calls) == 3
