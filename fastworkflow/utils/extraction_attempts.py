"""Retry policy for LLM parameter extraction: the first attempt runs greedy (temperature 0).

``dspy.BestOfN`` copies the language model with ``temperature=1.0`` and a fresh ``rollout_id`` for *every*
attempt, the first included, so every extraction was a temperature-1.0 sample whatever temperature
``LLM_PARAM_EXTRACTION`` was given. :class:`FirstAttemptGreedy` keeps BestOfN's contract (``N``
attempts, the reward function, the threshold, ``fail_count``, the best prediction wins, the first one at or
above the threshold returns at once) but runs attempt 1 at temperature 0 (``FIRST_ATTEMPT_TEMPERATURE``) with no rollout id override. Only
the retries (attempts 2..N, which happen when the reward is below the threshold or an attempt raised) use
BestOfN's diverse sampling: temperature 1.0 and distinct rollout ids.
"""
import litellm  # noqa: F401  (must precede dspy; see fastworkflow/utils/dspy_utils.py)
import dspy

from fastworkflow.utils.logging import logger

FIRST_ATTEMPT_TEMPERATURE = 0.0  # hard-coded temperature 0 for attempt 1: under test, not yet a decision
RETRY_TEMPERATURE = 1.0


class FirstAttemptGreedy(dspy.BestOfN):
    """``dspy.BestOfN`` whose first attempt is greedy (see the module docstring)."""

    def forward(self, **kwargs):
        lm = self.module.get_lm() or dspy.settings.lm
        start = lm.kwargs.get("rollout_id", 0)
        rollout_ids = [start + i for i in range(self.N)]
        best_pred, best_trace, best_reward = None, None, -float("inf")

        for idx, rid in enumerate(rollout_ids):
            mod = self.module.deepcopy()
            if idx == 0:
                mod.set_lm(lm.copy(temperature=FIRST_ATTEMPT_TEMPERATURE))
            else:
                mod.set_lm(lm.copy(rollout_id=rid, temperature=RETRY_TEMPERATURE))

            try:
                with dspy.context(trace=[]):
                    pred = mod(**kwargs)
                    trace = dspy.settings.trace.copy()

                    # NOTE: Not including the trace of reward_fn.
                    reward = self.reward_fn(kwargs, pred)

                if reward > best_reward:
                    best_reward, best_pred, best_trace = reward, pred, trace

                if reward >= self.threshold:
                    break

            except Exception as e:
                logger.warning(f"Extraction attempt {idx + 1} failed (rollout id {rid}): {e}")
                if idx > self.fail_count:
                    raise e
                self.fail_count -= 1

        if best_trace:
            dspy.settings.trace.extend(best_trace)
        return best_pred
