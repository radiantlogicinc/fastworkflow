"""One input -- the model's context window -- and every byte budget derived from it.

Before this module the framework carried seven independent byte
knobs (``FW_TRAJECTORY_MAX_BYTES``, ``FW_ANSWER_REHYDRATION_MAX_BYTES``,
``FW_RESULT_PAGE_MAX_BYTES``, ``FW_SEARCH_ANSWER_MAX_BYTES``,
``FW_RESULT_HANDLE_HOT_MAX_BYTES``,
``FW_OFFLOAD_MIN_SAVING_BYTES``), each with a constant of its own. The two
result-handle budgets are gone with the result-handle package; the history is
kept because it is what the fractions below were derived against. Every one of
them answers the same question -- how much of the model's context window may
this thing occupy -- so every one of them is now a fixed FRACTION of ONE input
and none of them has to be set to move a workflow to a bigger or smaller model.

The input, in this order:

1. ``FW_MODEL_CONTEXT_TOKENS``, when the deployment states it outright.
2. the main agent model's own metadata -- ``litellm.get_model_info(model)``
   ``max_input_tokens`` for the model named by ``LLM_AGENT`` -- so a workflow
   that changes models changes its budgets by changing its model.
3. the documented fallback, ``REFERENCE_WINDOW_TOKENS`` (131,072), which is the
   window the current values were calibrated against.

The conversion from tokens to bytes happens exactly once, here:
``BYTES_PER_TOKEN = 4``. It is a presentation constant, not a tokenizer: these
budgets bound UTF-8 bytes of prompt text and 4 bytes/token is the ratio the
measured runs sat at. Nothing else in the framework converts tokens to bytes.

Calibration. The fractions are pinned so that the reference window -- 131,072
tokens, which is what ``litellm`` reports as ``max_input_tokens`` for
``cerebras/gpt-oss-120b`` -- reproduces the previously hand-set values EXACTLY:
28,000 executor prompt target, 250,000 rehydration bytes, 3,072 bytes for a
result page, and 1,024 bytes of minimum offload
saving. ``tests/test_context_budget.py`` asserts that identity, so a change to
a fraction that would move one of those values fails the suite.

Overrides. Each budget keeps its ``FW_*_MAX_BYTES`` name as a TUNING override,
for the case where one budget has to move without moving the others. They are
not the interface: a deployment sets the window (or lets the model's metadata
set it) and leaves them alone. An override below the budget's floor is refused
with a warning and the derived value stands, exactly as the individual knobs
behaved before.

What is NOT here. The per-attribute cap on values written to the observability
STORE does not belong to this family: it bounds a database row, not a model
prompt. It is a constant (``fastworkflow.tracing.MAX_ATTR_BYTES``).
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from fractions import Fraction
from typing import Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# The one input
# ---------------------------------------------------------------------------

#: The explicit setting. First in the resolution order, and the only one a
#: deployment normally touches.
MODEL_CONTEXT_TOKENS_ENV = "FW_MODEL_CONTEXT_TOKENS"

#: The main agent model. Its metadata is the input when the setting is absent;
#: this is the same name ``dspy_utils.get_lm`` builds the agent's LM from, so
#: the window the budgets are cut from is the window the agent actually has.
AGENT_MODEL_ENV = "LLM_AGENT"

#: The tokens-to-bytes conversion, stated once for the whole framework.
BYTES_PER_TOKEN = 4

#: The calibration reference: ``litellm.get_model_info("cerebras/gpt-oss-120b")``
#: reports ``max_input_tokens = 131072`` (litellm 1.92.0). Every accepted
#: result-search experiment ran on that model, so this is the window the pinned
#: values below were measured at, and it is also the fallback.
REFERENCE_WINDOW_TOKENS = 131_072
REFERENCE_WINDOW_BYTES = REFERENCE_WINDOW_TOKENS * BYTES_PER_TOKEN  # 524,288

#: Below this a context window cannot hold a system prompt and one observation,
#: so a smaller value is a typo rather than a model.
MIN_WINDOW_TOKENS = 4_096

#: The three answers ``context_window_tokens`` can give for where the number
#: came from. The model-metadata source carries the model id after a colon.
SOURCE_SETTING = "setting"
SOURCE_MODEL_METADATA = "model_metadata"
SOURCE_FALLBACK = "fallback"


def env_value(name: str) -> str:
    """The raw setting of *name*, from the fastworkflow env file or the process.

    ``fastworkflow.get_env_var`` short-circuits on its ``default`` before it
    consults ``os.environ``, so a variable exported into the process but absent
    from the env file would read as the default. Both are checked, file first --
    which is also what makes an override written into a workflow's own
    ``fastworkflow.env`` take effect, where the per-knob readers this module
    replaced saw only ``os.environ``.
    """
    value = None
    try:
        import fastworkflow

        value = fastworkflow._env_vars.get(name)
    except Exception:  # noqa: BLE001 - a budget must never fail a turn
        value = None
    if value is None:
        value = os.environ.get(name)
    return str(value or "").strip()


# ---------------------------------------------------------------------------
# The budgets
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class BudgetSpec:
    """One derived budget: its fraction of the window and its tuning override.

    ``fraction`` is an exact rational, so ``reference_bytes`` is reproduced
    without a rounding step and a window twice the size gives a budget exactly
    twice the size. ``override_env`` is ``None`` for a budget with no tuning
    override, which is always the derived value.

    ``ceiling`` is ``None`` for a budget that scales without limit. When set,
    it caps the DERIVED value only: past it a bigger window no longer moves the
    budget, and an explicit tuning override may still exceed it.
    """

    name: str
    fraction: Fraction
    override_env: Optional[str]
    floor: int
    #: What this budget is for, one line, for the documentation table and the
    #: provenance record.
    what: str
    ceiling: Optional[int] = None

    @property
    def reference_bytes(self) -> int:
        """The value at the reference window -- this budget's pinned calibration value."""
        return self.bytes_for(REFERENCE_WINDOW_TOKENS)

    def bytes_for(self, window_tokens: int) -> int:
        """This budget at *window_tokens*, capped at ``ceiling`` and floored at ``floor``."""
        window_bytes = int(window_tokens) * BYTES_PER_TOKEN
        derived = int(window_bytes * self.fraction)
        if self.ceiling is not None:
            derived = min(self.ceiling, derived)
        return max(self.floor, derived)


#: The executor prompt target: how many bytes the whole ReAct prompt (instructions,
#: inputs, command list and trajectory) may occupy before compaction offloads
#: observations, oldest first. The trajectory gets what the rest leaves over.
#: 28,000 / 524,288 at the reference window.
TRAJECTORY = BudgetSpec(
    name="trajectory_max_bytes",
    fraction=Fraction(875, 16_384),
    override_env="FW_TRAJECTORY_MAX_BYTES",
    floor=1,
    what="executor prompt target",
)

#: The answer-time extraction budget: how many bytes of rehydrated evidence the
#: extract call may be given. 250,000 / 524,288 at the reference window -- by
#: far the largest share, because the extract call is one call with no loop
#: after it. Its floor is the old ``MIN_MAX_BYTES``: below it the prompt could
#: not hold one page of evidence.
ANSWER_REHYDRATION = BudgetSpec(
    name="answer_rehydration_max_bytes",
    fraction=Fraction(15_625, 32_768),
    override_env="FW_ANSWER_REHYDRATION_MAX_BYTES",
    floor=4_096,
    what="answer-time rehydration budget for the extract call",
)

#: The minimum trajectory saving an offload has to buy to be worth doing --
#: a threshold ON the trajectory, so it scales with it.
#: 1,024 / 524,288 at the reference window.
OFFLOAD_MIN_SAVING = BudgetSpec(
    name="offload_min_saving_bytes",
    fraction=Fraction(1, 512),
    override_env="FW_OFFLOAD_MIN_SAVING_BYTES",
    floor=0,
    what="minimum UTF-8 bytes an offload must free",
)

#: The model that reads a searched observation (``observation_offloading.search``).
SEARCH_MODEL_ENV = "LLM_OBSERVATION_SEARCH"

#: Every budget, in documentation order. ``docs/context_budget.md`` renders this
#: list and ``tests/test_context_budget.py`` walks it.
BUDGETS: tuple[BudgetSpec, ...] = (
    TRAJECTORY,
    ANSWER_REHYDRATION,
    OFFLOAD_MIN_SAVING,
)

#: The pinned values of the accepted stack, as constants, for the modules that
#: export a named default and for the calibration test. These are the budgets AT
#: THE REFERENCE WINDOW and nothing reads them to decide anything at runtime.
REFERENCE_TRAJECTORY_MAX_BYTES = TRAJECTORY.reference_bytes            # 28,000
REFERENCE_ANSWER_REHYDRATION_MAX_BYTES = ANSWER_REHYDRATION.reference_bytes  # 250,000
REFERENCE_OFFLOAD_MIN_SAVING_BYTES = OFFLOAD_MIN_SAVING.reference_bytes  # 1,024


# ---------------------------------------------------------------------------
# Resolving the input
# ---------------------------------------------------------------------------

#: ``model id -> max_input_tokens or None``. The metadata lookup is a table read
#: in litellm, but it is reached from the compaction hot path of every agent
#: step, so it is answered once per process per model.
_model_window_cache: dict[str, Optional[int]] = {}


def _model_window_tokens(model: str) -> Optional[int]:
    """``max_input_tokens`` for *model*, or None when litellm does not know it."""
    if model in _model_window_cache:
        return _model_window_cache[model]
    tokens: Optional[int] = None
    try:
        import litellm

        info = litellm.get_model_info(model) or {}
        raw = info.get("max_input_tokens") or info.get("max_tokens")
        if raw is not None:
            candidate = int(raw)
            if candidate >= MIN_WINDOW_TOKENS:
                tokens = candidate
    except Exception as error:  # noqa: BLE001 - an unmapped model is not an error
        logger.debug(
            "context window for %r is not in the model metadata: %s: %s",
            model, type(error).__name__, error,
        )
        tokens = None
    _model_window_cache[model] = tokens
    return tokens


def context_window_tokens() -> tuple[int, str]:
    """``(tokens, source)`` -- the one input and where it came from.

    ``source`` is ``"setting"``, ``"model_metadata:<model id>"`` or
    ``"fallback"``. It is on the provenance record so a measured run says which
    of the three answered, rather than leaving a reader to re-derive it.
    """
    raw = env_value(MODEL_CONTEXT_TOKENS_ENV)
    if raw:
        try:
            value = int(raw)
        except ValueError:
            logger.warning(
                "%s=%r is not an integer; falling back to the model metadata",
                MODEL_CONTEXT_TOKENS_ENV, raw,
            )
        else:
            if value >= MIN_WINDOW_TOKENS:
                return value, SOURCE_SETTING
            logger.warning(
                "%s=%d is below the minimum %d; falling back to the model metadata",
                MODEL_CONTEXT_TOKENS_ENV, value, MIN_WINDOW_TOKENS,
            )
    model = env_value(AGENT_MODEL_ENV)
    if model:
        tokens = _model_window_tokens(model)
        if tokens is not None:
            return tokens, f"{SOURCE_MODEL_METADATA}:{model}"
    return REFERENCE_WINDOW_TOKENS, SOURCE_FALLBACK


def reset_cache() -> None:
    """Forget the per-model metadata answers. For tests and for a model change."""
    _model_window_cache.clear()


# ---------------------------------------------------------------------------
# Reading a budget
# ---------------------------------------------------------------------------

def budget_bytes(spec: BudgetSpec, window_tokens: Optional[int] = None) -> int:
    """*spec* at the resolved window, unless its tuning override says otherwise.

    *window_tokens* names a window other than the agent's, for the one budget
    that is cut from a different model's: pass it and the derivation, override
    parsing, floor and warnings stay stated here once.
    """
    raw = env_value(spec.override_env) if spec.override_env else ""
    derived = spec.bytes_for(
        context_window_tokens()[0] if window_tokens is None else window_tokens)
    if not raw:
        return derived
    try:
        value = int(raw)
    except ValueError:
        logger.warning(
            "%s=%r is not an integer; using the derived budget %d",
            spec.override_env, raw, derived,
        )
        return derived
    if value < spec.floor:
        logger.warning(
            "%s=%d is below the minimum %d; using the derived budget %d",
            spec.override_env, value, spec.floor, derived,
        )
        return derived
    return value


def trajectory_max_bytes() -> int:
    return budget_bytes(TRAJECTORY)


def answer_rehydration_max_bytes() -> int:
    return budget_bytes(ANSWER_REHYDRATION)


def offload_min_saving_bytes() -> int:
    return budget_bytes(OFFLOAD_MIN_SAVING)


# ---------------------------------------------------------------------------
# Provenance
# ---------------------------------------------------------------------------

def budget_provenance() -> dict:
    """The one record a runner files: the input, its source and every budget.

    One call, one dict, no re-derivation on the reader's side. ``overrides``
    names only the budgets a tuning override actually moved, so an empty
    ``overrides`` is the statement "these are the derived budgets".
    """
    tokens, source = context_window_tokens()
    budgets: dict[str, int] = {}
    overrides: dict[str, int] = {}
    for spec in BUDGETS:
        effective = budget_bytes(spec)
        budgets[spec.name] = effective
        if effective != spec.bytes_for(tokens):
            overrides[spec.override_env] = effective
    return {
        "context_window_tokens": tokens,
        "context_window_source": source,
        "bytes_per_token": BYTES_PER_TOKEN,
        "context_window_bytes": tokens * BYTES_PER_TOKEN,
        "reference_window_tokens": REFERENCE_WINDOW_TOKENS,
        "budgets": budgets,
        "overrides": overrides,
    }


__all__ = [
    "AGENT_MODEL_ENV",
    "ANSWER_REHYDRATION",
    "BUDGETS",
    "BYTES_PER_TOKEN",
    "BudgetSpec",
    "MIN_WINDOW_TOKENS",
    "MODEL_CONTEXT_TOKENS_ENV",
    "OFFLOAD_MIN_SAVING",
    "REFERENCE_ANSWER_REHYDRATION_MAX_BYTES",
    "REFERENCE_OFFLOAD_MIN_SAVING_BYTES",
    "REFERENCE_TRAJECTORY_MAX_BYTES",
    "REFERENCE_WINDOW_BYTES",
    "REFERENCE_WINDOW_TOKENS",
    "SEARCH_MODEL_ENV",
    "SOURCE_FALLBACK",
    "SOURCE_MODEL_METADATA",
    "SOURCE_SETTING",
    "TRAJECTORY",
    "answer_rehydration_max_bytes",
    "budget_bytes",
    "budget_provenance",
    "context_window_tokens",
    "env_value",
    "offload_min_saving_bytes",
    "reset_cache",
    "trajectory_max_bytes",
]
