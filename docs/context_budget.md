# Context budgets: one input, every byte budget derived from it

`fastworkflow/context_budget.py`.

fastWorkflow bounds several things by UTF-8 bytes: how much trajectory the ReAct
loop carries into the next step, how much evidence the answer-time extract call
is given, and how much an offload must save to be worth doing. Every one of them answers the same question — *how much of
the model's context window may this occupy* — so there is one input and the rest
are fractions of it.

Before 3.4.0 they were seven independent environment variables with seven
unrelated constants, which meant that moving a workflow from a 131k-token model
to a 32k-token one was seven correlated edits nobody made.

## The one input

The model's context window, in tokens. Resolved in this order:

| # | Source | How |
|---|--------|-----|
| 1 | `FW_MODEL_CONTEXT_TOKENS` | the deployment states it outright; read from the workflow's `fastworkflow.env` first, then the process environment. Values below `MIN_WINDOW_TOKENS` (4,096) or unparseable are refused with a warning and the next source answers. |
| 2 | the main agent model's metadata | `litellm.get_model_info(<LLM_AGENT>)["max_input_tokens"]` (falling back to `max_tokens`). A model litellm does not know is not an error; the next source answers. Answered once per process per model id. |
| 3 | the documented fallback | `REFERENCE_WINDOW_TOKENS` = **131,072**, the window the reference configuration runs at. |

`context_budget.context_window_tokens()` returns `(tokens, source)`, where
`source` is `"setting"`, `"model_metadata:<model id>"` or `"fallback"`.

**Tokens to bytes** is converted exactly once, in this module:
`BYTES_PER_TOKEN = 4`. It is a presentation ratio, not a tokenizer — these
budgets bound prompt text in UTF-8 bytes and 4 bytes per token is the ratio the
measured runs sat at. At the reference window that is **524,288 bytes**.

## The budgets

Each budget is a fixed rational fraction of the window in bytes, floored at the
minimum its own module always refused to go under. The fractions are calibrated
so that `cerebras/gpt-oss-120b` — for which litellm reports
`max_input_tokens = 131072` — reproduces the reference configuration's pinned
values exactly.

| Budget | What it bounds | Fraction of the window | At 131,072 tokens | Tuning override |
|---|---|---|---|---|
| `trajectory_max_bytes` | executor prompt target: the whole executor ReAct prompt (instructions, inputs, command list, trajectory) | 875/16384 (≈ 5.34 %) | **28,000** | `FW_TRAJECTORY_MAX_BYTES` |
| `answer_rehydration_max_bytes` | answer-time rehydration budget for the extract call | 15625/32768 (≈ 47.68 %) | **250,000** | `FW_ANSWER_REHYDRATION_MAX_BYTES` |
| `offload_min_saving_bytes` | minimum UTF-8 bytes an offload must free | 1/512 (≈ 0.195 %) | **1,024** | `FW_OFFLOAD_MIN_SAVING_BYTES` |

The executor prompt target bounds the whole executor ReAct prompt, not the
trajectory alone. When the prompt is over it, observations are offloaded
oldest-first until it fits. Its default is 875/16384 of the window in bytes (4
bytes per token), so it is a fraction of the agent model's context window: 28,000
bytes at the 131,072-token reference window.

The rehydration budget is by far the largest prompt share because the extract
call is one call with no loop after it.

**Observation search is not a context budget here.** When enabled (disabled 2026-10-09, code kept), `search_memory` sends the
full archived observation text to the search model (`LLM_OBSERVATION_SEARCH`, or
`LLM_AGENT` when that role is unset). Workflow authors must keep observations
small enough to fit that model's input context. See
[`docs/observation_search.md`](observation_search.md).

### Overrides are tuning, not the interface

The `FW_*_MAX_BYTES` names remain so that one budget can be moved without moving
the others — an experiment, a provider with an unusual prompt accounting. They
are not how a deployment picks its budgets: it sets the window, or lets the
model's own metadata set it, and leaves these alone. An override below its
budget's floor, or one that does not parse, is refused with a warning and the
derived value stands.

Unlike the readers they replace, an override written into a workflow's own
`fastworkflow.env` now takes effect: the reader consults the env file before the
process environment.

## Provenance

```python
from fastworkflow import context_budget
context_budget.budget_provenance()
```

```json
{
  "context_window_tokens": 131072,
  "context_window_source": "model_metadata:cerebras/gpt-oss-120b",
  "bytes_per_token": 4,
  "context_window_bytes": 524288,
  "reference_window_tokens": 131072,
  "budgets": {
    "trajectory_max_bytes": 28000,
    "answer_rehydration_max_bytes": 250000,
    "offload_min_saving_bytes": 1024
  },
  "overrides": {}
}
```

One call, one JSON-serialisable dict, so a runner records what the run was
bounded by without re-deriving it. `overrides` names only the budgets a tuning
override actually moved, so an empty `overrides` is itself the statement "these
are the derived budgets".

## What is not a context budget

The per-attribute cap on span attributes written to the **observability store**
is not one: it bounds a database row, not a model prompt, and nothing about it
scales with a model's window. It is the constant
`fastworkflow.tracing.MAX_ATTR_BYTES` (16,384), not a setting.
