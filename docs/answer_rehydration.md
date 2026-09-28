# Answer-time rehydration

The ReAct loop and the extract step read the same `trajectory` for two different
jobs, and only one of them benefits from compaction.

The **loop** is a reader that can ask for more. When compaction swaps a 3 KB
listing for its 250 B offload label, the agent can still call `search_memory` on
that alias. That is what holds the peak prompt at
~37k tokens against the control's ~80k, and the benchmark measured it as a clear
win on trajectory correctness: 0 silent misroutes in 512 execute spans, 44 of 45
searches answered on valid handles, 479 of 479 observations archived.

The **extract step** is a reader that cannot ask for anything. It has no tools,
it runs once, and `trajectory` is its only evidence input
(`fastworkflow/utils/react.py`: `self.extract = ChainOfThought(fallback_signature)`).
At answer time the compaction that helped the loop is what leaves the writer
holding pointers, and the same benchmark measured that too: 35 of 260 deliverable slots
answered with "see Observation O23" about a listing sitting in that attempt's own
archive, and two attempts in ten that wrote up work which never ran.

So immediately before the extract call — and nowhere else — the extractor is
given its **own copy** of the trajectory with the evidence put back.

## What it does

Where the loop's trajectory carries an offload label —

> Use search_memory tool to search inside Observation O18 returned by list_permissions…

— the extractor's copy carries the raw archived observation for `O18` in its
place, re-printed with its alias line and the context clause recorded for it.

## What it does not do

* **Nothing is chosen by a model.** There is no ranking, no selection, no
  summarisation, no second LM call. Every byte added was produced by a command in
  this turn and stored under a digest.
* **Nothing is invented.** An alias with no archived text is left as its label and
  reported `unresolved`; a context clause that was never recorded prints the plain
  handle line rather than a guess.
* **The ReAct loop's own trajectory is never touched.** `rehydrate` returns a
  copy. The loop keeps the trajectory it ran on, the turn record is unchanged, and
  the 28 KB packed target, offloading and routing all
  behave exactly as they did.
* **The archive is read-only here.** No backend is called and no row is written.

## The budget

The extraction budget bounds the UTF-8 bytes of the whole extractor
trajectory. It is a fraction of the model's context window — 15625/32768 of it,
which is **250,000 bytes** at the 131,072-token reference window,
the size of the control's ~80k-token answer-time prompt. See
[`docs/context_budget.md`](context_budget.md) for the one input and the whole
table; `FW_ANSWER_REHYDRATION_MAX_BYTES` remains as a tuning override. The walk
goes **most recent first** and **stops at the first
replacement that would not fit**; everything older stays as it is.

Every alias left that way is named in one deterministic line appended to the copy
under the key `answer_rehydration_note`:

```
Not rehydrated for the answer (evidence exists under these observations): O5, O9, O14
```

(That is the line until 2026-09-28. Since fix-94m9 the prefix,
`answer_rehydration.NOT_REHYDRATED_PREFIX`, also tells the extractor to say so
in the answer:

```
Not rehydrated for the answer (evidence exists under these observations, but the answer's evidence limit was reached; say in the final answer that the rows of these observations are not included in it): O5, O9, O14
```

It changes the extract prompt only when the budget binds, but it is an
agent-visible wording change on the default path made without re-measuring
answers. The agent signature, the `search_memory` description and the offload
label now also say observations are *normally* restored; see
[`observation_search.md`](observation_search.md).)

Ascending by execute ordinal, always the same line for the same run. It exists so
the extractor can report those slots as **unresolved** rather than guessing at
them — the opposite of the pointer answer, which claims the evidence was seen.
The line is appended after the budget rather than reserved inside it, so a
rehydration that stops early can exceed the budget by the length of one line;
see the known limits in [`docs/observation_search.md`](observation_search.md).

If the extract call still overflows the model's context window, the existing
truncation fallback runs unchanged — on the copy, never on the loop's trajectory —
and the run records `rehydration_overflow`.

## There is no flag

Rehydration is what the extract step does, for every workflow: the loop keeps its compacted trajectory and the
writer gets the evidence behind it. A run with nothing offloaded
rehydrates nothing and its extract call is the call it always was — the
rule is the trajectory's content, not a setting.

| Variable | Default | Meaning |
|---|---|---|
| `FW_ANSWER_REHYDRATION_MAX_BYTES` | derived (**250,000** at a 131,072-token window) | Tuning override on the extraction byte budget. Below 4,096 or unparseable, the derived budget stands and a warning is logged. |

It is read **env file first, then the process environment**:
`fastworkflow.get_env_var` short-circuits on its default before consulting
`os.environ`, so a variable exported into the process but absent from the
workflow env file would otherwise read as the default — and, unlike the
readers it replaces, one written into the workflow's own `fastworkflow.env`
now takes effect.

## Events

Recorded through `observation_offloading.state.record_event`, so they land where
every other offloading measure does: the in-process ring (`snapshot_events()`)
and the `offload_events` table of the workflow's observability database, read
with `ObservabilityStore.offload_events(kind="rehydration_finished", ...)`.

| Event | Carries |
|---|---|
| `rehydration_started` | `budget_bytes`, `bytes_before`, `scope_id` |
| `rehydration_finished` | `bytes_before`, `bytes_after`, `bytes_added`, `rehydrated_labels`, per-alias `{alias, kind, added_bytes}`, `dropped_aliases`, `unresolved_aliases`, `stopped_on`, `extract_prompt_tokens`, `extract_duration_ms`, `rehydration_overflow` |
| `rehydration_overflow` | how many times the fallback truncated, the budget, `bytes_after` |
| `rehydration_failed` | the exception type and detail; the extract call then runs on the plain trajectory |

`extract_prompt_tokens` is read from the LM history when it is available. History
can be disabled or carry no usage block; the measure is then simply absent, never
estimated.

## Where it is wired

`fastWorkflowReAct._extract_prediction` (and `_async_extract_prediction`) is the
single place the copy is built. Every extract call site goes through it:

* `fastworkflow/utils/react.py` — `forward` (agent-selected finish and the
  iteration ceiling), `resume` (an `ask_user` continuation), `aforward`;
* `fastworkflow/observation_offloading/continuation.py` —
  `StructuredContinuationReAct._finish_prediction`, which is the one extract call
  of a segmented turn and therefore the site the measured configuration uses.

A failure anywhere in the copy — an unreadable or broken archive —
falls back to the plain call on the trajectory object it was handed, and records
`rehydration_failed`. An answer over pointers is worse than one over evidence
and far better than no answer.
`tests/test_answer_rehydration.py::ExtractHook::test_a_broken_store_costs_the_evidence_and_not_the_answer`
asserts exactly that.

## Related

* [`docs/observation_search.md`](observation_search.md) — the archive, the `O`
  namespace, `search_memory`, and the retention and known-limits contract.
* [`docs/context_budget.md`](context_budget.md) — the one input the 250,000-byte
  budget is derived from.
