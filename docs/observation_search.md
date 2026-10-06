# Observation search

Large, older command results can be saved outside the ReAct prompt. A replacement
is emitted only when it is shorter than the original in both characters and
UTF-8 bytes, and only when the swap frees at least 1 KB (see
[Offload eligibility](#offload-eligibility)). Recent-observation protection and
the trajectory budget still apply. Replan copies follow the same savings rule and
persist any newly labelled command observation before returning a pointer.
Non-command observations in a replan copy stay inline. If this irreducible
evidence exceeds the byte target, the runtime records the overage and continues.

The replacement format is:

> Offloaded observation O8 returned by show_holders. It contains identity_uid: identities holding this permission; label: their display names. Normally restored for the final answer.

The closing sentence is the per-label reminder (`labels.LABEL_RESTORE_MARK`) of
the restore promise. The promise itself, and what follows from it (search only
for a value the next step needs, not to collect rows for the report), is stated
once, in the agent signature and the `search_memory` tool description, instead
of in every label; the longer per-label sentence cost ~100 bytes a label and
made fewer observations worth offloading (fix-2hxv). Labels in either earlier
wording ("Use search_memory tool to search inside Observation O8 ..." and "It is
restored in full when the final answer is written, so search it ...") still
parse, so recorded trajectories resume.

**The promise says "normally"** (fix-94m9, 2026-09-28). Until then the mark
read "Restored in full for the final answer." and the signature and tool
description said every observation "is restored in full when the final answer
is written". That was not always true: answer-time rehydration stops at its
byte budget (see [`answer_rehydration.md`](answer_rehydration.md)), and the
oldest observations are then left as labels. The mark is now "Normally restored
for the final answer." (one byte longer per label). The `search_memory`
description and the `WorkflowAgentSignature` docstring now say every observation
"is normally restored in full" and add "If the answer's evidence limit is
reached, the oldest observations are not restored and the answer names them."
Labels ending in the previous short mark still parse (`LABEL_RE` reads only the
prefix). These are agent-visible prompt changes on the default path, made
**without re-measuring** agent behaviour.

The command is the original `execute_workflow_query` command argument. Output
field descriptions come from the resolved command's authored `Signature.Output`
metadata. When those descriptions are unavailable, the label explicitly describes
the beginning of the command output. Hashes remain in the archive and tracing
manifest rather than taking space in the prompt label.

## Offload eligibility

An execute observation is eligible for offloading when replacing it with **its
own label** frees at least `MIN_OFFLOAD_SAVING_BYTES` (1,024) UTF-8 bytes:

```
utf8_bytes(command response, alias line stripped)
    - utf8_bytes(that step's actual offload label)  >=  1024
```

The label is built before the question is asked, because eligibility is a
property of the swap and not of the observation: the same 1.5 KB of output is
worth replacing under a 170 B label and is not under a 500 B one, and the label
carries the step's own alias, the full command argument and the command's
authored `Output` descriptions. `FW_OFFLOAD_MIN_SAVING_BYTES` overrides the
minimum (`0` means "offload whenever the label is smaller"); a value that is not
a non-negative integer logs a warning and falls back to the default.

Nothing else about compaction changes: oldest-first selection, the five most
recent execute observations protected, the 28,000 B packed target and
`replacement_saves_space` are as they were. A protected observation is never
priced, so no label is built for it. `replan_trajectory_skeleton` applies the
same rule — a label merely shorter than its observation is no longer enough
there either.

Decision records report `reason: below_min_saving` with the computed
`offload_saving_bytes` and `label_size`; `response_size` still carries
`characters`, `utf8_bytes` and `estimated_tokens`, and the `offload` event still
reports `estimated_tokens` beside the new `offload_saving_bytes`.

**This replaces a 1,000-estimated-token floor** (`ELIGIBILITY_THRESHOLD_TOKENS`,
~4 KB of ASCII), which asked how big an observation was rather than how much
residency replacing it would buy. Every listing page between ~1.3 KB and 4 KB
stayed resident for a whole turn although its label costs a few hundred bytes.
Replayed over eleven saved benchmark attempts (606 execute observations), the
new rule makes 80 more observations eligible — 1,272 to
3,555 B of listing pages and portraits — makes **none** ineligible, and would
have changed 73 actual offload decisions, always by offloading something that had
stayed resident. End-of-turn packed bytes fall by 1.7–26.8 KB per attempt, and no
recorded replan skeleton grows. Runs recorded before this change used the token
floor; their artifacts are unchanged and their numbers are not comparable
observation-by-observation.

## Canonical observation handles

Every `execute_workflow_query` observation is printed with its canonical handle
on the first line, inline results included:

> Observation O42 (execute_workflow_query ran in global)
> 477 holder(s).
> ...

`O{n}` is the execute ordinal — the n-th `execute_workflow_query` step of the
turn — **never** the ReAct step number. `compact.execute_ordinals` assigns it,
so the printed handle carries `ordinal_offset` (the count of execute steps the
context-window fallback has truncated away) and matches the alias an offload
label or the archive uses for the same observation. Non-execute tool outputs
(`search_memory`, `ask_user`, `what_can_i_do`, `intent_misunderstood`) get no
handle: there is nothing to search inside them.

### The context instance the command ran in

When the command ran inside a non-root command context, the line also names that
context and, where the workflow declares one, that context's instance identity:

> Observation O22 (execute_workflow_query ran in Account e8a0c3a1-… Alan Cooper)
> permission_uid  label
> 85cde168  Active Directory_Cloud Administrator
> ...

**Why.** A listing produced by navigating into a context carries no identifier of
the instance it belongs to: those permission rows do not repeat the account uid,
and the only thing tying them to Alan Cooper is that the previous step entered
his account. The link lives in the ORDER of the commands, so a reader that is not
allowed to use history — a `search_memory` answer, answer-time rehydration, the
extract step, a human scrolling a store — cannot recover it. In the benchmark
run that motivated the line, 12 of its 14 unresolved rows had exactly this cause.

**The context is the one the command RAN IN, not the one it entered.** It is
captured at dispatch (`CommandExecutor._remember_execute_context`), before the
command can move the context, and filed against the execute step's own `O` alias
in a turn-scoped table; `annotate_execute_observations` reads it rather than
recomputing it, because by the time the line is printed the current context has
already moved. So `open_account_by_uid` reads as the `DirectoryExplorer` command
it is, and the `list_permissions` that follows it is the one that belongs to the
account. An observation is evidence about the context it was produced in.

**A command that moved the context says so.** Read alone, "ran in X" is easily
taken for "now in X". So when the context after the command differs from the one
it ran in, the line ends with `; and resulted in a context change`; the
command's own response says where it moved to:

> Observation O5 (execute_workflow_query ran in Identity 4a0d… Angelica Schneider; and resulted in a context change)
> Context is now 'DirectoryExplorer'

A move is a different context object after dispatch than before, whether the
command returned or raised (`CommandExecutor._remember_context_change`); the
response is never read. The flag is held in process memory only. A resumed turn
keeps the lines it already printed; only a label rehydrated for the final answer
in another process prints without the suffix.

**The instance identity is declared, never derived.** fastWorkflow has no notion
of a context instance's identity — the current context is an arbitrary
application object — so `fastworkflow/context_identity.py` reads a declaration
off the same context callback class that already declares `get_parent` and
`enter_command`: a classmethod `instance_label(command_context_object) -> str`,
or an `instance_label_attr = "uid"` naming an attribute to read. A context that
declares neither prints its NAME alone, and an object that carries no identity
yields no identity: nothing is invented to fill the gap, for the reason
`tracing.context_handle` gives for refusing to mint an `instance_key` — a guess
that looks concrete is worse than an honest absence. The root context has an empty
clause and is printed as `ran in global`; a step with no recorded clause prints
the bare `Observation O{n} (execute_workflow_query)` line.

`context_clause` is the one place the clause is made printable. It removes
parentheses, semicolons and newlines and caps the name at 60 and the label at 80 characters,
which is what lets `ALIAS_LINE_RE` treat the closing `)` as unambiguous and match
lines printed before the clause existed. Each printed line emits a `context_line`
event carrying the alias, the clause, whether an instance was named and the bytes
the clause cost — the line is presentation and reaches neither the step span
(closed with the raw tool return) nor the archive, so the event is the only place
it can be measured.

This is not behind a flag. It is an extension of the canonical handle line,
which is not behind a flag either, and gating presentation would mean two
shapes of printed observation to
reason about for a change whose whole cost is ~40 bytes per observation.

`annotate_execute_observations` writes the line during the ReAct
`on_step_complete` hook, before compaction measures the packed target, so the
byte budget is checked against the trajectory the agent actually receives. The
line is written once and never rewritten: a surviving step's ordinal cannot
change, so a disagreement between a printed handle and the recomputed one is a
defect, recorded as an `alias_conflict` event, with the printed text left as it
stands. There is no step-number fallback anywhere — an `O` the run never printed
stays an explicit `no matching offloaded handle` miss.

**Archived text excludes the handle line**, the context clause included. The
line is presentation only:
`strip_alias_line` recovers the exact command response, and that response — not
the printed text — is what the archive stores, what its `text_sha256` covers,
what the offload label describes, and what the authored-output lookup matches
against the action log. Digests of observation text therefore stay comparable
with observations recorded before handles were printed, and `search_memory`
answers from the unmodified command response — from a bounded leading prefix of
it when the whole response does not fit the search model's budget, with the
truncation disclosed to the model. Offload eligibility, the minimum
saving and the savings rule are likewise evaluated on the response alone, so
printing a handle can never be what makes an offload look profitable — a
response that saves 1,023 B stays inline even though the printed text is ~34 B
longer.

**The clause is persisted, so the subject survives the process.**
The clause was captured at dispatch into a turn-scoped process map and nothing
else, so a turn resumed in another process had no subject for any of its
observations: the rehydrated handle line lost its clause, and a reader that had
only the resumed process could not say whose evidence a stored listing was.
`record_context_clause` now writes through to an `offload_subjects` row in the
workflow's observability database, keyed by `(turn_key, alias)`, and
`context_clause_of` reads through to it when the map misses and refills the map
from what it finds — the bounded runtime cache is REBUILT from the durable
record rather than kept a second way.

The table is part of the observability store's schema (see *Retention,
redaction and known limits*), carries the turn's `channel_id` beside its
`turn_key`, and is erased and pruned with the turn by the store's own
transactions, like the evidence it describes. An alias nobody stamped has no
row, which reads back as UNRECORDED — the state every reader already handles —
and never as a guessed subject. `reclaim_scope` drops the process-local copy
and never the row: residency, never evidence.

**The subject is handed to `search_memory` beside the evidence.**
Because the archived text is the raw response, a stored `list_permissions` page
is a table of permission rows with nothing in it saying whose permissions they
are, and the search model is instructed to use only the observation it is given
— so a subject-specific question could only be refused or answered from the
requesting agent's unsupported premise. `ObservationSearchSignature` therefore
takes a third input, `subject`, built by `declaring_subject` from the clause
recorded for that alias. It keeps three states apart: the recorded clause, the
recorded EMPTY clause (the command ran at the workflow root, which declares no
subject), and UNRECORDED, which says so and tells the model not to infer one.
The metadata travels beside the evidence and never inside it, so
`text_sha256` still covers exactly the bytes the command returned; and it is
paid for out of the same `search_observation_max_bytes` budget the evidence is
cut to (`evidence_max_bytes`), because the bound exists to fit the search
model's window and the whole input is what the provider measures.

**At answer time an offloaded observation is read back whole.** The extract step
has no tools, so an offload label is all the evidence it has unless something
puts the text back. Answer-time rehydration does exactly that, on the
extractor's own copy of the trajectory and under a byte budget: see
[Answer-time rehydration](answer_rehydration.md).

## Every execute observation is archived

Persistence no longer waits for an offload decision. When a step completes, the
same `on_step_complete` hook that prints the handle calls
`archive_execute_observations`, which writes **every** `execute_workflow_query`
observation into the scoped SQLite archive under its canonical `O` alias.
Offloading is then purely a residency decision — whether the text stays in the
prompt — and never a decision about whether the text can be found again.

Before this, only an observation that compaction chose to replace was persisted,
so an alias the run had just printed on an inline result resolved to
`no matching offloaded handle`: the evidence was visible in the prompt and
unreachable through `search_memory` at the same time.

What is stored is the raw command response, with the presentation line removed by
`strip_alias_line`, and `text_sha256` is the digest of exactly those bytes — the
same convention the offload path uses, so the later offload of an alias finds the
identical row rather than writing a second one. Writes are insert-or-nothing
(`ON CONFLICT DO NOTHING` plus a digest check), and a digest already written in
this process for that alias is skipped, so revisiting a step across the many
compaction passes of a turn costs nothing and can never produce a duplicate.
The archive key is the alias actually **printed** on the observation when there
is one, so the handle the agent can see is always the key its text is stored
under — including in the `alias_conflict` case, where the printed alias stands
and the recomputed one is not used.

Each first write is recorded as an `observation_archived` event (alias, step,
digest, bytes, hot-cache evictions) and puts a copy in the bounded hot cache, so
the existing cap and oldest-first eviction still apply — inline observations now
compete for that cache too, and an evicted alias simply resolves from SQLite at
the `sqlite` tier. The eager archive is independent of the offload decision, so
changing the eligibility rule moves only residency: an observation the rule keeps
inline is archived and searchable exactly like one it offloads.

Persistence is an availability optimisation on the hot path of every agent step,
so a failure must never cost evidence. A failed write records an
`archive_refused` event (`reason: persistence_failed_original_retained`) and
leaves the observation inline and unchanged; nothing is raised into the agent
loop. Rewriting a completed observation's text under a live alias is refused the
same way: the stored evidence stands.

## Inline and offloaded searches, and what a miss means

`search_memory` resolves any alias printed in the current scope, whether its text
is still inline or already replaced by a label, and answers from the same
archived bytes either way. The search event records the answering tier
(`hot`/`sqlite`) as before, plus `still_inline`:

| `still_inline` | meaning |
| --- | --- |
| `true` | the observation was still in the prompt when the agent searched it |
| `false` | it had been offloaded to a label |
| `null` | this process has no record of that alias being printed in this scope |

So a miss with `still_inline: null` is a wrong-handle selection — an invented or
mis-remembered `O` — while a miss on a handle the run did print would be a
retrieval failure. The flag is process-local bookkeeping, so a turn resumed in
another process reports `null` until it prints handles again; `status` still
says whether the search was answered. There is still no step-number fallback and no nearest-handle
guess: an alias that was never printed is an explicit miss, recorded with
`status: missing`, and no model is called.

## Search output residency

An execute observation that grows large is offloaded and replaced by a label. A
`search_memory` answer is not: it is a non-execute observation, so
`compact_trajectory` never selects it, and `replan_trajectory_skeleton` labels
execute observations only, so the answer is carried into every later segment of
the turn in full. Whatever a search answer costs, it costs for the rest of the
turn — and its size is model output, capped only by the 2,048-token completion
limit (roughly 8 KB).

So a search observation is held to the same 3 KB budget a listing observation
has. The budget covers the **whole observation**, header and marking included,
not just the answer body:

```dotenv
FW_SEARCH_ANSWER_MAX_BYTES=3072   # default; values below 1024 fall back to it
```

An answer that fits is presented exactly as before, byte for byte:

```
Observation O34 (tier=hot):
Christopher Hubbard (identity_uid=c062...) holds it; 3 of the 22 remediation ...
```

An answer that does not fit is archived whole, then cut at a line boundary by
`text_page` — the same rule paging uses, so an identifier the answer offers as
evidence is never split mid-token and a row is never halved into a shorter,
plausible-looking one — and the observation says what happened:

```
Observation O34 (tier=hot, bounded):
00000000000000000000000000000000 Person 0 account_uid=account-00000
... 38 whole rows ...
[search_memory BOUNDED ANSWER: shown 2,612 of 27,889 UTF-8 bytes of the answer
for O34; 25,277 bytes are NOT shown. This is not the complete answer, and nothing
missing from it is thereby absent from O34. Full answer archived as O34#a1
(sha256 9f3c1a2b4d5e). To get the rest, call search_memory on O34 again with a
narrower question naming the entity or predicate you still need.]
```

A bounded answer is never presented as a complete one. The marking states the
omission in bytes, denies the absence inference an incomplete answer would
otherwise invite, and gives an action the agent can actually take: the same
observation, a narrower question — which is evidence-grounded, where re-reading
a truncated answer is not.

`O34#a1` is a **record key, not a handle**. The agent-visible `O` namespace is
execute ordinals only, and `search_memory` validates its `alias` against
`O[1-9]\d*`, so this key can never be passed back as an observation: an answer
record is not a searchable observation. The prefix files the answer under the
observation that produced it and the suffix separates repeated searches of the
same observation within one scope. Operators and evaluation tooling read the
complete text with `archived_search_answer("O34#a1", scope=...)`, digest-verified
by the archive and with no second model call; the search event carries the whole
answer regardless, so a bound never loses the evidence.

Archiving happens **before** the cut, and evidence outranks the byte budget: if
that write fails, the answer is not bounded at all. The complete text is returned
inline, over budget, and `search_answer_archive_refused` records
`persistence_failed_complete_answer_retained` — the same choice a failed offload
makes when it keeps its observation inline.

The answered search event gains `answer_bounded`, `answer_utf8_bytes`,
`observation_utf8_bytes` and, when bounded, `answer_shown_utf8_bytes`,
`answer_omitted_utf8_bytes`, `answer_archive_key` and `answer_sha256`.

The bound is a tail guard, not a saving. Measured over the saved benchmark
stores, the largest of 27 recorded answers
was 1,855 B — 60% of the budget — peak completion usage was 790 of 2,048 tokens,
and the `completion_limit` branch has never been taken. Search answers held
2.1–7.4% of end-of-turn packed bytes and 0.0–6.8% at peak, behind execute
observations, thoughts and arguments, and `what_can_i_do` output in every run.
Nothing in the code prevented an 8 KB answer; the runs simply had not produced
one. Search observations were deliberately **not** made eligible for oldest-first
compaction: compaction offloads execute observations only, an offload label is
about the size of a typical answer (median 355 B) so neither
`replacement_saves_space` nor the 1 KB minimum saving would admit the swap, and
reading a label back would cost a paid model call to recover a few hundred bytes.
(That measurement was taken while eligibility was still the 1,000-token floor;
the 1 KB minimum saving refuses these answers for the same reason, only more
directly.)

## Configuration

Setting the search model independently of the main agent is recommended but not
required: when `LLM_OBSERVATION_SEARCH` is unset, search runs on `LLM_AGENT`
and the credential configured for it, and the evidence budget below is then
sized from the agent model's window instead.

```dotenv
# fastworkflow.env
LLM_OBSERVATION_SEARCH=cerebras/gpt-oss-120b
```

```dotenv
# fastworkflow.passwords.env
LITELLM_API_KEY_OBSERVATION_SEARCH=<provider API key>
```

The standard `get_lm` routing rules also support `litellm_proxy/` routes and
provider ambient credentials.
The search uses temperature 0, a 2,048-token completion limit, a 120-second
request timeout and one provider retry. `FW_LM_CACHE=0` disables response caching
for independent benchmark calls.

Compaction budgets are derived from ONE input — the model's context window —
in `fastworkflow/context_budget.py`; see
[`docs/context_budget.md`](context_budget.md) for the input, its resolution
order and the whole table. The values below are what a 131,072-token window
(`cerebras/gpt-oss-120b`, the reference main agent model) produces. Each
name remains as a **tuning override**, optional, and falls back to the derived
budget on a value that is not a valid integer or is below its minimum:

| override | derived at a 131,072-token window | meaning |
|---|---|---|
| `FW_OFFLOAD_MIN_SAVING_BYTES` | 1024 | minimum UTF-8 bytes an offload must free |
| `FW_TRAJECTORY_MAX_BYTES` | 28000 | packed-trajectory target and replan bound |
| `FW_OFFLOAD_HOT_MAX_BYTES` | 262144 | hot handle cache cap |
| `FW_SEARCH_ANSWER_MAX_BYTES` | 3072 | presentation bound on a search answer |
| `FW_SEARCH_OBSERVATION_MAX_BYTES` | 131072 (cut from the *search* model's window, capped at 131,072; the override may exceed the cap) | one observation handed to the search model (fix-deus; see *The read is bounded too*) |

Observation offloading itself has no switch: `build_tool_agent` always returns a
`StructuredContinuationReAct` with `search_memory` in its tools, and the forced
replan bound defaults to the module constant
`observation_offloading.continuation.MAX_FORCED_REPLANS` (3, therefore 4
segments; 2 and 3 until 2026-09-27). `FW_MAX_FORCED_REPLANS` overrides it per
process (fix-vd5c): a value outside `[0, 10]` is clamped and one that is not an
integer is ignored, each with one warning per raw value. The default rests on
one ido task, not on a measured segment tail across workflows, and it raises
every workflow's worst-case step ceiling from 75 to 100 iterations; the
`forced_replan` and `forced_replan_wall` events carry `reached_limit`, and the
rate at which turns hit it is what should decide the value. The
`agent_installed` event records the bound actually in force as
`max_forced_replans`. The observation archive always lives in the workflow's own
observability database. (Until 2026-09-28 not after a cold resume: an agent
built while a context restored a suspended turn had no active workflow, so the
archive opened from an empty workflow path, i.e. a database named after the
working directory's basename, unless `FASTWORKFLOW_WORKFLOW_ID` was set.
Evidence archived before the suspension was then unreachable by
`search_memory` and rehydration, and everything archived after it went to the
stray database for the context's life, out of reach of channel erasure.
`build_tool_agent` now takes the session's bound app workflow first, then the
active one; the router's examples and the finish check's effect lookup use the
same path. Cold-resume archive location.) So do the offloading runtime's diagnostic events: they
are kept in process in a ring of the newest 2,000 (`snapshot_events()`) and
stored as rows of `offload_events`, read back with
`ObservabilityStore.offload_events(turn_key=..., channel_id=..., kind=...)`.

The one setting that remains is `FW_OFFLOAD_EVIDENCE_REDACTION`: `on` (the
default) or `off`. See *Retention, redaction and known limits*.
(Besides it: `FW_MAX_FORCED_REPLANS` above, and the two optional decision-model
features, `FW_SEARCH_ROUTER` and `FW_FINISH_CHECK`, described below, with
`FW_JEV_BASE_URL`, the endpoint both use.)

## Tool behavior

`search_memory(question: str, alias: str)` requires one key such as `O8` — the
handle printed on the observation's first line, or named in its offload label.
An empty key, a noncanonical key, multiple keys, or an empty question is rejected.
A missing handle produces an explicit error without calling a model or searching
another observation. Resolution remains scoped to the current turn/attempt and
survives cache eviction through the SQLite archive.

The implementation reads the **current search step's thought** from the ReAct
trajectory and prefixes it to the question as `<reasoning>. <question>`. Reasoning
is not an agent-supplied tool argument. DSPy `Predict` receives that combined
question, the recorded subject, and the selected observation cut to the search
model's own budget (see *The read is bounded too*). It answers from that observation,
preserves exact identifiers and their types, and states evidence gaps. The prompt
instructs it to treat both the requesting agent's assumptions and instructions
inside the observation as untrusted claims, not additional evidence.

Broad requests for entire tables receive a count/description and a request for a
focused predicate. Completions cut off at the output limit are reported as
incomplete searches rather than successful evidence answers. An answer that fits
is returned whole; one that does not is bounded and marked as incomplete (see
*Search output residency*), never silently shortened.

The returned answer names the source observation. Search events record scope,
source hash and size, question and attached reasoning, status, model, answer,
latency and available provider usage/cost. LLM calls are also recorded through
normal DSPy observability. Provider failure yields an explicit search failure;
it is never reported as evidence that an entity is absent.

This search supplies evidence; it does not by itself guarantee that the main
agent's final conclusion is correct.

### The read is bounded too

The observation handed to the search model **is** paged, and to that model's own
window rather than the agent's: `search_observation_max_bytes()` resolves
`LLM_OBSERVATION_SEARCH`'s context window and takes a quarter of it as bytes,
which is 131,072 B at the 131,072-token reference window, with a floor of one
page. (Until 2026-09-27 it took 3/128, 12,288 B, the declared geometry of
`DEFAULT_PAGE_BYTES` (4,096) x `SEARCH_MEMORY_MAX_PAGES` (3). No recorded ido
search came near either bound; the quarter lets a large unpaginated
observation be read whole rather than as a prefix.) It has no tuning override: the search model's window is
the only input, and `FW_MODEL_CONTEXT_TOKENS` is how a deployment corrects that
window. (That sentence is history since fix-deus, 2026-09-27; what replaced it
follows.)

**The derived bound has a ceiling, and its own override** (fix-deus). The
quarter of the window is capped at `SEARCH_OBSERVATION_CEILING_BYTES`,
**131,072 B**, so a very large search window no longer turns one search into a
megabyte prompt re-sent on every search of that observation. On the example
configuration's search model, `mistral/mistral-small-latest`, whose litellm
metadata reports a 262,144-token window, that halves the bound from 262,144 B
to 131,072 B: observations of 128–256 KB are now read as a prefix, with the
bounded-evidence marker below. `FW_SEARCH_OBSERVATION_MAX_BYTES` sets the bound
outright, in bytes, and may exceed the ceiling; it goes through the same
parsing as every other budget override, so a value below the 4,096-byte floor,
or one that is not an integer, is refused with a warning and the **derived**
value (not the floor) stands.

**The window is the smaller of the setting and the search model's own**
(fix-deus). Until 2026-09-27 a valid `FW_MODEL_CONTEXT_TOKENS` won outright.
That setting usually describes the *agent's* window, so a large one sized the
evidence past a small search model's window. Now, when both the setting and
the search model's litellm metadata are known, the smaller one answers and
`search_window_source` names whichever that was (the setting on a tie). With
only one known, it answers; with neither, the agent's window
(`context_window_tokens`) does. The search model's metadata is only consulted
when `LLM_OBSERVATION_SEARCH` is set: with it unset, the window is the agent's,
resolved as before, so a setting still wins outright there.

The subject metadata is paid for out of that same budget, so nothing
travelling to the model escapes the bound the model's window imposes. Without
it, an execute observation archived at full size (measured at 440,000 B) was
re-sent whole on every search of it.

The cut is `bounded_evidence`, built from successive `text_page` calls so the
budget is actually spent on a text with no line structure, and the observation
says what was left unread:

```
[search_memory BOUNDED EVIDENCE: answered from the first 12,150 of 440,102 UTF-8
bytes of O34; 427,952 bytes were NOT read. ... Re-asking O34 reads the same first
bytes however the question is worded; to reach the rest, re-run <command> with a
narrower filter or a smaller page and search the new observation.]
```

The action is deliberately **not** "ask a narrower question": every search reads
from byte 0, so the same observation answers from the same bytes however the
question is phrased — the opposite of the bounded-*answer* case above, where
re-asking is the right move. Only the producing command changes the bytes.

The search model is told the same thing, in band with the evidence: when the
observation was cut, a one-line `[TRUNCATED: …]` notice is appended to the text
it receives, saying how many further bytes exist and that nothing missing from
the prefix may be reported as absent. So the model never reads a prefix of a
list as the whole list.

When the provider refuses even the bounded prompt, the outcome is typed rather
than generic: `is_context_window_error` matches `ContextWindowExceededError` on
the exception's class chain, or the providers' wordings as a fallback, and the
observation opens with `[search_memory INPUT OVER WINDOW:` — what was sent, that
the retry cannot succeed, the agent's move, and the operator's (set
`FW_MODEL_CONTEXT_TOKENS` to the search model's real window, or use a larger
`LLM_OBSERVATION_SEARCH` model).
The provider message itself is inspected, never printed: it can carry payload or
credentials.

The search event carries `observation_bytes`, `observation_sent_bytes`,
`observation_bounded`, `observation_max_bytes` and `evidence_max_bytes`, so the
share of searches answered from a prefix is measurable rather than inferred.

### Answers that never reach the search model

Two kinds of search are answered in code:

- **A short observation** (at most `SHORT_OBSERVATION_BYTES`, 256 B) is
  returned verbatim, with up to three handles in the turn whose command and
  subject match the question better (the tool's description says: mention
  the question's words more). Event status `short_verbatim`.
- **A request for every row of a listing** is answered by copying the rows
  (`observation_offloading/listing.py`), as many whole rows as fit the answer
  bound, with a closing line stating how many were shown and ending with the
  same short `Restored in full for the final answer.` marker the labels carry.
  (Since fix-94m9, 2026-09-28, that marker is `Normally restored for the final
  answer.`, the same `labels.LABEL_RESTORE_MARK`.)
  A listing is served only when its parse is provably complete. Event status
  `rows_served`.

**Short observations.** The header names the observation's own subject:
`[search_memory SHORT OBSERVATION: O3 is the complete response of
list_entitlements, in Account 9f1e Heidi Turner, shown verbatim because it is
too short to search]`; a root-context observation says `, at the workflow
root`, and an unrecorded subject adds nothing (fix-lzdz). The searched handle
is scored with the same formula as the others (`relatedness`: 3 per question
word in the command name, 2 per word in the subject clause), and another
handle is offered only when it scores **strictly more** -- so a short answer
about the right subject is no longer undercut by a longer observation about
another one (the Heidi/Alan case). Until 2026-09-27 every handle scoring above
0 was offered, and the hints read "observations in this turn that …" and "No
other observation in this turn matches the question's words; run …". The four
hints now read:

- with suggestions: "If it does not answer the question, other observations of
  this turn that mention the question's words more are: …. Search one of those
  instead.";
- with none: "No other observation in this turn matches the question's words
  more than this one; if it does not answer the question, run the command that
  produces what you need.";
- when the turn's handles could not be listed (fix-kvq0): "Other observations
  of this turn could not be listed, so none is suggested here." A locked,
  unavailable or broken archive no longer blocks the search for 30 s or
  raises: `list_summaries` waits at most `SUMMARY_READ_TIMEOUT_SECONDS` (0.5 s)
  for the database, any failure is caught, and the searched handle's own
  subject is then read from process memory only;
- in a broad scope (fix-y570): "Other observations are not listed in this
  scope, so none is suggested here." A scope is broad when its turn key is its
  channel (`archive.is_broad_scope`) -- the process-default scope and the
  between-turns fallback -- because its rows span every turn (and, for the
  default scope, every session) that fell back to it. `list_summaries` returns
  nothing for such a scope, and it now also filters by `channel_id` as well as
  `turn_key`. `get()` and `list()` are unchanged; their cross-channel read
  under those scopes is tracked separately (fix-tyzj). (Since 2026-09-28,
  fix-tyzj: every archive read and subject write is scoped to the channel as
  well as the turn key -- `get`, `list`, `get_subject`, `capture_record`,
  `forget_subject` and the stored-digest check all filter on `channel_id`, and
  `put_subject`'s upsert updates only a row of the same channel. A scope that
  pairs another channel with this turn's key reads nothing and cannot replace
  or forget this channel's subjects; persisting under it still raises the
  collision error, even for identical text. Since 2026-09-28 that error reads
  "runtime handle alias is already stored for this turn (different text or
  another channel)"; it said "collides with different text", which was wrong
  for identical text from another channel.)

Relatedness reads Unicode letters and digits, casefolded, still split at
underscores; the English stopwords and the plural fold are kept (fix-1593,
relatedness half). The observation's text is the backend's, printed between
framework lines, so any line of it shaped like a framework marker -- a line
containing `[search_memory`, `Observation O<n> (`, or an offload-label prefix,
case-insensitively -- is printed with a visible `> ` in front
(`labels.quote_marker_lines`; fix-znxq, verbatim-quoting part).
(Since 2026-09-28, fix-vpe3, the relatedness score, the related-handle
suggestions and the short-observation answer live in
`observation_offloading/related.py`; `search.py` re-exports every name, so
imports through `search` are unchanged, and behaviour is unchanged.) The same
quoting applies to the preamble, column line and rows of a served listing, so
the closing line is the only unquoted marker in it. Nothing reads these texts
back, so the quoting is one-way.

**Served rows fit the bound exactly, or the model answers** (fix-3zxk).
`served_rows` picks the largest row count whose own closing line still fits the
answer bound, and returns `None` when not even one row fits beside the text
above the rows -- the search then goes to the model, with `listing_skip_reason`
`rows_do_not_fit`. The closing now reads "rows 1-N of the M rows listed in O5"
(until 2026-09-27: "… rows in O5"). Before this, a large preamble produced a
5,410 B answer against a 3,000 B budget, closings overran by up to 3 B, and an
oversized row produced a zero-row answer.

Whether a request wants every row is decided by an optional decision-model
router (`observation_offloading/search_router.py`). It is off unless
`FW_SEARCH_ROUTER=jev` and `JEV_API_KEY` are both set, because it sends the
question, the agent's reasoning and the observation's first four lines to
TypeSafe. **Data egress:** that text leaves the deployment with only the
capture policy's credential patterns scrubbed by default; names, identifiers
and other personal data in it are sent as they are. It makes one attempt with a
2-second timeout and fails open to the search model (hard wall clock and at
most 3 calls per turn since 2026-09-27; below). `FW_SEARCH_ROUTER_MODEL`
pins the model (default `jev-1.13.0`); the router is built once per workflow
path, model and key, so a rotated `JEV_API_KEY` builds a new client (the key is
cached only as a fingerprint). A set `FW_SEARCH_ROUTER` that cannot take effect
-- a value other than `jev`, `typesafe-sdk` not installed, or no `JEV_API_KEY`
-- logs one warning per cause and routing stays off; an unset flag is silent.
(Since 2026-09-27 the list of causes is longer, and the flag may name a
registered provider: see *Decision providers, endpoint and vendor time*.)
When the turn is traced, each routing call is also an `fw.search.route` span.
Its tokens are recorded there and on the event, and are **not** included in the
`fw.llm.call` usage or cost totals: the call goes through the vendor SDK, not
litellm. A workflow can add examples to the router's
question in `<workflow>/search_router_examples.json`. Install with the
`jev` extra (`router` is a deprecated alias of it, kept for existing installs).

Since 2026-09-27 the router shares the finish check's plumbing (see *Decision
providers, endpoint and vendor time* below): the 2-second timeout is a hard
wall-clock cutoff, a turn routes at most `ROUTER_CALLS_PER_TURN` (3) searches,
and once those are used, or the turn's vendor time is spent, `route` returns
the error `router_budget` with no call, no span and no warning, and the search
model answers (`listing_skip_reason` `router_budget`). What it sends passes
`jev_client.egress`: when the capture policy would withhold any of the three
values at call time, nothing is sent and the router returns the error
`policy_withheld` (`error_stage` `redaction`, unwarned; its span, when traced,
has status error and `error_type` `policy_withheld`).

### How a listing is recognised: fail-closed

`listing.parse_table` returns a listing only when it is sure the rows are the
whole listing, and otherwise refuses (returns `None`), so no search is ever
served a listing that silently stopped early. Since 2026-09-27 (fix-zoup,
fix-n4z9, fix-36gv, fix-k56c, fix-3zxk, fix-1593):

- **The first header candidate that accepts a row decides.** A later row it
  cannot place (an aligned row with more cells than the header, a markdown or
  tabbed row with another cell count) refuses the whole text; no later line is
  retried as a header. Retrying from every later line is what made refusals
  quadratic: a 104 KB aligned listing refused at its last line took 2.51 s and
  key/value lines 4.59 s; both now take ~2 ms.
- **Parsing runs after the short-observation check**, so a short observation
  is never parsed: `listing_parsed` is `False` and `listing_shape` `None` on
  every `short_verbatim` event.
- **An aligned listing does not end on an ordinary row.** A one-cell line
  directly under aligned rows that reads as one of them -- a wrapped (indented)
  label, separators collapsed to one space, a single token as wide as the first
  cells (empty trailing cells) -- refuses the parse. Header offsets are used
  only when the rows are padded to them.
- **A listing with more around it is refused**: any row-shaped line anywhere
  after it (a second group or table); a last row starting with `total`, `sum`
  or `count` -- or, since 2026-09-28 (fix-4riz), a labelled last row, one whose
  first cell (read per shape: aligned, markdown, tabbed) ends with `:` or is
  one word followed by `:` and a space (`Note:  2 items`, `Legend: x ...`),
  matched by shape rather than a word list; the lax
  `require_complete=False` read drops such a row as it drops a summary row --
  or text above or below the rows saying there is more --
  `remaining=N>0`, `complete=false`, `has_more=true`, a `shown=` or `total=`
  that disagrees with the row count, `pages>1`, `N of M` with N≠M, `A-B of M`
  not covering 1..M, `page=2` or later, `next page` / `next cursor`, `more
  rows` (but not `no more rows`), `truncated`, `not shown`. ido's `shown=`
  stays a built-in generic count.
- **Wider recognition, conservatively**: column names may be Unicode (`Größe`,
  `名前`); markdown also accepts the compact `|-|-|` separator and the
  pipe-less `a | b` / `--- | ---` form, only when the separator has exactly one
  cell per column (and, pipe-less, the column line reads as column names).

Some footers are now refused rather than dropped, for example a single word
that fits the first column of a padded table, a `Total  2` row, or a line with
two or more spaces after the listing. On 4,000 random listings the new parser
agreed with the old one on 3,298, refused 702, and never accepted a text the
old one refused. With the router off, the only visible effect is telemetry:
`listing_parsed` is `False` more often. The labelled-footer rule (fix-4riz)
was compared old against new over 3,832 distinct recorded ido texts from 361
databases: 494 accepted in strict mode and 2,467 in lax mode, both before and
after, with no difference. Times, URLs, `x:1`, a colon in a later cell and a
`Note:` in an earlier row are still accepted.

Every search event says which path the search took, so a `router` of `None` is
never ambiguous (fix-wheb):

| field | meaning |
|---|---|
| `router` | the router's verdict (choice, `p_all_rows`, `for_report`, `latency_ms`, usage or error), or `None` when no routing call was made. Since 2026-09-27 every record also carries `vendor_ms` (the turn's vendor time so far; `None` without a turn budget) and `provider` (the flag value that selected who answers: `jev` or a registered name); an error record carries `error_status`, `error_request_id`, `error_code` (a single machine-readable token from the body, never its text) and `error_stage` (`redaction`, `request`, or `budget` when the turn's vendor time ran out mid-call) |
| `router_enabled` | whether a router was attached to this agent |
| `listing_parsed` | whether the observation parsed as a complete listing |
| `listing_shape` | `aligned`, `markdown` or `tabbed`, or `None` |
| `listing_skip_reason` | why rows were not served: `router_disabled`, `no_listing`, `router_error`, `router_budget` (the turn's routing calls or vendor time were used up), `policy_withheld` (since 2026-09-28: the capture policy withheld a value the router would send), `not_all_rows`, `below_threshold`, `rows_do_not_fit` (not one whole row fits the answer bound beside the text above the rows), `short_observation`, `missing_handle`; `None` when they were |
| `for_report` | the router's verdict on whether the rows were wanted only for the final answer; recorded, it does not change the path |

The router's error values are the error's class name (`CallTimedOut`,
`OutOfTime`, `VendorBusy`, an SDK error class, …), or `policy_withheld` or
`router_budget`; the last two are not failures and are not warned about. A
`policy_withheld` router record is counted under `listing_skip_reason`
`router_error`, which has no reason of its own. (History: since 2026-09-28 it
has its own reason, `listing_skip_reason` `policy_withheld`, and a withheld
route no longer uses one of the turn's `ROUTER_CALLS_PER_TURN` calls. Router
records and `finish_check` events also carry `vendor_calls_in_flight`, the
vendor worker slots held, when it is above 0, and `vendor_calls_orphaned`, the
abandoned requests still running past their slot, when that is above 0.)

A `short_verbatim` event also records `related_scores` (the match score of each
offered handle, in the order of `related`) and `subject_recorded`; since
2026-09-27 also `own_score` (the searched handle's own score, which an offered
handle must beat), `related_lookup_failed`, `related_scope_refused` and
`related_lookup_error` (`None`, an exception class name, or
`archive_unavailable`). When the lookup failed, `subject_recorded` reflects
process memory only. A
`rows_served` event also records `served_over_bound` (whether the served text
exceeded the answer bound: the preamble, column line and closing line are
always served, so an oversized preamble can push it over -- history since
fix-3zxk: such a search now goes to the model as `rows_do_not_fit`, so the field
should always be `False`) and `trailing_lines_dropped` (non-blank lines after the
listing's end, such as a footer, that serving omits).

## Finish-time execution check

When the agent chooses `finish`, an optional check
(`observation_offloading/finish_check.py`) asks the same decision model, for
every step of the turn's initial plan and every subject the request names,
whether the turn's record shows the step executed. It reads a ledger built from
the turn's full trajectory -- one row per step with its command, the context it
ran in and the context it left, identifiers resolved to the labels retrieved
listings gave them, the subjects found in its full output (from the archive),
its outcome and the first 200 bytes of its output -- all passed through the
capture policy before sending. Steps it judges unexecuted are listed in one note
that replaces the finish observation; the agent may act on it or finish anyway.
One note per turn, never with fewer than two iterations left in the current
segment, never for optional or user-gated steps. The count the note states
includes the iterations later forced-replan segments still hold
(`shown_iterations_left` on the event); the two-iteration gate does not.
It checks the turn's initial plan. While the check is on the planner is asked
for a structured plan (steps, parts, subjects, optional / needs-the-user
flags); when that call fails to parse or returns no steps, the plain-text
planner runs and its plan is parsed back into steps, so a text-fallback plan is
checked too -- with no subjects, which only a structured plan names. With the
check off the planner stays plain text and nothing is parsed.

**Which planner runs, and when the check counts as on** (fix-5vtw, fix-rm98,
fix-7eu6). The planner and the agent now share one decision,
`workflow_agent.finish_check_active(agent)`: a checker attached when the agent
was built, and finish reminders not switched off. The structured planner runs
only for a turn's **initial** plan and only when that is true; replans (after a
parameter-extraction error or an `ask_user` reply) always use the plain-text
planner, and the check still checks the initial plan. The structured call runs
with DSPy's JSON-adapter retry turned off
(`CommandsSystemPreludeAdapter(use_json_adapter_fallback=False)`): that retry
used a plain `JSONAdapter` that dropped the available-commands prelude, so an
unparseable structured reply now goes straight to the text planner -- at most
3 planner calls instead of 6, and `plan_source` `structured` / `text_fallback`
is accurate. The text planner, the agent loop and intent clarification keep
DSPy's default, which still has that retry (tracked as fix-6jzi). (Since
2026-09-28, fix-6jzi: that retry keeps the command list.
`CommandsSystemPreludeAdapter` overrides DSPy's private
`_make_json_adapter_fallback()` hook to return
`CommandsSystemPreludeJSONAdapter`, which puts the same available-commands
prelude in the system message, so a chat-format reply that fails to parse is
retried with the commands visible, in the agent loop, intent clarification and
the plain-text planner alike. Call counts are unchanged. The structured
planner keeps its retry off. The hook is private DSPy API -- checked against
DSPy 3.3.0, while `pyproject.toml` still allows `dspy ^3.0.1` -- and
`tests/test_chat_adapter_commands.py` fails if DSPy renames it.)

**Structured planning is disabled** (2026-09-28, owner decision; supersedes
the structured-planner sentences in the two paragraphs above). The planner is
plain text for every plan, with the check on or off. While the check is on
(`finish_check_active`), the turn's initial plain-text plan is parsed back into
steps by `parse_text_plan` and that is the plan the check verifies: it has no
subjects, so the per-subject questions are never asked of a live plan. Replans
are unchanged. The structured signatures, `STRUCTURED_PLAN_GUIDE`,
`turn_plan.render`, the structured adapter with its retry off and the
zero-steps / parse-error fallback are commented out in
`fastworkflow/workflow_agent.py` and `fastworkflow/turn_plan.py`, not deleted,
and no setting re-enables them. `fw.planner.plan` / `.replan` keep contract v2:
`plan_source` is `text` or `none`, `subjects` is `[]`.
`CommandsSystemPreludeAdapter(use_json_adapter_fallback=...)` and the
command-list-keeping JSON retry (fix-6jzi) stay; nothing in the planner passes
`False` any more.

**`FW_EVAL_FINISH_REMINDERS=0` turns off everything the check adds** (fix-rm98):
no note, and the plain-text planner, so no plan markers. The checker stays
attached, so the `evaluation_controls` event and the `disabled` skip records
are unchanged. It is read like `FW_FINISH_CHECK`: the workflow's
`fastworkflow.env` first, then the process environment (until 2026-09-27 the
process environment only). Any set value other than `0` -- in either place --
makes the agent build fail with `ValueError("FW_EVAL_FINISH_REMINDERS must be
exactly 0 when set")`; an empty value counts as unset and surrounding
whitespace is ignored (an empty value used to raise).

**The plan survives a cross-process resume** (fix-ju1v). With a checker
attached, the turn's plan and its status are session state (the additive keys
`turn_plan` and `turn_plan_status`; no `SCHEMA_VERSION` bump), and a resumed
agent's empty step record is seeded from the suspended trajectory, so a turn
suspended on `ask_user` in one process is checked against the same plan and
the same ledger rows in another. A session state written without the key
resumes with `no_plan_cause` `lost_on_resume`; a stored plan that fails
validation is treated the same way rather than failing the restore. A turn
resumed from a suspension the context-window fallback had already cut is
skipped with reason `ledger incomplete`, because its record lacks the cut
steps. With the check off, the blob carries `turn_plan: null` and
`turn_plan_status: "not_planned"`, and nothing is seeded. Since 2026-09-28 the
same holds for a check attached but switched off with
`FW_EVAL_FINISH_REMINDERS=0`: no trajectory is seeded, no dispatch outcome
recorded and no plan restored.

**The `ask_user` replan sees the request after a cross-process resume**
(fix-ksdu, 2026-09-28; with or without the check). The replan that runs on
the user's reply is built before `resume()` restores the agent's `inputs` and
`current_trajectory`, so in a process that only imported the suspension the
planner used to get `{}` for both -- no request and no trajectory.
`plan_with` now reads `ContinuationReAct.planner_view()`: the live `inputs`
and `current_trajectory` when set (the same objects, so a same-process replan
prompt is byte-identical), else a copy of the suspended stash's `input_args`
and `trajectory`. The stash has no `action_N` keys and carries offload labels,
not raw observations. What is persisted is unchanged.

**Steps that never ran are errors** (fix-lnzw). A step the NLU stage stopped
before any command ran -- a failed parameter extraction, an ambiguous or
misunderstood command, or the framework's own `abort` after one -- is recorded
at dispatch (`finish_check.record_dispatch`, into `dispatch_outcomes`) and its
ledger row shows outcome `error`, the measured vocabulary, instead of being
read from its text. `go_up` and `reset_context` count as ran. The record is
exported with a suspension (the optional react-blob key `dispatch_outcomes`)
and reset at each new turn.

**Identifiers are resolved from any listing shape** (fix-ft18). A ledger row's
`refers_to` labels still come first from `id  label` rows; to them are added
the rows of every listing `listing.parse_table` reads in the step's output
(keyed by first cell), each blank-line-separated block parsed on its own and
only the first `LISTING_PARSE_MAX_BYTES` (64 KiB) of each response. An
identifier is a long token, a parameter value of the command, or a listed
first cell a context clause names (4+ characters or non-numeric, so
"TodoList 1" is not read as item 1). Replayed over 82 recorded ido attempts the
resolved rows are identical to before. JSON outputs still yield no labels.
Since 2026-09-28 the ledger reads listings in `parse_table`'s label mode
(`require_complete=False`): a page of a longer listing (`Page 1 of 3`,
`shown=`, `more rows`) or one group of several still labels its rows, each
group read after the one before it, while a malformed row still refuses its
block. Served rows still come only from a complete listing.

Off unless `FW_FINISH_CHECK=jev` and `JEV_API_KEY` are both set;
`FW_FINISH_CHECK_MODEL` pins the model (default `jev-1.13.0`). **Data egress:**
the plan, the request and the ledger (commands, contexts, subject names, the
head of each output) are sent to TypeSafe with only the capture policy's
credential patterns scrubbed by default. A set `FW_FINISH_CHECK` that cannot
take effect -- a value other than `jev`, `typesafe-sdk` not installed, or no
`JEV_API_KEY` -- logs one warning per cause and the check stays off; an unset
flag is silent. The checker is built once per model and key, so a rotated key
builds a new client. (Since 2026-09-27 the list of causes is longer and the
flag may name a registered provider: see *Decision providers, endpoint and
vendor time*. What is sent -- the subjects' names and kinds, the `refers_to`
keys and `names_in_output` included -- now all passes the same outbound filter,
`jev_client.egress` (fix-hu4f); the note and the stored `flagged_steps` still
use the raw names, which stay in the process.) Each event records
`calibration` (`"ido-v7-2026-09"`): `FLAG_MIN`, `ASK_MIN` and the question
wording were calibrated on ido with the default model and need not transfer,
and a `FW_FINISH_CHECK_MODEL` other than the default logs one warning per
process (fix-ft18).

Each call has a 4-second timeout and the whole check an 8-second budget. The
budget is best-effort: it is checked before each call, not enforced during
one, so a call started just before it runs out can take up to its own
4-second timeout past it. Any failure, and an exhausted budget, means no note.
(That paragraph is history since fix-i94q, 2026-09-27: both are now **hard
wall-clock** cutoffs, and the 8 seconds start **before** the ledger is built,
so building it counts. The call and the check also draw on the turn's shared
10-second vendor budget; see *Decision providers, endpoint and vendor time*.
Building the ledger itself is not cut off: since 2026-09-28 each archive read
it makes waits at most 0.5 s (`LEDGER_READ_TIMEOUT_SECONDS`) for a locked
database, where it used to wait the evidence reads' 30 s, and the first
failed read ends the check as `error` at stage `ledger` -- until then a failed
read left the inline text. Its context-clause reads go through the same
process caches as every other clause read, a "no subject" answer included, so
a second check on the same turn reads no alias from the archive again.)
The check's tokens are recorded on the `fw.finish_check` span and the event,
and are **not** included in the `fw.llm.call` usage or cost totals (the call
goes through the vendor SDK, not litellm).

**The size of a check is bounded** (fix-az9q). A plan naming more than
`SUBJECTS_MAX` (12) subjects is checked step by step only -- "does step k need
a command?" and "was step k executed at all?", at most two questions per step
-- with reason `subjects capped`, and no subject name is sent, not even in
`names_in_output`. (More precisely: the subject list is not sent and
`names_in_output` is empty, but the request, the step texts and the ledger
still are, and they usually name the subjects.) A step missed for one subject among many then goes
unflagged; a step missed for all of them does not. (Recorded turns named at
most 9 subjects.) The ledger, measured as UTF-8 JSON, is kept under
`LEDGER_MAX_BYTES` (96 KiB) by emptying rows' `head` and then their
`refers_to`, oldest rows first; the 118 recorded attempts built ledgers of at
most 52 KB (p95 43 KB), so none is trimmed. A ledger still too large for one
request -- detected from the error body's `max_tokens_exceeded` code, no longer
from the error's text (fix-oivq) -- is halved **once**; later chunks go
straight to the halves, and a half still too large ends the check as `error`
(stage `request`). Until 2026-09-27 it was halved again and again.

**The note's wording** (fix-hfbr, partial; fix-03lt). The head line and the
step list are the measured ones and are unchanged. The tail now reads: "Run any
that are still needed, or say in your answer why not. Do not repeat a change
this turn's record shows was already made, and do not make a change the user
has not confirmed: ask them instead. Skip any step the user has since declined
or changed. You have N steps left." (until 2026-09-27: "Run them if they are
still needed, or say in your answer why they were not. You have N steps
left."). About one flag in four names a step that did run, and the checked plan
is the initial one even after the user answered an `ask_user` question, so the
note now tells the agent not to repeat a change, not to bypass confirmation,
and to skip what the user declined. The tail is not measured. It asks for
confirmation in words only: a framework-owned approval gate is fix-47a9, which
fix-hfbr still waits on. (Since 2026-09-28 fix-47a9 is deferred and fix-hfbr
no longer waits on it: see the next paragraph.)

**Only provably read-only steps are checked** (fix-hfbr, 2026-09-28). A step
the check would hold the agent to is asked about only if every command it and
its parts name is declared `read_only` in the workflow's runtime manifest
(`finish_check.command_effects`: the metadata `register_runtime_metadata`
retained at CLI or FastAPI startup, else `workflow_runtime.json` merged over
the core manifest; a plan's command is matched by its last `/` segment, taking
the most severe kind among the keys sharing that name). A step naming a
`write` command, or one the manifest does not declare (`unknown`), is never
asked about, its text is left out of `plan_steps`, and it is never named in
the note -- so a false flag cannot drive a repeated change or a change the
user has not confirmed. A step naming no command is still checked; "does step
k need a command?" decides it. Undeclared counts as not read-only here, the
safe direction for a note that asks the agent to act (the owner's "treat
undeclared commands as read-only" decision was for fix-47a9's approval gate,
not for this). A missing manifest, one that fails to parse or merge, or a
lookup that raises makes every command `unknown`, so such a workflow has only
its command-less steps checked; nothing raises. The lookup is read from the
bound app workflow when the agent is built. Replayed on the 82 recorded ido
attempts with ido's manifest (cached answers, no new calls): flags on write
steps 7 to 0 (23 before user-gated steps were skipped), precision 0.80, recall
0.88 to 0.87.

Every finish of an agent with the check attached records one `finish_check`
event, with a `reason` even when nothing was asked: `disabled` (finish
reminders switched off with `FW_EVAL_FINISH_REMINDERS=0`), `cap reached` (this turn's note already fired),
`no plan`, `nothing to check` (every step optional or user-gated),
`no room to act`, `error`, `every step executed` or `unexecuted steps`. A
check that ran also records questions, requests, splits, tokens, latency,
`flagged_steps` -- each by step index and subject index into the plan's
subjects, with the subject kind and `p_unmet`, never step or subject text
(fix-7scj) -- and `scores`, one entry per checked step x subject (x part) with
`applies`, `executed` and `unmet`, capped at 200 with the rest counted in
`scores_truncated`. When the check runs and is traced there is also an
`fw.finish_check` span, closed even when the check raises. With the check off
no event is recorded. Install with the `jev` extra.

Since 2026-09-27 the reasons also include `policy_withheld` (the capture policy
withheld a value the check would send, at ledger build or at call time:
nothing is sent, `requests` is 0, no warning, no `error_*` fields; spelled with
an underscore, unlike the others), `subjects capped` (above) and `ledger
incomplete` (above). `subjects capped` replaces `every step executed` /
`unexecuted steps` on a capped check -- `fired` and `flagged_steps` say what it
found -- and `policy_withheld` and `error` take precedence over it. `ledger
incomplete` is decided right after `no plan`. Since 2026-09-28 there is also
`no read-only steps`, decided right after `nothing to check`: some step would
be held to the plan, but every such step names a command not declared
read-only, so nothing is sent. New fields:

| field | on | meaning |
|---|---|---|
| `provider` | every event | the flag value that selected who answers: `jev` or a registered provider's name (fix-5uva) |
| `calibration` | every event | `"ido-v7-2026-09"`, what the thresholds and wording were calibrated on (fix-ft18); on events only, not on the `fw.finish_check` span |
| `vendor_calls_in_flight` | every event, when above 0 | vendor worker slots held when the event was built, abandoned calls included (since 2026-09-28) |
| `vendor_calls_orphaned` | every event, when above 0 | abandoned vendor requests still running after their worker slot was given back (`jev_client.calls_orphaned`); at 8 (`ORPHANED_CALLS_MAX`) new calls are refused as `VendorBusy` (since 2026-09-28) |
| `user_replies` | every event | how many `ask_user` steps the turn's trajectory holds (fix-03lt) |
| `unchecked_for_effect` | every event recorded once the plan is read (from `nothing to check` on) | how many steps the check would hold the agent to were skipped as not provably read-only (fix-hfbr, since 2026-09-28) |
| `no_plan_cause` | `no plan` | `planner_empty` (the planner returned nothing), `plan_unreadable` (no steps could be read from it), `lost_on_resume` (resumed from a session state written without the plan) or `not_planned` (fix-ju1v) |
| `vendor_ms` | a check that ran | the turn's cumulative vendor time, all features together (fix-sotm) |
| `subjects_capped` | a check that ran | 0, or the subject count when the plan was over `SUBJECTS_MAX` (fix-az9q) |
| `ledger_bytes`, `ledger_rows_trimmed` | a check that ran | the ledger's size as sent, and how many rows lost a field to fit (fix-az9q) |
| `error_status`, `error_request_id`, `error_code`, `error_stage` | `error` | HTTP status, the vendor's request id, a single machine-readable token from the error body (never its text), and where it failed: `ledger`, `request`, `budget` (`OutOfTime`), `check` or `note` (fix-conv) |

`error_type` is the error's class name: `CallTimedOut` (a call cut off at its
own cap) and `VendorBusy` (every vendor worker busy, or 8 abandoned requests
still running) are at stage `request`,
`OutOfTime` (the check's or the turn's time ran out) at stage `budget`. A
`build_ledger` failure is logged with its traceback. The span's attributes and
contract version are unchanged.

### Decision providers, endpoint and vendor time

Both decision-model features -- the router and the finish check -- share this
plumbing (`observation_offloading/jev_client.py`, `decision.py`). All of it is
new on 2026-09-27; with `FW_FINISH_CHECK` and `FW_SEARCH_ROUTER` unset, none of
it runs except the SDK log filter.

**Activation.** A set flag that cannot take effect logs one warning per cause
per process and the feature stays off. The causes are now: a value that is
neither `jev` nor a registered provider's name; `typesafe-sdk` not installed;
no `JEV_API_KEY`; a rejected `FW_JEV_BASE_URL` (fix-2so9); a capture profile
that withholds command output -- one whose policy returns a badge for an
opaque-payload value on the offload-observation path, such as `evidence` --
or a `FW_OBS_CAPTURE_PROFILE` naming an unknown profile; and
`FW_OFFLOAD_EVIDENCE_REDACTION=off` (fix-7tz5). A declared field policy that
keeps observations whole is still allowed. The capture-policy causes read
"`<FLAG>=jev but <cause>; <feature> stays off`" and are checked last, so an SDK
or key problem is reported first. Every value sent then passes
`jev_client.egress`, which returns nothing to send when the stored form is or
carries a capture badge, or when redaction is off: the per-value backstop
behind `policy_withheld`.

**Endpoint** (fix-2so9). Both features send their key and payloads to
`FW_JEV_BASE_URL`, default `https://api.typesafe.ai` (the SDK's own default),
read from the env file first, then the process. It must be `https`, or plain
`http` only to a loopback address (a local stand-in), with no credentials,
query or fragment and a well-formed port; anything else logs one warning that
does not include the URL, and both features stay off. A non-default host is
logged once at INFO. The SDK's own `TYPESAFE_BASE_URL` is **ignored**, with one
warning: the client is always built with an explicit endpoint, so a variable
in the process environment can no longer redirect the key and payloads.

**SDK logging** (fix-qgeu). The SDK logs whole request and response bodies,
unredacted, at DEBUG. A filter attached to the `typesafe_sdk` logger when
`jev_client` imports the SDK drops those records whatever the log levels,
keeping its INFO status line (status, latency, request id). There is no opt-in.

**Failures** (fix-conv). A failure is warned about at most once per five
minutes (`FAILURE_WARN_INTERVAL_SECONDS`) per (feature, error type, HTTP
status), saying how many more like it were not logged since the last warning;
until 2026-09-27 each kind was warned about once per process, and the note's
own failures without limit.

**Vendor time** (fix-i94q, fix-sotm). The SDK's timeout is per network phase,
so a body trickled in slices, or a slow resolver, outlasted it many times over.
Every call now runs on its own daemon worker thread, at most `VENDOR_WORKERS`
(4) process-wide, and the caller waits at most `min(cap, time left)`: the cap
is 2 s for a routing call and 4 s for a finish-check call; the time left is the
smaller of the caller's own deadline (the finish check's 8 s) and the turn's
`TurnBudget`, `TURN_VENDOR_SECONDS` (10 s) of time actually spent waiting on
vendor calls by every feature together. Past that the request is abandoned and
the caller fails open; it is never retried and never delays interpreter exit.
A call with under 0.05 s left is not sent. When all four workers are busy a
call fails at once with `VendorBusy` ("vendor busy") instead of queueing. The
SDK's own timeout is set to the cap plus 0.5 s, so the cutoff always decides.
(Since 2026-09-28 every call passes it explicitly -- a call cut at its own cap
included, the client's own timeout kept when shorter -- and an abandoned call
gives its worker slot back at most 0.5 s after its cutoff even when a trickled
body keeps the request running; such requests are counted by
`jev_client.calls_orphaned`. Their threads and sockets are bounded instead:
while `ORPHANED_CALLS_MAX` (8) of them still run, every new call fails at once
with `VendorBusy` and sends nothing, until they end.)
A new turn gets a fresh budget and a resume in the same process keeps it. The
limits are module constants; there are no settings for them.

**Providers** (fix-5uva, stages 1-2). The features ask vendor-neutral questions
(`decision.YesNo`, `decision.OneOf`) of a `decision.DecisionProvider`; Jev is
the built-in one (`jev_client.JevProvider`). Code already running in the
process may register another with
`decision.register_decision_provider(name, factory)` (and
`unregister_decision_provider`) and select it by setting `FW_FINISH_CHECK` or
`FW_SEARCH_ROUTER` to its name; nothing reads a module path from the
environment. Names are case-insensitive; `jev`, the off values and malformed
names are refused. The factory takes no arguments, is built once per
registration and shared by both features; one that raises, or returns
something that is not a provider, warns once and the feature stays off.
Selecting a registered provider warns once per flag that the thresholds and
questions were calibrated with Jev, and with one `JEV_API_KEY`,
`FW_JEV_BASE_URL` and the `*_MODEL` flags are not read (the event's `model`
comes from the provider's `model` attribute). The capture-policy gate applies
to every provider, including one with `third_party=False`. A LiteLLM-backed
provider (stage 3) is deferred. The request bodies Jev receives are pinned by
a golden test (`tests/test_decision_provider.py`).

**The SDK is pinned exactly** (fix-em0l): `typesafe-sdk = "0.7.2"`, so
`pip install 'fastworkflow[jev]'` no longer picks up a later 0.7.x. A bump must
rerun the SDK contract test (`tests/test_jev_client.py::test_the_sdk_surface_fastworkflow_uses`)
and the golden request-body tests in `tests/test_decision_provider.py`.

## Retention, redaction and known limits

Offloading writes evidence to disk and bounds several things by bytes. What
follows is the contract as it ships, including the places where it is looser
than a one-line summary would suggest.

**Where the evidence lives.** In the workflow's own observability database,
`<FASTWORKFLOW_STATE_ROOT>/workflows/<workflow-id>/observability.sqlite3`, the
same file as its turn records and spans. Three tables, added to the store's
schema with feature markers (`offload_evidence_v1`, `offload_events_v1`)
rather than a schema-version bump:

- `offload_evidence` — one row per archived observation (and per archived
  search answer): the stored bytes, their digest, and the capture record that
  produced them (policy version, profile, redaction mode, whether the stored
  bytes differ from what the command returned, and the raw byte count);
- `offload_subjects` — the context each observation is evidence about;
- `offload_events` — the offloading runtime's diagnostic events.

Every row is keyed by the TURN that produced it and carries that turn's
`channel_id`, like a span or an artifact. The database is created owner-only
(the file `0600`, its directory `0700`) whichever code path opens it first, and
no setting turns recording off, for fastWorkflow's own entry points and for
programs that embed the library alike.

**Evidence lives and dies with its turn.** `forget_channel`, Clear
conversations and retention pruning delete a turn's evidence, subjects and
events in the same transactions that delete its turn record, and drop the
process-local copies that could still serve them. There is no preservation
mode: an experiment run's evidence is erased by a Clear or by forgetting its
channel, exactly like a chatbot conversation's.

**Redaction happens when the evidence is written.** With
`FW_OFFLOAD_EVIDENCE_REDACTION=on` — the default — a command response is stored
as the trace sink's credential scrub and capture policy leave it: it redacts, it
does not truncate. Event text is protected the same way, because events carry
search questions, reasoning and answers. `off` stores responses and events
verbatim, and each row says which mode produced it. That is the developer
setting, for reproducing exactly what the agent read.

Redacting at the write would change what the agent reads back mid-turn, so the
process that wrote a redacted row also keeps its raw text in memory while the
turn is live, and every read that process makes during the turn — the
trajectory, `search_memory`, answer-time rehydration — is exact. That memory is
released when the turn is over: when the agent starts the next turn, or when the
session closes. A suspended turn keeps it. A turn resumed in a *different*
process has no such memory and reads the stored, redacted text; that is the
accepted cost of never writing raw bytes to disk.

**Subject clauses are not redacted.** `offload_subjects` holds a context name
and an instance label rather than command output, and is stored in the clear.
If your context labels can carry anything sensitive, treat that table as
unredacted.

**Older evidence files are deleted, not imported.** Earlier builds kept the
evidence in a separate file beside the database, named with an
`.offload-handles.sqlite3` suffix. Opening the store deletes that file, its
write-ahead-log files and any `.preserve` marker beside it; nothing in it is
carried over.

**Pruning runs once per process start.** The evidence is pruned on the store's
own age horizon and size cap (`FW_OBS_RETENTION_DAYS`, `FW_OBS_DB_MAX_BYTES`),
one whole turn at a time. That prune is triggered when a trace sink opens the
store, and again the first time the offloading archive opens a database in a
process, so a database no sink ever opened is still bounded. A long-lived
process does not prune again while it runs.

**A program that embeds the library gets the same record.** A
`WorkflowExecutionContext` built without a sink opens the bound app workflow's
own sink when `bind_app_workflow()` runs — the same sink, and so the same
prune, fastWorkflow's entry points open — and moves it to the new workflow's
database when it is rebound to another workflow. A sink the caller passes, to
the constructor or to `set_trace_sink()`, is always kept. Passing
`tracing.NoOpTraceSink()` records no spans and no turn records, but it does
not turn offloading off: the evidence, subjects and events above are still
written (redacted as configured) to the workflow's `observability.sqlite3`,
under `FASTWORKFLOW_STATE_ROOT`, and pruned when the archive opens it.

**Worst-case agent work in one turn.** A turn runs at most four segments of 25
decisions each, plus the three continuation-planner calls that open the second,
third and fourth segment: 100 tool-or-finish decisions and three extra model
calls before the turn is forced to answer. (Three segments, 75 decisions and two
planner calls until 2026-09-27.) Size provider spend and request timeouts against that
ceiling rather than against a typical turn.

**A garbled model reply after a tool has run fails the turn.** When the provider
returns a reply the adapter cannot parse, the call is retried only while the turn
has executed nothing. Once any observation exists the turn fails instead, because
replaying the trajectory would re-run commands that already ran.

**The continuation byte measure reports overage, it never enforces it.** The
measure that decides when a segment is over its trajectory target counts
observation bytes only, so the framing around them and the thought and tool
fields are outside the number. The prompt can therefore run a few hundred bytes
over the target, and the runtime records the overage rather than trimming to fit.

**The rehydration control note is added after the budget.** When rehydration
stops at the 250,000-byte extraction budget, the line naming the observations it
could not put back is appended afterwards rather than reserved inside it. It
costs a few hundred bytes at most, and it is worth more than the evidence those
bytes would have bought.

**The bounded-evidence notice is added after the evidence budget.** On the same
terms: when the observation handed to the search model was cut to its byte bound,
the one-line notice saying so is appended after the bound. The model input can
therefore exceed the evidence budget by roughly two hundred bytes.

**After a restart, one search answer can come back over its bound.** Bounding an
answer requires archiving the complete text first, under a key numbered by a
per-scope counter that lives in process memory. A turn resumed in a fresh process
restarts that counter, so the write can collide with a key the earlier process
already used; the archive refuses it, and an answer that cannot be archived is
returned inline whole rather than cut — past the 3,072-byte presentation bound.

**An evicted suspended session keeps a little memory until the process exits.**
When the session manager evicts a suspended session, the per-session bookkeeping
on the offload path is not freed with it. Both parts are capped — the hot
observation cache by bytes, the in-process event buffer at 2,000 events — so the
residue is small and bounded per session, but it is held until the process exits.

**An abandoned vendor request may still be billed.** A decision-model call cut
off at its hard wall-clock bound is abandoned, not cancelled: the request may
still complete vendor-side, and cost tokens, after the turn has moved on. It is
never retried, and its daemon thread holds one of the four worker slots until
it ends by itself. (Since 2026-09-28 the slot is given back at most 0.5 s after
the cutoff; the thread may run on, counted by `jev_client.calls_orphaned`. While
8 such threads run, new vendor calls are refused as `VendorBusy`: a vendor
endpoint that keeps trickling can switch both features off until it stops.)

**A turn resumed in another process starts a fresh vendor budget.** The
per-turn 10-second vendor budget and 3-call routing cap live in the agent, not
in the suspended session state, so a cross-process `ask_user` resume can spend
another full budget.

**Broad scopes still read other channels' evidence by alias.** Under the
process-default scope and the between-turns fallback, `list_summaries`
enumerates nothing (fix-y570), but the archive's `get()` and `list()` are still
keyed by `(turn_key, alias)` alone, so a known alias resolves across channels
(fix-tyzj, open). (No longer true since 2026-09-28: fix-tyzj scopes every
archive read and subject write to the channel as well; see the broad-scope
hint above.)

**The finish check's thresholds were calibrated on one workflow.** `FLAG_MIN`,
`ASK_MIN`, the question wording and the published precision and recall come
from ido attempts with the default Jev model (`calibration` on every event).
Another workflow, model or registered provider is unmeasured.

## Validation

`MinimumOffloadSaving` covers the 1 KB rule: savings of exactly 1,023 / 1,024 /
1,025 B, the saving taken from the step's real label rather than a constant (same
bytes of output, two command arguments, one offload), an authored description
lengthening both label and decision, a 3 KB listing page older than the protected
five offloaded when over target, a 1.2 KB result kept, short facts untouched, the
recency five protected before any label is built, the printed alias line excluded from
the measured saving, the eager archive holding both the kept and the offloaded
observation, search answers still never offloaded, the environment override in
both directions with bad values falling back, and the replan skeleton applying
the same minimum.

Focused tests cover labels, byte/character savings, replan persistence, scoped
resolution, required keys, current-step reasoning, archive eviction and existing
continuation behavior. `PrintedObservationHandles` covers the printed handle:
interleaved tools, truncation via `ordinal_offset`, the replan skeleton, the
inline/label/archive alias being one identifier, savings accounting with the
added line, and an unknown handle staying an explicit miss.
`EagerObservationArchive` covers the eager archive: observations archived whether
or not they are offloaded, an inline and an offloaded search receiving the same
bytes and digest, eviction and a cleared cache resolving from SQLite, repeated
persistence keeping one row, another turn's alias staying invisible, a failed or
conflicting write keeping the inline evidence and recording `archive_refused`,
the `still_inline` flag, archiving under the printed alias in the
`alias_conflict` case, and the replan skeleton persisting without a second row.
`BoundedSearchAnswers` covers the search output bound: every recorded answer size
presented unchanged, a long answer bounded, marked and archived whole, the cut
landing on a line boundary with identifiers intact, a newline-free answer cut on
a character boundary, the observation fitting the budget at every admissible
bound, a bad or too-small bound falling back to the default, the full answer
retrievable by the key the marking names and invisible to another scope, repeated
searches keeping one record each, the record key rejected by `search_memory` and
never offered as a handle, a failed answer archive keeping the complete answer
inline, a bounded observation getting no `O` alias and not being archived as one,
and the packed and replan cost of a search staying inside the budget.
Provider tests are opt-in:

```bash
FW_TEST_OBSERVATION_SEARCH_LIVE=1 python -m pytest \
  tests/test_observation_offloading.py tests/test_observation_search.py
```

Configure the search model and credentials before enabling provider tests.
Without the opt-in, deterministic integration checks run and paid cases skip.
