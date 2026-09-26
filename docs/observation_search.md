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

> Use search_memory tool to search inside Observation O8 returned by show_holders. It was offloaded to memory and contains identity_uid: identities holding this permission; label: their display names.

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

> Observation O42 (execute_workflow_query)
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

> Observation O22 (execute_workflow_query, in Account e8a0c3a1-… Alan Cooper)
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

**The instance identity is declared, never derived.** fastWorkflow has no notion
of a context instance's identity — the current context is an arbitrary
application object — so `fastworkflow/context_identity.py` reads a declaration
off the same context callback class that already declares `get_parent` and
`enter_command`: a classmethod `instance_label(command_context_object) -> str`,
or an `instance_label_attr = "uid"` naming an attribute to read. A context that
declares neither prints its NAME alone, and an object that carries no identity
yields no identity: nothing is invented to fill the gap, for the reason
`tracing.context_handle` gives for refusing to mint an `instance_key` — a guess
that looks concrete is worse than an honest absence. The root context prints no
clause at all, so a root-context line is byte-for-byte the plain handle line
shown earlier.

`context_clause` is the one place the clause is made printable. It removes
parentheses and newlines and caps the name at 60 and the label at 80 characters,
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

Observation offloading itself has no switch: `build_tool_agent` always returns a
`StructuredContinuationReAct` with `search_memory` in its tools, and the forced
replan bound is the module constant
`observation_offloading.continuation.MAX_FORCED_REPLANS` (2, therefore 3
segments). The observation archive always lives in the workflow's own
observability database. So do the offloading runtime's diagnostic events: they
are kept in process in a ring of the newest 2,000 (`snapshot_events()`) and
stored as rows of `offload_events`, read back with
`ObservabilityStore.offload_events(turn_key=..., channel_id=..., kind=...)`.

The one setting that remains is `FW_OFFLOAD_EVIDENCE_REDACTION`: `on` (the
default) or `off`. See *Retention, redaction and known limits*.

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
`LLM_OBSERVATION_SEARCH`'s context window and takes 3/128 of it as bytes, which
is 12,288 B at the 131,072-token reference window — exactly the declared
geometry of `DEFAULT_PAGE_BYTES` (4,096) x `SEARCH_MEMORY_MAX_PAGES` (3) — with
a floor of one page. It has no tuning override: the search model's window is
the only input, and `FW_MODEL_CONTEXT_TOKENS` is how a deployment corrects that
window. The subject metadata is paid for out of that same budget, so nothing
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

**Worst-case agent work in one turn.** A turn runs at most three segments of 25
decisions each, plus the two continuation-planner calls that open the second and
third segment: 75 tool-or-finish decisions and two extra model calls before the
turn is forced to answer. Size provider spend and request timeouts against that
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
