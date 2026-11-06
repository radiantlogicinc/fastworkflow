# Policy verifiers: design

## Problem

Distillation compares a run that went wrong with one that went right and writes what it
learns into `Insights/` as prose. That prose is appended to the agent's and planner's
prompts. A prompt is advice: a small model reads "read every order before changing one"
and still changes the first order it sees. On recorded failures the same few shapes
recur across tasks:

- a write before every candidate record was read;
- an argument value the command rejects;
- a write that changes nothing;
- an answer that claims work no command performed.

Each shape is decidable from what the agent loop already holds: the commands run, their
results, and what the user said.

A **policy verifier** states such a shape as a predicate and checks it while the agent works.
When it holds, the step is not taken and the agent reads the rule's text instead.

## Model

- **A table, not scattered conditions.** `DecisionTable(version, rows)` is ordered and
  versioned, and first match wins. A `PolicyRow` has an id, the phases it applies to, an
  outcome, a reason, a pure predicate, and a rewrite (text, or a function that names what
  was wrong). Every decision records the row id and table version, so a behaviour change is
  attributable to one rule of one table.
- **Deterministic.** Predicates make no model call. Where a fact lives only in the user's
  words, a model *reads* it into the rows' inputs, and the rows still *decide*:
  - what was asked for, mapped onto the workflow's declared operations;
  - whether a message consents to the assistant's last proposal;
  - optionally, whether a question repeats an earlier one.
- **Rewrite, do not refuse.** The rewrite becomes the step's observation. A refusal costs an
  iteration and is usually re-proposed; a rewrite says what to do instead.
- **Facts are declared.** `ContractFacts`, supplied with the table, says:
  - which commands write;
  - the identifier shapes (defaulting to the workflow's `_commands/_split.py`
    `identifier_patterns`);
  - the closed argument domains;
  - the operation vocabulary;
  - a `failure_pattern` for tools that report errors as successful text.

  The engine knows no workflow.
- **No switch.** A workflow that has `_policy/table.py` has its rules enforced; one without
  it is byte-for-byte the unmodified agent. What a table *would* do is answered offline by
  `fastworkflow policy replay`, on recorded runs.

## Phases and where each is checked

| phase | known then | hook |
|---|---|---|
| `BEFORE_SIDE_EFFECT` | the command the model chose, its real name and validated parameters | `command_executor._invoke_command_impl`, after intent detection and parameter extraction, before the response generator runs |
| `AFTER_OBSERVATION` | a non-command tool the model chose (a question to the user) | `_run_loop`, before the tool is dispatched |
| `BEFORE_FINISH` | the answer | `_run_loop`, when the agent picks `finish` and after `finish_reviewer` passed |

**Why commands are checked at the executor.** That is the only place the command exists in
canonical form. In the loop it is the agent's text: tags or no tags, a misspelled name, a
JSON list as a string. Rules written against that text misread it. A block returns a failed
`CommandOutput` with `artifacts["policy_blocked"]`. Nothing executes. The command tool keeps
it out of the action log, and distillation's trace reader leaves it out of the actions.

**Which commands are checked.** Only those the model chose in a loop step. The loop sets
`AgentPolicy.dispatching` around that call. So these are not checked:

- a command the user typed (`/command`);
- a command a skill's code runs (`code_steps`, which are the author's logic);
- the framework's internal abort;
- assistant mode and MCP direct actions (`perform_action`).

**Finish.** The answer is extracted when the agent picks `finish`, and the table checks it.

- If a row fires, the rewrite is that step's observation and the loop reopens, as for the
  existing finish reviewer.
- If none fires, that extraction is kept (`run_state["policy_extract"]`) and becomes the
  run's answer, so a passing check costs no extra model call.
- The upstream `finish_reviewer` (plan coverage) speaks first; the policy speaks only when it
  had nothing to say.
- A run that is not the turn's last is not checked (`policy_final_run`, set alongside
  `_attach_finish_check`): a non-final sub-task, a non-final skill phase, or a non-final
  fan-out item. Otherwise sub-task 1 would be blamed for sub-task 2's request.
- A run that hit its step cap never picks `finish`, so it is exempt.

## Bounds

A rule that is wrong about the situation must not be able to trap the agent:

- **Per-rule valve.** A rule blocks the same kind of step (rule, tool, command) at most twice
  per conversation; the third attempt goes through.
- **Finish budget.** An answer is pushed back at most three times per turn, and again only
  after progress since the previous push: a successful write, or a new user reply.
- **Fail open.** A predicate that raises, a model reading that fails or exceeds 25 s, or a
  guard that errors all let the step through, and the failure is logged.

## State

`AgentPolicy` (one per agent, i.e. per conversation) keeps a `PolicyState`:

- every command that ran, with its parameters, response and success;
- every user message (`@user`) and question/answer pair (`ask_user`), each with its consent
  reading;
- the dialogue, the intent reader's request list, and the valve counts.

It is conversation-scoped: a record read in an earlier turn is still read.

It is recorded from the points where each fact first exists:

| fact | recorded in |
|---|---|
| a turn's message | `_run_agent` |
| a reply to `ask_user` | `_post_ask_user_response`, which both topologies route through |
| an executed command | `_execute_workflow_query` |
| the answer | the loop's `_final_extract` |

It crosses processes in `WorkflowExecutionContext.serialize_state` as `policy_state`. That
key is present only when a policy is enabled, so states written without one are unchanged.
Each distillation pass's fresh agent starts from the session agent's state.

## Tracing

`fw.agent.step` v8 adds these keys on a step a rule rewrote:

- `policy_source`, `policy_outcome`, `policy_mode`, `policy_table_version`;
- `policy_intent`, `policy_requests`;
- `policy_judgement` (on `ask_user` steps);
- `answer` (on `finish` steps).

The aggregate `SPAN_CONTRACT_VERSION` is 24. The debug UI shows a **Policy** section on such
steps.

## Configuration

The table is found, not configured: `<workflow>/_policy/table.py`, exposing `TABLE` (a
`DecisionTable`) and optionally `FACTS` (`ContractFacts`). `_policy`, like `_commands`, is a
framework folder.

A table file that exists but is malformed fails at agent construction: rules the developer
wrote must not be skipped without a word.

The only optional setting is the model for the readings:

| setting | meaning |
|---|---|
| `LLM_POLICY_INTENT` / `LITELLM_API_KEY_POLICY_INTENT` | model for the readings; `LLM_AGENT` when unset |

## Producing a table

Run the loop described in the `distill-policy-verifiers` coding skill:

1. Record trials as experiments.
2. Distil: `fastworkflow distill` with teacher/student, `--self`, or
   `--pool-students/--pool-teachers`; `--provenance` records which pair produced which
   insight.
3. Cluster the insights into failure modes (`insights-to-failure-modes`).
4. Compile `_policy/table.py`.
5. Audit it with `fastworkflow policy replay`.
6. Ship it: the table is on as soon as `_policy/table.py` exists.
7. Compare against the same workflow without the table (a control arm).

The replay drives the same `AgentPolicy` over recorded spans:

- the turn message and its `fw.ask_user` reply;
- canonical commands from `fw.command.execute` under each step's tool call;
- the recorded intent, or with `--live` a fresh reading.

So its verdicts follow the live rules. `fastworkflow/examples/retail_workflow/_policy/` is a
complete worked example: corpus, provenance, taxonomy, table and tests.

## Files

| file | role |
|---|---|
| `fastworkflow/policy_decision.py` | phases, rows, table, decision point, facts, phase inputs, `PolicyState` |
| `fastworkflow/policy_runtime.py` | `AgentPolicy`: recording, the four decision points, valve, finish budget, persistence |
| `fastworkflow/policy_intent.py` | intent/consent reader and question judge (model readings, fail-open) |
| `fastworkflow/policy_loader.py` | settings → decision point and facts; `attach_policy` |
| `fastworkflow/policy_replay.py` | offline replay of recorded runs |
| `fastworkflow/utils/react.py` | loop hooks: `policy`, `policy_final_run`, `_policy_finish`, `_final_extract` |
| `fastworkflow/command_executor.py` | `_policy_rewrite` at the canonical seam |
| `fastworkflow/distillation_compare.py` | `compare_self`, `compare_pool`, `pool_pairs`, `provenance_records` |
