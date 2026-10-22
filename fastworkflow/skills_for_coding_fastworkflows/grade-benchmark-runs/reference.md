# Grading reference: templates, schema, example, helper scripts

Companion to [SKILL.md](SKILL.md). Everything here is a template. Replace the toy library-loans
workflow (members, loans, holds, notes) with your own records, commands and tiers.

## Contents

- [Cause and tag vocabulary](#cause-and-tag-vocabulary)
- [Checklist file](#checklist-file)
- [Brief template](#brief-template)
- [Per-stage addendum template](#per-stage-addendum-template)
- [Grader prompt](#grader-prompt)
- [Grade JSON schema](#grade-json-schema)
- [Worked example](#worked-example)
- [Grader reliability](#grader-reliability)
- [Trace dump script](#trace-dump-script)
- [Validate and total](#validate-and-total)
- [Losses by cause and stage](#losses-by-cause-and-stage)

Per-stage attribution and the metrics to report are in [pipeline-and-metrics.md](pipeline-and-metrics.md).

## Cause and tag vocabulary

This is the one closed list. The brief quotes it, graders use only it, and `check_grades.py` and
`losses.py` below encode it. Add a workflow-specific `other:` name to all three at once, and only
after it has recurred (three times in one stage).

| Cause (`cause` field and `[cause:…]`) | First wrong thing | Stage |
|---|---|---|
| `split` | A sub-task asks for something other than what the user asked | split / merge |
| `bad-match` | Faithful sub-task, wrong skill (or none where one fits and the text plan failed) | skill match |
| `wrong/plan` | Plan or skill arguments wrong for a right job | plan |
| `wrong/agent` | Plan right; the agent did otherwise, stopped early, or stopped paging | executor |
| `wrong/argument` | Right command; wrong record, uid or option | executor |
| `wrong/command-failed` | Right command errored; nothing recovered | executor |
| `budget/real-work` | Budget used up by productive steps (otherwise, use the cause of the waste) | executor |
| `ask` | Asked what the prompt answered, or ended with a question | executor |
| `fabricated` | Made up, with nothing earlier wrong | answer |
| `unsure a\|b` | Undecided between two causes from this list | — |
| `other:skill-arg:<param>` | A skill-call argument wrong (use with `wrong/plan`, or alone) | plan |
| `other:skill-gap` | The skill's expansion lacks a step the request needs | plan |
| `other:skill-fallback` | A skill call fell back to the text planner (`skill_fallback`), and the text plan failed (use with `wrong/plan`) | plan |
| `other:phase-when` | A phase skipped by a `when` that misread the input | plan |
| `other:phase-failed` | A phase's facts could not be filled, and the job stopped (`facts_error`) | phases |
| `other:phase-facts` | Wrong or missing facts passed between phases | phases |
| `other:fan-out-cap` | Fan-out items cut by the cap (`items_total` > `item_count`) | phases |
| `other:flat-phase` | A loss inside a flat (single-phase) skill run | executor |
| `other:plan-followed` | Code followed the plan exactly where an agent used to improvise a fix | executor |
| `other:non-target:<cmd>` | The answer came through another command than the target | executor |
| `other:finish-note` | A finish-check note sent the agent somewhere harmful | finish check |
| `other:answer-phase` | Complete facts, poor answer-only writing | answer |
| `other:answer-by-code` | Code-written answer lacking what its step text lacked | answer |
| `other:premise-repeated` | The answer repeated a false claim from the prompt over correct outputs | answer |
| `other:truncated-output` | An output was cut, and the cut hid what was needed | commands |
| `other:state-not-restored` | A write task did not start from its baseline | run setup |

Other tags, closed too:

- `[split:dropped|joined|invented|referent|header|harmless]` (`harmless` is a note, not a loss);
- `[finish-note:helped|ignored|harmful|none]`, and the `finish_note` field, which also allows
  `mixed` across several sub-tasks;
- `[fabricated:yes|no]`;
- `[grade a/b]`;
- `[item:x]`;
- `[skill:<name>]`.

Marks: ✅ ◐ ⚠️ ❌ 🚩 ⛔ and `n/a` (unscoreable). The checker treats "⚠" and "⚠️" as the same mark.

## Checklist file

Keep one owner-approved file per benchmark (or tier). It is versioned with the app and frozen
once approved. Give every task a heading, its line count and its lines. Give a single-command tier
a scoring rule plus a target table.

```markdown
# Checklists: library benchmark (approved <date> by <owner>)

| tier | tasks | lines per run |
|---|---|---|
| T1 one command | 40 | 40 (1 per task) |
| T2 one record, several asks | 12 | 41 |
| T5 working sessions | 6 | 88 |

## T1 scoring rule
One line per task: "the target command ran on the named record and the answer reports its output
correctly". Navigation (open_*, find_* then open_*_by_uid, go_up, reset_context, what_can_i_do)
never counts. A right answer through a non-target command: <owner's rule>, always tagged
[other:non-target:<cmd>] and target_ran "other:<cmd>".

| task | target command (context) | note |
|---|---|---|
| L1-07 | Member/list_loans | "every loan": first page only is ◐ |
| L1-12 | Member/add_note | write: the member entered last before it; restore after the round |
| L1-19 | Directory/find_member | two members share the name: list both, no ask, no pick |

## T2
### L2-03 overdue-member-check-ann-lee: 4 lines
a. every loan Ann Lee has out · b. which of them are overdue · c. her open holds ·
d. limit: renew nothing
```

`check_grades.py` needs the frozen sizes as `task<TAB>tier<TAB>lines`. Write that file from the
approved checklist file when it is frozen, by counting each task's lines, and version it with the
checklist. Never derive it from a stage's grades, because that would accept whatever line counts
the graders wrote.

## Brief template

```markdown
# Grader brief: <workflow> benchmark, <tiers>, <runs> runs per task

You grade recorded runs of <workflow>, a fastWorkflow app for <domain>. It was run <unattended:
no ask_user tool | attended>, with <model routes> in every role and the LLM cache off. Grade each
run on its own, with the same checklist.

Read in this order:
1. The checklist file <path>: your tasks' lines. Never change lines or line boundaries.
2. Your dumps <dir>/<run>/<tier>/<task>.md: the prompt, the final answer, and the trace in time
   order. A long output is cut; the note names the span. To read the full value, open the store
   READ-ONLY:
   sqlite3 "file:<db>?mode=ro" "select attributes from spans where span_id='<id>'"
3. For comparison only: the earlier stage's grades <path> (fields task, run, score,
   turn_comment). Grade independently, then compare.

Method:
- The graded answer is the final answer (turns.answer). If the turn was split, check that the merge
  dropped and invented nothing relative to the sub-task answers.
- Trust the trace, not what you know about the data. Every name, id, count and value in the answer
  must be in a command output: cite the FULL 32-character span id (copy it whole from the dump)
  in the line's evidence. Not found anywhere: 🚩.
- Marks: ✅ ◐ ⚠️ ❌ 🚩 ⛔, and n/a only for lines the checklist file marks unscoreable.
  Score = (✅ + ½◐) / scored lines. Pass = all scored lines ✅.
- Cause = the FIRST mistake in time order (split → skill match → plan → phases → executor →
  finish check → answer), judged against each stage's own input. Use ONLY the causes and tags in
  <vocabulary link>, spelled exactly. Write each [cause:...] immediately followed by every
  [item:x] it explains.
- Always give [fabricated:yes|no] and [finish-note:helped|ignored|harmful|none].
- Navigation never scores. Wasted-but-recovered steps on a passing turn are not a cause.

Judgment calls already made (apply them the same way):
- A misread value that appears in some output is ⚠️, not 🚩.
- A partial read (some pages) is ◐ on an "every" line, and ⚠️ plus [fabricated:yes] when the answer
  states a total the pages read do not support.
- A comparison answered for one side only is ◐.
- A name with a prefix or format difference that clearly identifies the same record is ✅.
- An answer that says a write happened when no write succeeded is 🚩.
- Fetched but not reported is ❌.
- Answers written by code are literal, one line per step: grade content, not style.

Hard limits: open the store read-only only. Write exactly the one grades file you are told to.
No helper scripts, no temporary files anywhere: use inline one-liners that print. Never run the
agent, never call the backend, never edit a repository file.

Output: a JSON list, one object per (task, run) plus one summary object per task, in the schema
<link>. Copy turn_key exactly from each dump's first line.
```

## Per-stage addendum template

```markdown
# Stage <n> grading addendum (read after <brief> and <the previous addendum>)

All of it applies, with these differences:

- Dumps: <dir>/R{1,2,3}/t{1..5}/<task>.md. Copy turn_key exactly from each dump's first line.
- Compare with STAGE <m>: <grades path>, same task, all runs. Field "vs_stage<m>" on each run row,
  "vs_stage<m>_summary" on the summary row.
- Same checklist lines as stage <m>. Rule changes: <none | "<date>, owner: <rule>; keep the tag
  [other:...] so it can be re-applied">.
- New since stage <m> (what you will see in the traces):
  - <A new skill and its phases; what its answer looks like (code-written, literal)>
  - <A refusal or check the framework now makes, and its message text>
  - <A behaviour that changed on purpose, so it is not marked as a mistake>
- New tags: <[other:xyz] for ...>.
- Known run caveats: <first job of each worker stalled: ignore durations>.
```

After about five addenda, fold them into a new brief and start the chain again.

## Grader prompt

One prompt per group. Keep it short: the brief and the addendum carry the method.

```text
You are a grader for stage <n> of the <workflow> benchmark. Read <addendum path> first and follow
it, and every document it points to, exactly.

Your group: tier <t>, tasks <first> to <last>. Dumps: <dir>/R1/t<t>/, <dir>/R2/t<t>/,
<dir>/R3/t<t>/. Grade every task of your group for every run, then one summary row per task.

Write exactly one file: <dir>/grades_<group>.json (a JSON list of row objects). Check that it
parses with an inline one-liner. Then reply briefly: the score per task per run, any 🚩 or ⛔ rows,
and every lost line with its cause tag.
```

## Grade JSON schema

One object per (task, run):

| Field | Type | Meaning |
|---|---|---|
| `task` | str | Task label as in the checklist file |
| `run` | str | `R1`, `R2`, … or `summary` |
| `turn_key` | str | Copied from the dump's first line |
| `checklist` | list | `{item, line, mark, evidence}`; `evidence` = "answer vs output (full span id)"; `mark` from the vocabulary, including `n/a` |
| `score` | str | `"a/b"` over scored lines (`n/a` excluded); a may be a half |
| `pass` | bool | Every scored line ✅ |
| `fabricated` | `"yes"`/`"no"` | Any 🚩, or a stated total over a partial read |
| `split` | str | `not split`, `faithful`, `harmless over-split`, or `bad: <why>` |
| `cause` | str or null | The earliest cause, from the vocabulary; null on a pass |
| `cause_item` | str or null | The line it first cost |
| `first_wrong_span_id` | str or null | Where it first went wrong (full span id) |
| `finish_note` | str | `helped`, `ignored`, `harmful`, `none` or `mixed` |
| `turn_comment` | str | Tags first, exact spellings, then at most about 600 characters of explanation |
| `vs_stage<m>` | str | One sentence: better, worse or same than the earlier stage's runs, and why |
| `judgment_calls` | list[str] | Every call another grader might make differently, and every rule conflict |

Add fields the workflow's features need, and keep them in every later stage. Examples:

- `skill` and `phased`, and `phases` as a list such as `["find ok 2 (by code)", "answer ok 0"]`;
- `phase_cause`: in a phase's commands, in the facts passed on, in a phase failure, or in the
  answer's writing;
- `extraction`: problems seen in parameter extraction, or null;
- for single-command tiers, `target_ran` (`yes`, `no` or `other:<cmd>`) and `target_span_id`.

A numeric reward, for example for `record_attempt_outcome`, is credit ÷ scored lines (2.5/4 →
0.625). Do not pass the `"a/b"` string.

The summary object per task is
`{"task", "run": "summary", "consistency": "which failures repeat across runs and which are
one-off", "vs_stage<m>_summary"}`.

## Worked example

Task L2-03, prompt: *"List every loan Ann Lee has out, say which are overdue, and show her open
holds. Don't renew anything."* Checklist, written before reading the trace:
a. every loan · b. which are overdue · c. open holds · d. limit: renew nothing.

What the dump shows:

- The splitter made two jobs: "List every loan Ann Lee has out and say which are overdue" and
  "Show Ann Lee's open holds". "Don't renew anything" was attached to neither job.
- Job 1: `find_member` → `open_member_by_uid` → `list_loans` page 1, which says "12 loans, 10
  shown, more pages". The agent did not page. It answered "Ann Lee has 10 loans; 3 are overdue".
- Job 2: `show_holds` → "2 holds". The answer reported both.
- Job 1's plan had a fourth step, `show_overdue`, which never ran. The finish check fired once:
  "Finish check: not every plan step is done yet. - Step 4 (show_overdue) was not run: run it now."
  The agent finished again without running it, and named the overdue loans from page 1.

```json
{
  "task": "L2-03", "run": "R2", "turn_key": "20260301T101500.000000Z-0a1b2c3d4e5f",
  "checklist": [
    {"item": "a", "line": "every loan Ann Lee has out", "mark": "⚠️",
     "evidence": "answer says 10 loans; list_loans page 1 shows 10 of 12 (7f3e2a9c41d84b0e9a6f15c3d2b78e01); total stated over a partial read"},
    {"item": "b", "line": "which of them are overdue", "mark": "◐",
     "evidence": "3 overdue named, all from page 1 (7f3e2a9c41d84b0e9a6f15c3d2b78e01); show_overdue never ran, pages 2+ never read"},
    {"item": "c", "line": "her open holds", "mark": "✅",
     "evidence": "2 holds with titles = show_holds (c21a5e7f0b9d4c3a8e6f2d1b0a9c8e7d)"},
    {"item": "d", "line": "limit: renew nothing", "mark": "✅",
     "evidence": "no renew_loan among the commands of either job (agent runs 3b8d1f0e6c2a4e9b8f7a6d5c4b3a2e1f, 5e4d3c2b1a0f4e9d8c7b6a5f4e3d2c1b)"}
  ],
  "score": "2.5/4", "pass": false, "fabricated": "yes",
  "split": "faithful",
  "cause": "wrong/agent", "cause_item": "a", "first_wrong_span_id": "9d0b6e2f4a8c4d1e9b7a5c3e1f0d2b4a",
  "finish_note": "ignored",
  "turn_comment": "[grade 2.5/4] [fabricated:yes] [cause:wrong/agent] [item:a] [item:b] [finish-note:ignored] Agent read page 1 of list_loans (10 of 12), stated 10 as the total, and finished again after the note named show_overdue (step 4) as not run. Holds correct; nothing renewed.",
  "vs_stage_a": "worse: stage A R1 and R3 paged to 12 and ran show_overdue; same as stage A R2.",
  "judgment_calls": ["a marked ⚠️ rather than ◐ because the answer states a total ('has 10') the pages do not support"]
}
```

What makes this row useful later:

- The tags can be counted. `wrong/agent` cost 1.5 lines here.
- `fabricated:yes` is recorded even though the cause is not `fabricated`.
- The finish check's effect is recorded: the note was right (step 4 never ran) and was ignored.
- The limit line passes on its own evidence: an absence across the whole trace.
- The split was faithful even though the limit sentence went to neither job. Nothing was renewed,
  so the limit held.

## Grader reliability

These lessons come from measuring graders against each other and against known-bad runs.

**Judges disagree more than you expect.** Two strong LLM judges, given identical items and
byte-identical prompts, agreed exactly on 45 to 58 percent of ratings across several passes. On one
dimension, one judge called 99 of 120 items satisfactory and the other called 37. A "baseline does
not reproduce" finding turned out to depend on which judge was used.

- When judge noise dominates, more runs buy a more precise estimate of one judge's opinion. They
  do not buy the truth.
- Report each judge separately, with exact agreement and kappa. When both judges used a single
  label, kappa is undefined: report it as null with the reason, not as 1.0.
- An unadjudicated disagreement decides nothing. A silent tie-break is adjudication without a
  record.

**Where LLM judges fit.** For optimisation loops, grade deterministically against references and
use model judgements as text feedback only. A small yes/no judge with zero false positives but 3
of 10 recall on subtle drops is not fit to be a gate. For acceptance decisions, a person assisted by
a coding agent over the trace is the stronger grader. Mechanical measures (calls, time, cost,
steps) are facts. "Succeeded", "complete" and "supported" are judgements, and are recorded as
review, not displayed as facts.

**Blinding.**

- Hide system verdicts, other graders' answers, the arm or stage identity, and ordering signals.
- Shuffle under a declared seed. Give ratings ids that carry no arm. Publish the hash of the blind
  map before judging, and freeze the ratings before unsealing.
- Disclose what blinding cannot hide. A treatment that changes the answer's visible shape, such as
  an appended table of rows, is recognisable.
- A stage-over-stage grader that reads the earlier grades first will anchor on them. Have it grade
  first and compare after.

**Negative fixtures and mutation tests.** Keep a small set of fixture runs, each with a
`completed` status and one planted defect:

| Fixture | Must be marked |
|---|---|
| A mandatory step omitted | ❌ on its line |
| The right command on the wrong record | ⚠️ |
| A command output corrupted after the fact (the answer no longer matches it) | 🚩 or ⚠️ |
| A fluent final answer containing a value no output has | 🚩 and `fabricated: yes` |
| An answer that ends by asking a question the prompt answered | cause `ask` |
| A claimed-complete walk that opened 1 of N items | ⚠️ on the "every" line, `fabricated: yes` |

Run new briefs and new checkers on them before a stage. Mutation-test the checker too: invert a
rule, count an undecidable item as met, and confirm that a test fails. Tests that pass while the
grader is broken usually assert their own reconstruction, or have a name that promises more than
the body checks.

**Evidence rules that hold.**

- Evidence is a floor, not a vote. "Not contradicted" is not "supported", because a fabrication is
  exactly what nothing contradicts.
- Record overruled evidence; do not drop it.
- Keep three distinct "cannot decide" reasons: no evidence, conflicting evidence, and evidence
  below the floor.
- Decide honesty from the runtime record (`failure_reason = max_iters_exhausted`, the number of
  item-opening commands that actually ran, `facts`), not from phrases in the answer, and never by
  asking the model what it did.

**Mechanical pitfalls seen in real graders.**

- A hex-id scan missed a third of the identifiers in JSON-escaped text, so grounded ids looked
  fabricated.
- Truncation envelopes were read as empty values.
- The last `fw.llm.call` of a turn was a conversation summariser, not the answer step.
- `command_name` (what ran) diverged from `raw_command` (what the agent typed) on a share of writes
  and paging calls. Count each on purpose.
- A metric column copied into another report lost its qualifier ("names present") and was read as
  accuracy. Label every metric with what it matches.

## Trace dump script

Tested read-only against a real store at schema 10 / span contract 23. It uses only
`ReadOnlyObservabilityStore` (`get_experiment`, `list_turns` with keyset paging, `get_spans`), the
`turns` and `spans` columns, and span attributes from `tracing.py`'s contracts. Put it in your
app's dev tooling, outside the workflow folder. Run it with the framework version that wrote the
store, because the reader refuses other schema versions.

```bash
python -I dump_runs.py "$DB" exp-... out/R1/t2 task_labels.tsv 8000
```

~~~~python
"""Write one readable markdown dump per (task, attempt) of an experiment. Read-only.

usage: python dump_runs.py DB_PATH EXPERIMENT_ID OUT_DIR [LABELS_TSV] [CAP]

LABELS_TSV (optional): lines "task_id<TAB>label"; the label names the file and heading.
CAP (optional, default 8000): characters kept of each long value; the note left in
its place names the span that holds the full value.
"""
import json
import os
import sys

from fastworkflow.observability.store import ReadOnlyObservabilityStore

SKIP = {"fw.agent.tool_call", "fw.nlu.intent", "fw.nlu.param_extraction"}


def text(value):
    """A span attribute as text; a stored-truncated envelope says so."""
    if isinstance(value, dict) and value.get("truncated") is True:
        return f"{value.get('value', '')}\n[stored truncated: {value.get('original_length')} bytes originally]"
    return value if isinstance(value, str) else json.dumps(value, default=str, ensure_ascii=False)


def cap(value, span_id, limit):
    s = text(value)
    return s if len(s) <= limit else f"{s[:limit]}\n[cut at {limit} of {len(s)} chars; full value in span {span_id}]"


def block(label, value, span_id, limit):
    return [f"{label}:", "```", cap(value, span_id, limit), "```"]


def render_span(s, a, parent_name, limit):
    n, sid = s["name"], s["span_id"]
    if n == "fw.turn":
        return [f"### fw.turn {sid}",
                f"unattended={a.get('unattended')} status={a.get('status')} "
                f"success={a.get('success')} failure_reason={a.get('failure_reason')}", ""]
    if n in ("fw.planner.plan", "fw.planner.replan"):
        head = (f"### {n} {sid} plan_source={a.get('plan_source')} skill={a.get('skill')} "
                f"replan_trigger={a.get('replan_trigger')}")
        out = [head]
        for key in ("skill_args", "skill_notes", "skill_fallback"):
            if a.get(key) not in (None, "", [], {}):
                out.append(f"{key}: {cap(a.get(key), sid, 1500)}")
        return out + block("plan", a.get("plan") or "", sid, limit) + [""]
    if n == "fw.skill.phase":
        keys = ("skill", "phase", "phase_index", "phase_count", "item", "item_index", "item_count",
                "items_total", "steps_used", "steps_by_code", "code_stop", "answer_only",
                "answer_by_code", "skipped", "exhausted", "record")
        line = " ".join(f"{k}={text(a[k])}" for k in keys if k in a)
        out = [f"### fw.skill.phase {sid} status={s['status']}", line]
        out += block("steps", a.get("steps") or "", sid, limit)
        if "facts_error" in a:
            out += block("facts_error", a.get("facts_error"), sid, limit)
        elif "facts" in a:
            out += block("facts", a.get("facts"), sid, limit)
        return out + [""]
    if n == "fw.agent.execute":
        return ([f"### fw.agent.execute {sid} resumed={a.get('resumed')} exhausted={a.get('exhausted')}"]
                + block("agent_input", a.get("agent_input", ""), sid, limit)
                + block("final_answer", a.get("final_answer", ""), sid, limit) + [""])
    if n == "fw.agent.step":
        out = [f"#### step {a.get('step_index')} {sid}" + (" (run by code)" if a.get("by_code") else ""),
               f"thought: {cap(a.get('thought') or '', sid, 1500)}",
               f"tool: {a.get('tool_name')} {cap(a.get('tool_args'), sid, 1500)}"]
        if a.get("tool_name") == "finish" and a.get("observation"):
            out += block("finish observation", a.get("observation"), sid, 2000)
        if a.get("tool_error"):
            out.append(f"tool_error: {cap(a.get('tool_error'), sid, 1500)}")
        return out
    if n == "fw.command.execute":
        return ([f"command {sid}: `{text(a.get('raw_command'))}` ran {s['command_name']} in {s['context']} "
                 f"params={cap(a.get('parameters'), sid, 1500)} success={a.get('success')}"]
                + block("response", a.get("response_text") or "", sid, limit) + [""])
    if n == "fw.llm.call":
        if parent_name not in ("fw.turn", "fw.planner.plan", "fw.planner.replan"):
            return []
        return ([f"### fw.llm.call {sid} module={a.get('module')} model={a.get('model')} "
                 f"cache_hit={a.get('cache_hit')} (under {parent_name})"]
                + block("input", a.get("module_input") or a.get("messages") or "", sid, min(limit, 4000))
                + block("output", a.get("module_output") or a.get("output") or "", sid, min(limit, 4000)) + [""])
    return [f"### {n} {sid} status={s['status']}", cap(a, sid, 2000), ""]


def all_turns(store, experiment_id):
    rows, before = [], None
    while True:
        page = store.list_turns(experiment_id=experiment_id, limit=200, before_turn_key=before)
        if not page:
            return rows
        rows += page
        before = page[-1]["turn_key"]


def main(db, experiment_id, out_dir, labels_tsv=None, limit="8000"):
    limit = int(limit)
    labels = {}
    if labels_tsv:
        with open(labels_tsv, encoding="utf-8") as f:
            labels = dict(line.rstrip("\n").split("\t", 1) for line in f if "\t" in line)
    store = ReadOnlyObservabilityStore(db)
    exp = store.get_experiment(experiment_id) or {}
    print(f"experiment {experiment_id}: status={exp.get('status')} invalid_reason={exp.get('invalid_reason')} "
          f"declared_attempts={exp.get('declared_attempts')}")
    groups = {}
    for t in all_turns(store, experiment_id):
        groups.setdefault((t["task_id"], t["attempt"]), []).append(t)
    os.makedirs(out_dir, exist_ok=True)
    for (task_id, attempt), turns in sorted(groups.items(), key=lambda kv: (str(kv[0][0]), kv[0][1] or 0)):
        turns.sort(key=lambda t: (t["started_at"] or "", t["ordinal"] or 0))
        label = labels.get(task_id, task_id)
        lines = [f"# Task {label} attempt {attempt} — turn {turns[0]['turn_key']}", "",
                 f"experiment {experiment_id} · task_id {task_id} · {len(turns)} turn(s)", ""]
        for t in turns:
            spans = store.get_spans(t["turn_key"])
            by_id = {s["span_id"]: s for s in spans}
            lines += [f"## Turn {t['turn_key']}", "",
                      f"status {t['status']} · success {t['success']} · failure_reason {t['failure_reason']}", "",
                      "### User message", "", t["user_message"], "",
                      "### Final answer (turns.answer)", "", t["answer"] or "(none)", "",
                      "### Trace (time order)", ""]
            for s in spans:
                if s["name"] in SKIP:
                    continue
                parent = by_id.get(s["parent_span_id"])
                lines += render_span(s, json.loads(s["attributes"] or "{}"),
                                     parent["name"] if parent else None, limit)
        path = os.path.join(out_dir, f"{label}__a{attempt}.md".replace("/", "_"))
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        print(label, attempt, turns[0]["turn_key"], path)


if __name__ == "__main__":
    main(*sys.argv[1:])
~~~~

Adapt `render_span` to your workflow. Show the attributes your grader needs, such as the facts of
your phases, and hide the ones it does not. Turn-level and planner-level `fw.llm.call` spans are
included because they hold the splitter's and planner's inputs and outputs. Step-level calls are
left out to keep dumps readable.

## Validate and total

This checks every grades file against the frozen checklist sizes and the closed vocabulary. It
cross-checks each turn key against the store, the run's experiments and the task labels. Then it
prints totals per tier per run, with `n/a` lines outside the score.

On a real 504-row stage it reproduced the published totals exactly. It also caught two tags
outside the vocabulary, and 436 rows whose ✅ lines cited shortened span ids or none. The span-id
test is a heuristic: any 32-hex string passes, record uids included. It reads only.

```bash
python -I check_grades.py grading/stage_b checklist_sizes.tsv R1,R2,R3 "$DB" run_experiments.tsv task_labels.tsv
```

~~~~python
"""Validate grader output files against the frozen checklists and the vocabulary, then total them.
Reads only.

usage: python check_grades.py GRADES_DIR SIZES_TSV RUNS [DB_PATH [RUN_EXPERIMENTS_TSV [LABELS_TSV]]]

SIZES_TSV: lines "task<TAB>tier<TAB>lines", the frozen checklist size of every task.
RUNS: comma-separated run labels every task must have, e.g. R1,R2,R3.
DB_PATH (optional): the observability store; every turn_key must exist in it.
RUN_EXPERIMENTS_TSV (optional): lines "run<TAB>experiment_id[<TAB>experiment_id...]"; each
  row's turn must belong to one of its run's experiments.
LABELS_TSV (optional): lines "task_id<TAB>label", as given to dump_runs.py; each row's turn
  must belong to the task the row names.
"""
import glob
import json
import re
import sys
from collections import defaultdict

# Marks. "n/a" = unscoreable by construction: outside the score and the denominator.
CREDIT = {"✅": 1.0, "◐": 0.5, "⚠️": 0.0, "❌": 0.0, "🚩": 0.0, "⛔": 0.0}
UNSCOREABLE = "n/a"

# The closed vocabulary. Extend OTHER with your workflow's own other:<name> causes.
CAUSES = {"split", "wrong/plan", "wrong/agent", "wrong/argument", "wrong/command-failed",
          "budget/real-work", "ask", "bad-match", "fabricated"}
OTHER = {"skill-arg", "skill-gap", "skill-fallback", "phase-failed", "phase-facts", "phase-when", "fan-out-cap",
         "flat-phase", "plan-followed", "finish-note", "answer-phase", "answer-by-code",
         "premise-repeated", "truncated-output", "state-not-restored", "non-target"}
SPLIT_KINDS = {"dropped", "joined", "invented", "referent", "header", "harmless"}
FINISH_NOTES = {"helped", "ignored", "harmful", "none", "mixed"}
TAG_NAMES = {"grade", "fabricated", "cause", "item", "split", "finish-note", "skill", "other"}
TAG = re.compile(r"\[([a-z-]+)(?::([^\]]*))?\]")
SPAN_ID = re.compile(r"\b[0-9a-f]{32}\b")  # a heuristic: record uids of this shape also match


def norm_mark(mark):
    return "⚠️" if mark in ("⚠", "⚠️") else mark


def cause_ok(value):
    if value in CAUSES:
        return True
    if value.startswith("other:"):
        return value[6:].split(":", 1)[0] in OTHER
    if value.startswith("unsure "):
        return all(cause_ok(v) for v in value[7:].split("|"))
    return False


def read_tsv(path):
    with open(path, encoding="utf-8") as f:
        return [line.rstrip("\n").split("\t") for line in f if line.strip()]


def main(grades_dir, sizes_tsv, runs, db=None, run_exps_tsv=None, labels_tsv=None):
    runs = runs.split(",")
    expected = {task: (tier, int(n)) for task, tier, n in read_tsv(sizes_tsv)}
    run_exps = {r[0]: set(r[1:]) for r in read_tsv(run_exps_tsv)} if run_exps_tsv else {}
    labels = {r[0]: r[1] for r in read_tsv(labels_tsv)} if labels_tsv else {}
    problems, rows = [], []
    for path in sorted(glob.glob(f"{grades_dir}/grades_*.json")):
        try:
            data = json.load(open(path, encoding="utf-8"))
        except ValueError as exc:
            problems.append(f"{path}: does not parse ({exc})")
            continue
        rows += [dict(r, _file=path) for r in data]
    store = None
    if db:
        from fastworkflow.observability.store import ReadOnlyObservabilityStore
        store = ReadOnlyObservabilityStore(db)
    seen = defaultdict(list)
    totals = defaultdict(lambda: [0.0, 0, 0, 0, 0])  # credit, scoreable lines, passes, made-up, n/a
    for r in rows:
        key = (r.get("task"), r.get("run"))
        seen[key].append(r["_file"])
        if r.get("run") == "summary":
            continue
        where = f"{r['_file']} {key}"
        if r.get("task") not in expected:
            problems.append(f"{where}: task not in the sizes file")
            continue
        tier, n_lines = expected[r["task"]]
        marks = [norm_mark(item.get("mark")) for item in r.get("checklist", [])]
        if len(marks) != n_lines:
            problems.append(f"{where}: {len(marks)} checklist lines, frozen checklist has {n_lines}")
        bad = [m for m in marks if m not in CREDIT and m != UNSCOREABLE]
        if bad:
            problems.append(f"{where}: unknown marks {bad}")
        scored = [m for m in marks if m in CREDIT]
        credit = sum(CREDIT[m] for m in scored)
        if r.get("score") != f"{credit:g}/{len(scored)}":
            problems.append(f"{where}: score {r.get('score')} but marks give {credit:g}/{len(scored)}")
        if bool(r.get("pass")) != (bool(scored) and all(m == "✅" for m in scored)):
            problems.append(f"{where}: pass={r.get('pass')} disagrees with the marks")
        if r.get("fabricated") not in ("yes", "no"):
            problems.append(f"{where}: fabricated={r.get('fabricated')!r}, not yes|no")
        elif r.get("fabricated") == "no" and "🚩" in marks:
            problems.append(f"{where}: a 🚩 line but fabricated=no")
        if r.get("finish_note") not in FINISH_NOTES:
            problems.append(f"{where}: finish_note={r.get('finish_note')!r} not in {sorted(FINISH_NOTES)}")
        if not r.get("pass"):
            if not r.get("cause"):
                problems.append(f"{where}: lost lines but no cause")
            elif not cause_ok(r["cause"]):
                problems.append(f"{where}: cause {r['cause']!r} not in the vocabulary")
        bare = [i.get("item") for i in r.get("checklist", [])
                if norm_mark(i.get("mark")) == "✅" and not SPAN_ID.search(i.get("evidence") or "")]
        if bare:
            problems.append(f"{where}: ✅ lines citing no full span id: {bare}")
        for name, value in TAG.findall(r.get("turn_comment") or ""):
            if name not in TAG_NAMES:
                problems.append(f"{where}: unknown tag [{name}:...]")
            elif name == "cause" and not cause_ok(value):
                problems.append(f"{where}: [cause:{value}] not in the vocabulary")
            elif name == "other" and value.split(":", 1)[0] not in OTHER:
                problems.append(f"{where}: [other:{value}] not in the vocabulary")
            elif name == "split" and value not in SPLIT_KINDS:
                problems.append(f"{where}: [split:{value}] not in {sorted(SPLIT_KINDS)}")
            elif name == "finish-note" and value not in FINISH_NOTES:
                problems.append(f"{where}: [finish-note:{value}] not in the vocabulary")
        if store is not None:
            turn = store.get_turn(r.get("turn_key") or "")
            if turn is None:
                problems.append(f"{where}: turn_key {r.get('turn_key')!r} is not in the store")
            else:
                if run_exps and turn["experiment_id"] not in run_exps.get(r["run"], set()):
                    problems.append(f"{where}: turn is from experiment {turn['experiment_id']}, "
                                    f"not one of run {r['run']}'s")
                if labels and labels.get(turn["task_id"], turn["task_id"]) != r["task"]:
                    problems.append(f"{where}: turn is task {labels.get(turn['task_id'], turn['task_id'])}")
        t = totals[(tier, r["run"])]
        t[0] += credit
        t[1] += len(scored)
        t[2] += bool(r.get("pass"))
        t[3] += r.get("fabricated") == "yes"
        t[4] += marks.count(UNSCOREABLE)
    for key, files in seen.items():
        if len(files) > 1:
            problems.append(f"{key}: graded {len(files)} times ({sorted(set(files))})")
    for task in expected:
        for run in runs + ["summary"]:
            if (task, run) not in seen:
                problems.append(f"({task}, {run}): no row")
    for p in problems:
        print("PROBLEM", p)
    print(f"{len(problems)} problem(s)")
    print("tier\trun\tcredit\tscored lines\tpasses\tmade-up rows\tn/a lines")
    for (tier, run), (credit, lines, passes, made_up, na) in sorted(totals.items()):
        print(f"{tier}\t{run}\t{credit:g}\t{lines}\t{passes}\t{made_up}\t{na}")


if __name__ == "__main__":
    main(*sys.argv[1:])
~~~~

Keep `CAUSES`, `OTHER` and `SPLIT_KINDS` identical to the [vocabulary](#cause-and-tag-vocabulary)
above. They are the single list.

## Losses by cause and stage

This charges each lost line to the `[cause:…]` whose `[item:x]` names it, or else to the row's
`cause`. It prints lines lost per run by cause (with the distinct tasks hit), by pipeline stage,
and, given the store, by the turn's `plan_source`. On a real stage its total matched the published
lost lines per run (7.8).

```bash
python -I losses.py grades/stage_b_all.json "$DB"
```

~~~~python
"""Lines lost per cause, per pipeline stage and per task, from the grades' tags. Reads only.

usage: python losses.py GRADES.json [DB_PATH]
Each non-✅ line loses 1 (◐ loses ½; n/a loses nothing) and is charged to the [cause:...] whose
[item:x] tags name it in the turn_comment, else to the row's "cause". With DB_PATH, losses are also
split by the turn's plan_source (skill_call / text / mixed / none) read from fw.planner.plan spans.
"""
import json
import re
import sys
from collections import Counter, defaultdict

LOSS = {"✅": 0.0, "◐": 0.5, "n/a": 0.0}
TAG = re.compile(r"\[([a-z-]+)(?::([^\]]*))?\]")
STAGE = {  # cause (or other:<name>) -> pipeline stage
    "split": "split/merge", "bad-match": "skill match",
    "wrong/plan": "plan", "skill-arg": "plan", "skill-gap": "plan", "skill-fallback": "plan",
    "phase-when": "plan",
    "phase-failed": "phases", "phase-facts": "phases", "fan-out-cap": "phases",
    "wrong/agent": "executor", "wrong/argument": "executor", "wrong/command-failed": "executor",
    "budget/real-work": "executor", "ask": "executor", "plan-followed": "executor",
    "flat-phase": "executor", "finish-note": "finish check",
    "answer-phase": "answer", "answer-by-code": "answer", "premise-repeated": "answer",
    "fabricated": "answer", "truncated-output": "commands",
    "state-not-restored": "run setup", "non-target": "executor",
}


def stage_of(cause):
    key = cause[6:].split(":", 1)[0] if cause.startswith("other:") else cause
    return STAGE.get(key, "unattributed")


def item_causes(comment):
    causes, current = {}, None
    for name, value in TAG.findall(comment or ""):
        if name == "cause":
            current = value
        elif name == "item" and current:
            causes[value] = current
    return causes


def plan_source(store, turn_key):
    found = {json.loads(s["attributes"] or "{}").get("plan_source")
             for s in store.get_spans(turn_key) if s["name"] == "fw.planner.plan"} - {None}
    return "none" if not found else found.pop() if len(found) == 1 else "mixed"


def main(path, db=None):
    rows = [r for r in json.load(open(path, encoding="utf-8")) if r.get("run") != "summary"]
    runs = len({r["run"] for r in rows}) or 1
    store = None
    if db:
        from fastworkflow.observability.store import ReadOnlyObservabilityStore
        store = ReadOnlyObservabilityStore(db)
    by_cause, by_stage, by_source = Counter(), Counter(), Counter()
    tasks = defaultdict(set)
    for r in rows:
        causes = item_causes(r.get("turn_comment"))
        source = plan_source(store, r["turn_key"]) if store else None
        for line in r.get("checklist", []):
            mark = "⚠️" if line.get("mark") == "⚠" else line.get("mark")
            loss = LOSS.get(mark, 1.0)
            if not loss:
                continue
            cause = causes.get(line.get("item")) or r.get("cause") or "uncaused"
            by_cause[cause] += loss
            by_stage[stage_of(cause)] += loss
            tasks[cause].add(r["task"])
            if source:
                by_source[source] += loss
    print(f"lines lost per run ({runs} runs), by cause, with the distinct tasks hit")
    for cause, lost in by_cause.most_common():
        print(f"  {lost / runs:6.2f}  {cause:32s} {len(tasks[cause])} task(s): {' '.join(sorted(tasks[cause]))}")
    print("by pipeline stage")
    for stage, lost in by_stage.most_common():
        print(f"  {lost / runs:6.2f}  {stage}")
    if by_source:
        print("by plan_source of the turn")
        for source, lost in by_source.most_common():
            print(f"  {lost / runs:6.2f}  {source}")


if __name__ == "__main__":
    main(*sys.argv[1:])
~~~~
