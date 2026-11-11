# Taxonomy file schema

One YAML file per corpus (or per cumulative corpus). Top-level keys:
`corpus`, `modes`, and optionally `collisions`, `gaps`, `coverage`. Start the
file with a header comment naming the corpus path(s), how it was produced
(distill command, runs, which side was the reference, judge model) and the
signal legend.

```yaml
corpus:
  execution_items: 13              # items in execution_agent_anti_patterns.md
  planning_items: 11               # items in planning_agent_insights.md
  tasks: [<task id>, ...]
  # teacher-student:   teacher: <run id>   student: <run id>
  # self / consistency, one entry per distillation round:
  rounds:
    - {experiment: <run id>, passed: [3, 4], failed: [1, 2], pairs: [[1, 3], [2, 3]]}

modes:
  - id: entity-left-unprocessed          # kebab-case; names the DEFECT
    name: Asked to act on all records of a kind; one was skipped
    insights: {execution: [E7, E8], planning: [P6]}   # every id exactly once across all modes
    what: >
      What went wrong, in workflow terms. Task values may be cited here as
      evidence - never in `check`.
    source: [teacher]      # optional: reference kinds behind the insights - self | teacher
    signal: +intent        # trajectory | +intent | +user_text | +question-judge | none
    check: >               # required when compiled. Positive evidence only.
      If the user asked for the operation on ALL records of a kind, then
      at finish every record listed by <list command> must have been read, and
      every record read in the eligible state must have a successful
      <operation command>.
    hook: before-finish    # before-side-effect | after-observation | before-finish
    compiled: R7           # the rule id it becomes, or false
    overlap: none          # mode / rule in another taxonomy, or none
    notes: >               # required when compiled is false - the reason
      ...

collisions:                # optional
  - between: [E14, P8]
    tension: what each pushes the agent to do
    resolution: how the cases are told apart

gaps:                      # optional: causes in the grading no insight names
  - trial: <run id>/<attempt>
    cause: what failed
    why_not_compiled: no insight in the corpus states it

coverage:
  insights: 24
  in_compiled_modes: 15
  compiled_rules: 7
  not_compiled:            # reason -> insight ids
    contradicts-policy: [E13, P8]
    needs-judge: [E9, P7]
  new_vs_<other taxonomy>: [<mode id>, ...]
  shared_with_<other taxonomy>: [<mode id>, ...]
```

## Field rules

- `insights`: both keys present; a list may be empty (`[]`).
- `source` (combined distillation): which kind of reference produced the
  mode's insights, from `provenance.json`. Lets results be split by source
  later ("do self-distilled rules fire better than teacher ones?").
- `hook` - when the verifier runs:
  - `before-side-effect`: a command was chosen and has not run (writes).
  - `after-observation`: a non-command step was chosen, e.g. a question to
    the user.
  - `before-finish`: the agent's answer exists and has not been sent.
- `check`: a predicate over the trajectory, naming the workflow's commands
  and their output fields. No record ids, amounts, dates or user names -
  the same check must hold for every task. Phrase it as one principle, not a
  list of the situations the insights described.
- `compiled: false` requires `notes` starting with one reason:
  `contradicts-policy`, `lexical-only`, `no-deterministic-signal`,
  `not-outcome-relevant`, `needs-judge`, `task-specific`, `covered-by`; then
  the specifics
  (for policy contradictions, quote the policy line).
- Incremental rounds: append modes under `# --- round N ---`; never renumber
  insights; update `corpus.rounds` and `coverage`.
