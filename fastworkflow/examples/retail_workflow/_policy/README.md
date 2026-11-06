# Retail policy verifiers — worked example

This folder shows what the policy-verifier loop produces for one workflow. The
engine is generic (`fastworkflow/policy_decision.py`, `policy_runtime.py`). Everything
here is **retail content**, kept as an example of the shape your own workflow's
files take.

This is the table used in the 15-task retail evaluation runs (`retail-demo/0.4.1`).

| File | What it is |
|---|---|
| `corpus/` | The distilled insights, cumulative over four rounds (69 execution, 48 planning), and `provenance.json`: which failing trial was compared with which passing reference, and what each pair produced |
| `taxonomy.yaml` | The insights clustered into 37 failure modes. Each mode either became a rule (`compiled: S1`…) or has a written reason it did not |
| `table.py` | The compiled rules (`TABLE`) and the workflow's declared facts (`FACTS`) |

The rules come from two kinds of distillation:
- **S1–S8**: self-distillation, a failing trial against a passing trial of the same model.
- **C2, C4, C5, C6, C8 and D1**: pool distillation, a passing trial of the same model
  where one exists, else a teacher's.

## Turn it on

There is nothing to set: the table is on because it is at `_policy/table.py` in the
workflow folder. To run the workflow without it, use a copy of the workflow without
`_policy/`. Some rules read what the user asked
for (`FACTS.intent_operations`). That takes one model call per user message, on
`LLM_POLICY_INTENT`, or on `LLM_AGENT` when that is unset.

## Audit it on recorded runs

```bash
fastworkflow policy replay fastworkflow/examples/retail_workflow \
    --experiment <experiment id>
```

`--table PATH` audits another table (a candidate) instead of `_policy/table.py`.

A rule should stay silent on passing trials. On failing trials it should fire
where the trial went wrong, and nowhere else.

## How it was made

1. Small-model (gpt-oss-20b) trials were recorded as experiments, with several
   trials per task, on six training tasks (16, 35, 41, 76, 87, 110).
2. Four rounds of distillation compared every failing trial with a passing one. The
   first used `fastworkflow distill --self`; later rounds used
   `--pool-students <runs> --pool-teachers <runs> --provenance …`, which takes a
   passing trial of the same model where one existed, else a passing teacher trial.
   Each later round distilled trials run under the previous round's table.
3. The insights were clustered with the `insights-to-failure-modes` coding skill.
4. Each mode with a deterministic check became a row (`distill-policy-verifiers` skill).
   Planning hints (nudges) were also tried in this research; they are not part of
   fastworkflow.

The domain policy quoted in `taxonomy.yaml` as `policy.md` is the tau2-bench retail
policy (`data/tau2/domains/retail/policy.md` in the tau2-bench repository). Task ids
in the taxonomy's notes are evidence of where a defect was seen. They are never rule
content.
