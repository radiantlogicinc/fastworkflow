# Judging changes: code and templates

Companion to [SKILL.md](SKILL.md). Both scripts read grade files in the
[grade JSON schema](../grade-benchmark-runs/reference.md#grade-json-schema). `judge_change.py` only
prints. `rescore.py` writes one new file and never touches its input. Both were checked against
real stored grades: they reproduced the published re-scored total and the published verdicts. Keep
them in your app's dev tooling, not in the workflow folder.

## Contents

- [Re-score stored grades under a new rule](#re-score-stored-grades-under-a-new-rule)
- [Judge one change on its target tasks](#judge-one-change-on-its-target-tasks)
- [Keep/revert table](#keeprevert-table)
- [Results page outline](#results-page-outline)
- [Offline replay checklist](#offline-replay-checklist)

## Re-score stored grades under a new rule

It writes the re-scored copy to a new file, which the judging script then reads, and it refuses to
overwrite anything. The example applies the rule "a right answer through a non-target command now
scores full": in rows tagged `[other:non-target:<cmd>]`, ◐ lines become ✅. The tag is on the row, so the rewrite touches every
◐ line in the row. That is exact for one-line tasks. For multi-line tasks, have graders put the tag
next to its `[item:x]` and match on the item.

```python
"""Re-apply a changed scoring rule to stored grades through their tags.

usage: python rescore.py GRADES_IN.json GRADES_OUT.json
Never edits GRADES_IN; writes the re-scored copy to GRADES_OUT (a new file) and prints totals.
"""
import json
import os
import re
import sys

CREDIT = {"✅": 1.0, "◐": 0.5}


def rescore(rows, tag_pattern, from_mark, to_mark):
    """Rows whose turn_comment carries the tag: lines marked from_mark become to_mark."""
    tag = re.compile(tag_pattern)
    out, changed = [], 0
    for r in rows:
        r = json.loads(json.dumps(r))                      # a copy: the stored grade stays as it is
        if r.get("run") != "summary" and tag.search(r.get("turn_comment") or ""):
            for line in r["checklist"]:
                if line["mark"] == from_mark:
                    line["mark"], changed = to_mark, changed + 1
            marks = [l["mark"] for l in r["checklist"] if l["mark"] != "n/a"]
            r["score"] = f"{sum(CREDIT.get(m, 0) for m in marks):g}/{len(marks)}"
            r["pass"] = bool(marks) and all(m == "✅" for m in marks)
            r["rescored"] = f"{tag_pattern}: {from_mark} -> {to_mark}"
        out.append(r)
    return out, changed


def total(rows):
    return sum(float(r["score"].split("/")[0]) for r in rows if r.get("run") != "summary")


if __name__ == "__main__":
    src, dst = sys.argv[1], sys.argv[2]
    if os.path.abspath(src) == os.path.abspath(dst) or os.path.exists(dst):
        sys.exit(f"refusing to overwrite {dst}: write the re-scored grades to a new file")
    rows = json.load(open(src, encoding="utf-8"))
    # The rule being re-applied: a right answer through a non-target command now scores full.
    new, changed = rescore(rows, r"\[other:non-target:[^\]]+\]", "◐", "✅")
    with open(dst, "w", encoding="utf-8") as f:
        json.dump(new, f, ensure_ascii=False, indent=1)
    print(f"{changed} line(s) changed; total {total(rows):g} -> {total(new):g}; wrote {dst}")
```

## Judge one change on its target tasks

```bash
python rescore.py grades/stage_a_all.json grades/stage_a_rescored.json   # earlier stage, current rules
python judge_change.py grades/stage_a_rescored.json grades/stage_b_all.json L2-03,L2-07,L5-04
```

```python
"""Per-run target-task scores of one change, earlier stage vs new stage, and a verdict. Reads only.

usage: python judge_change.py EARLIER.json NEW.json TASK[,TASK...]
Both files must already be scored under the same rules (re-score the earlier one first).
"""
import json
import sys
from collections import defaultdict


def per_run(rows, tasks):
    runs, seen = defaultdict(float), defaultdict(set)
    for r in rows:
        if r.get("run") != "summary" and r.get("task") in tasks:
            runs[r["run"]] += float(r["score"].split("/")[0])
            seen[r["task"]].add(r["run"])
    return [runs[k] for k in sorted(runs)], seen


def verdict(old, new):
    if not old or not new:
        return "UNDECIDED: missing runs"
    gain = sum(new) / len(new) - sum(old) / len(old)
    spread = max(max(old) - min(old), max(new) - min(new))
    if min(new) > max(old):
        return f"KEEP: every new run beats every old run ({gain:+.1f} per run)"
    if gain > 0 and gain > spread:
        return f"KEEP: gain {gain:+.1f} per run exceeds the spread {spread:.1f}"
    if gain < 0 and -gain > spread:
        return f"REVERT: loss {gain:+.1f} per run exceeds the spread {spread:.1f}"
    return f"REVERT or re-measure: {gain:+.1f} per run is within noise (spread {spread:.1f})"


if __name__ == "__main__":
    old_rows = json.load(open(sys.argv[1], encoding="utf-8"))
    new_rows = json.load(open(sys.argv[2], encoding="utf-8"))
    tasks = set(sys.argv[3].split(","))
    old, old_seen = per_run(old_rows, tasks)
    new, new_seen = per_run(new_rows, tasks)
    for name, seen in (("earlier", old_seen), ("new", new_seen)):
        for task in sorted(tasks - set(seen)):
            print(f"WARNING: target task {task} has no rows in the {name} stage")
        if len({len(r) for r in seen.values()}) > 1:
            print(f"WARNING: target tasks have different run counts in the {name} stage")
    if len(old) != len(new):
        print(f"WARNING: {len(old)} earlier runs vs {len(new)} new runs")
    print(" / ".join(f"{x:g}" for x in old), "->", " / ".join(f"{x:g}" for x in new))
    print(verdict(old, new))
```

It warns when a target task is missing from either stage, or when run counts differ. The verdict
follows the rule in SKILL.md §4. It is a starting point, not the decision. Adjust it
for bug fixes, costs, thin samples and single-event totals, and say why in the table.

## Keep/revert table

| Change (commit) | Target tasks | Old runs → new runs | Per run | Verdict |
|---|---|---|---|---|
| `member_review` skill (abc123) | L2-03 L2-07 L5-04 | 10 / 8.5 / 9 → 12.5 / 13.5 / 13 | +3.8 | KEEP: every new run beats every old run |
| Finish note names the page to read (def456) | L1-07 L1-12 | 1 / 0 / 0 → 1 / 1 / 1 | +0.7 | KEEP (bug fix): within noise by the rule, but it fixes a defect the traces show in every earlier stage, and it hurts nothing elsewhere |
| Splitter votes over 3 samples (0a1b2c) | split losses, all tiers | 11.5 → 6 lines in 3 runs | ~+1.8 | REVERT: single events drive both totals; 3× the calls |
| Example request added to a skill (3d4e5f) | L5-02 | 9 / 10 / 9 → 9.5 / 10 / 10 | +0.5 | REVERT: within noise |

## Results page outline

```markdown
# Stage <n> results: <workflow> benchmark, <date>, <runs> runs × <tiers> tiers, <jobs> jobs

<headline: mean lines per run of the total (percent); Δ vs stage <n-1>; Δ vs baseline (re-scored)>

## Scores by tier
| Tier | Lines | R1 / R2 / R3 | Mean | Passes per run | Stage <n-1> mean |
Made-up answers: <rows per tier over the runs> (stage <n-1>: <k>). Rule notes: <rules in force>.

## Each change since stage <n-1>
<keep/revert table>
<one line on whether the kept configuration was itself measured>

## Where the lost lines go (<x> per run)
| Cause / stage | Lines per run | Tasks | What happens |

## Next fixes, largest first
| Fix | At stake per run | Where (app or framework) |
"At stake" is the most a fix could recover, from the grades; it is not a forecast.

## What ran
<framework and app commits; models per role; cache off; unattended; lanes and workers;
state restores; grading: number of graders, checklists, compared with>
```

## Offline replay checklist

```
- [ ] Every benchmark prompt replayed, not a sample; same component code as the coming stage
- [ ] ≥3 repetitions per prompt; results compared as distributions (k-of-n per prompt)
- [ ] FW_LM_CACHE=0 set where fastworkflow.get_env_var reads it (the loaded env file), and
      verified: repeats of a multi-part prompt sometimes differ (count distinct answers), and
      usage_rollup's cache counts show no hits (absent cache_hit = unknown, not a miss)
- [ ] Checker run on deliberately broken outputs (dropped job, joined jobs, stuttered name,
      unresolved "each of them", another job's name, dropped kind word): each one fails
- [ ] Gain reported with its call cost; a replay-only gain is a hypothesis for the next stage
```
