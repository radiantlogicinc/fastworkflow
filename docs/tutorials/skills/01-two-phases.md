# 1. The simplest skill: two phases, no inputs

**Goal:** when someone asks for a library overview, the agent should list the branches,
then count the overdue loans and say how many there are. The skill takes nothing from the
request, and does the job in two **phases**.

## The skill file

<!-- file: tests/skill_tutorial_workflow/_skills/overdue_summary.py -->
```python
"""overdue_summary: the library's branches, then how many loans are overdue.

Two phases and no inputs: branches lists the branches; overdue counts the overdue loans
across the library and answers.
"""


class Signature:
    """List the library's branches, then say how many loans are overdue across the library."""

    plain_utterances = ["give me the library overview", "how many loans are overdue overall"]


class PlanGenerator:
    def branches(self):
        return ["list_branches"]

    def overdue(self):
        return ["count_overdue",
                "say how many loans are overdue across the library"]
```

## Line by line

- **The module docstring** says the job in a few sentences, for the next person who reads
  the file. The framework does not read it.
- **`class Signature`** is written exactly as a command's. Its **docstring is the skill's
  description**: the skill matcher chooses a skill for a request by it. A `Signature`
  without a docstring fails the load.
- **`plain_utterances`** are example requests. The matcher shows the first nine after the
  description, in every matcher prompt, so write them the way people ask, and generic.
- **No `Input`.** This skill takes nothing from the request. Leaving `Input` out and writing
  an empty `class Input(BaseModel): pass` mean the same.
- **`class PlanGenerator`**: a command generates a response, a skill generates a plan.
  **Each public method is a phase**, run in the order written: `branches`, then `overdue`.
  A method whose name starts with `_` is a helper, not a phase.
- **A phase is one executor run** with its own step budget: the agent of `branches` is told
  to do only its steps, as part 1 of 2 of a larger job. Short, separate runs keep a long
  job from drifting.
- **Each method returns its steps as text**, one string per line of the plan, exactly the
  words the agent will read. A method builds text only: it never calls a backend or reads
  state. With no inputs, the methods take only `self`.
- `list_branches` and `count_overdue` are **command steps**: the first word is a command of
  the workflow. Every command a step names is checked against the workflow's commands at
  load, and a typo fails the load (see [troubleshooting](12-troubleshooting.md)).
- `say how many ...` is an **answer step**: it names what the answer should say.

## What the planner sees

The planner sees the skill's *template*: every phase's steps, joined.

<!-- template: overdue_summary -->
```text
list_branches
count_overdue
say how many loans are overdue across the library
```

`fastworkflow skill preview` renders one markdown page per skill. From the repository root:

```sh
fastworkflow skill preview tests/skill_tutorial_workflow --out docs/tutorials/skills/previews
```

Without `--out` it writes next to the workflow, in `<workflow>/../docs/skill_previews`; add
`--check` in CI to fail when a committed preview is stale. This skill's preview (all of
[previews/overdue_summary.md](previews/overdue_summary.md) below its title):

<!-- preview: overdue_summary -->
```markdown
List the library's branches, then say how many loans are overdue across the library.

Source: `_skills/overdue_summary.py`. Kind: phased: `PlanGenerator` runs it as bounded sub-plans, each passing facts on.

## Inputs

None: every call runs the same plan.

## Example requests

- give me the library overview
- how many loans are overdue overall

## Phases

| phase | output | runs | fans out |
|---|---|---|---|
| `branches` | none | always | no |
| `overdue` | none | always | no |

## Facts

What each phase's output holds, filled in code from the commands the phase ran.

No phase passes facts on.

## Plan

One for every call: the skill takes no inputs.

### Phase 1: `branches`

Reads no earlier facts. Output: none.

1. `list_branches`

### Phase 2: `overdue`

Reads no earlier facts. Output: none.

1. `count_overdue`
2. `say how many loans are overdue across the library`
```

## At run time

When the matcher picks the skill for a request, the planner makes a **skill call**. With no
inputs it has no arguments to fill (`overdue_summary()`), so every call runs the plan
above. A *dry run* (see the [README](README.md#running-these-yourself)) runs it offline:

<!-- dry-run: overdue_summary {} -->
```text
Phase branches
  1. list_branches
  code runs: list_branches
Phase overdue
  1. count_overdue
  2. say how many loans are overdue across the library
  code runs: count_overdue
  the agent writes the answer from these runs
```

Each step written `command` with nothing left to fill is one **code runs** itself, before
the agent takes a step. The answer is the **last phase's**: its agent writes it from that
phase's runs. Nothing else passes from one phase to the next: the branches the first phase
listed are not in the answer. To say them too, the first phase must hand them on as a
**fact**: page 3.

A skill with no inputs and **one** method is the plainest form of all, *fixed steps*: it
is not a call; its steps are added to the request as a template that the text planner
adapts.

## What you learned

A skill is a `Signature` (what it does) and a `PlanGenerator` (its plan). Each public
method is a phase, one executor run, in the order written; it returns its steps as text.
