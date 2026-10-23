# 1. Fixed steps: a skill with no inputs

**Goal:** the smallest skill there is. When someone asks for a library overview, the agent
should list the branches, count the overdue loans and say both.

## The skill file

<!-- file: tests/skill_tutorial_workflow/_skills/overdue_summary.py -->
```python
"""overdue_summary: the library's branches and how many loans are overdue.

Fixed steps: the skill takes no inputs. It lists the branches, counts the overdue loans
across the library and says both.
"""


class Signature:
    """Summarize the library: its branches and how many loans are overdue."""

    plain_utterances = ["give me the library overview", "how many loans are overdue overall"]


class PlanGenerator:
    def summary(self):
        return ["list_branches",
                "count_overdue",
                "say how many branches the library has and how many loans are overdue"]
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
- **`class PlanGenerator`**: a command generates a response, a skill generates a plan. Each
  public method is a **phase**; this one has a single method, `summary`, that takes only
  `self`.
- **It returns its steps as text**, one string per line of the plan, exactly the words the
  agent will read. A method builds text only: it never calls a backend or reads state.
- `list_branches` and `count_overdue` are **command steps**: the first word is a command of
  the workflow. Fixed steps are a template the planner adapts, so they are not checked at
  load; from page 2 on, every command a step names is checked against the workflow's
  commands, and a typo fails the load (see [troubleshooting](12-troubleshooting.md)).
- `say how many ...` is an **answer step**: it names what the answer should say.

## What the planner sees

A skill with no inputs is a list of *fixed steps*. Its template is the list itself:

<!-- template: overdue_summary -->
```text
list_branches
count_overdue
say how many branches the library has and how many loans are overdue
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
Summarize the library: its branches and how many loans are overdue.

Source: `_skills/overdue_summary.py`. Kind: fixed steps, no inputs.

## Inputs

None: the steps are fixed.

## Example requests

- give me the library overview
- how many loans are overdue overall

## Steps

1. `list_branches`
2. `count_overdue`
3. `say how many branches the library has and how many loans are overdue`
```

## At run time

A skill with no inputs is not a typed call. When the matcher picks it for a request, its
steps are added to the request as a template, and the text planner writes the plan from
them, adapting them to the request. There are no phases and no facts, so there is nothing
for a dry run to show: what runs is the plan the planner writes.

A skill with no inputs is one method here. With two or more phase methods it is a skill
call with no arguments, run as phases (phases start on page 3).

## What you learned

A skill is a `Signature` (what it does) and a `PlanGenerator` (its steps, as text). With no
inputs it is a fixed list of steps the planner adapts.
