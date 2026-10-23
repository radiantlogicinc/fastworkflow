# Writing skills by hand: a tutorial

A **skill** is a job your workflow's agent should do the same way every time: the same
commands, in the same order, with the answer worded the same way. You write it as one Python
file, `_skills/<name>.py`, next to `_commands/`. It reads like a command file: a `Signature`
says what the skill does and what it takes, and where a command has a `ResponseGenerator`, a
skill has a `PlanGenerator` that writes the plan.

These pages build skills step by step, one new idea per page, on a toy library workflow. Every
skill shown is a real file in the fastworkflow repository, and every preview, template and run
shown on these pages is checked against the code by `tests/test_skill_tutorials.py`, so the
pages cannot drift from what the framework does.

## The toy workflow

`tests/skill_tutorial_workflow/` is a small library: two branches, four books (Dune, Dune
Messiah, Emma, Persuasion), three members and their loans. Its commands, all in the root
context of `_commands/`:

| command | takes | its `Output` |
|---|---|---|
| `list_branches` | nothing | `names`, `total` |
| `count_overdue` | `branch` (optional) | `total` |
| `find_book` | `query`, `start` (a page) | `book_uids`, `titles`, `total` |
| `show_book` | `book_uid` | `book_uid`, `title`, `author`, `copies`, `on_loan` |
| `list_loans` | `book_uid`, `start` | `borrowers`, `due_dates`, `total` |
| `find_member` | `query` | `member_uids`, `names`, `total` |
| `list_member_loans` | `member_uid`, `start` | `titles`, `due_dates`, `total` |

The skills these pages write are in `tests/skill_tutorial_workflow/_skills/`, and their
previews in [previews/](previews/). The workflow lives under `tests/` because it is teaching
and test material: `tests/` is not in the fastworkflow wheel, and `fastworkflow/examples/`
is.

## The pages

1. [Fixed steps: a skill with no inputs](01-fixed-steps.md): `Signature`, `PlanGenerator`,
   steps as text, the preview.
2. [An input](02-an-input.md): an `Input` field, its description as the planner's
   instruction, a flat skill.
3. [Phases that pass a fact](03-phases-and-a-fact.md): a facts model, `fact_from`, reading
   a phase by its name, an answer code writes.
4. [Skipping a phase](04-skipping-a-phase.md): `return []`, `=None`, and "(only when ...)".
5. [Picking a row by name](05-pick.md): `pick(...)`.
6. [Wording the answer](06-answer-wording.md): `listed`, `count`, `num`.
7. [The shape of a call](07-the-shape-of-a-call.md): `__call__`, and calls that run flat.
8. [Once per item](08-fan-out.md): `@phase(over=...)`.
9. [Refusing a call](09-refusing-a-call.md): `validate_extracted_parameters` and
   `fastworkflow.Refused`.
10. [A facts model chosen by the input](10-a-facts-model-per-input.md):
    `@phase(output=lambda input: ...)`.
11. [Composing skills](11-composing-skills.md): a `PlanGenerator` that extends another
    skill's.
12. [Troubleshooting](12-troubleshooting.md): the load errors, what they mean, how to fix them.

## Three readers of a skill

A skill has three readers, and the pages say which one each line is for:

- **the planner** (a model) chooses the skill for a request and fills its `Input`. It reads
  the `Signature` and the skill's *template*: the steps written with every value unknown.
- **code** (the framework, no model) runs every step it can run mechanically, fills each
  phase's facts from the commands' outputs, and writes the answer when the steps already
  hold it.
- **the agent** (a model) runs what code cannot: an open step, a judgement, an answer that
  needs writing.

## Running these yourself

From the repository root:

```sh
fastworkflow skill preview tests/skill_tutorial_workflow --out docs/tutorials/skills/previews
python -m tests.test_skill_tutorials book_by_title '{"title": "Dune"}'   # a dry run
pytest tests/test_skill_tutorials.py
```

A *dry run* (`dry_run` in the test) runs a skill call offline, without any model: each
phase's steps as the runtime writes them, the steps code runs (by the rules the runtime
uses) against the toy commands, the facts filled from those runs, and the answer code
writes. Where code would hand over to the agent, it says so and stops.

The reference for everything here: [skill_file_layout.md](../../skill_file_layout.md) (the
one-page guide), [skill_phases.md](../../skill_phases.md) (phases at run time) and the
`fastworkflow.skill` module documentation.
