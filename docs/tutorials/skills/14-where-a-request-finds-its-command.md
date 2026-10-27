# 14. Where a request finds its command

**Goal:** know which commands a request can reach from where the workflow stands, how the
agent plans it, with a planning command that covers it or without one, and how code checks
that plan. Contexts exist so that each model that matches a request chooses among few
commands, and that holds for both kinds of command.

## What you declare

Two things, both already seen:

- **the bases of each context**, in `_commands/context_inheritance_model.json` (below). A
  book and a member are both records, so both have `Record`'s commands;
- **the record that encloses each record**, by `get_parent()` in the context's `_<Context>.py`
  (page 5): a book's and a member's parent is the library.

<!-- file: fastworkflow/examples/library/_commands/context_inheritance_model.json -->
```json
{
    "Book": {
        "base": ["Record"]
    },
    "Member": {
        "base": ["Record"]
    }
}
```

There is no other hierarchy file. A `context_hierarchy_model.json` left over from an older
workflow is ignored, with a warning that it can be deleted.

## The lookup path

A request is matched against one scope at a time, in this order, and the first scope that
recognises it wins:

1. the current context's own commands;
2. its bases' commands;
3. the same two steps for the record that encloses it (`get_parent()`), then for that
   record's parent, up to the root;
4. the global commands, in the `_commands/` root.

Standing on the book Dune, that is:

| scope | its commands |
|---|---|
| `Book` (own) | `show_book`, `list_loans`, `loan_report`, `shelf_report` |
| `Record` (a base) | `show_notes`, `add_note` |
| `Library` (encloses the book) | `list_branches`, `count_overdue`, `find_book`, `open_book`, `find_member`, `open_member`, and the planning commands of pages 1 to 12 that live in `_commands/Library/` |
| global | `library_hours` |

A scope that appears twice (a base shared by two records on the chain) is asked once.

**A command found in an enclosing scope runs on that scope's object**, and the workflow does
not move. "How many loans are overdue?", said on Dune, is recognised by the `Library` scope
and runs `count_overdue` on the library; the workflow is still on Dune afterwards, and the
run is filed as the library's. The object is chosen as soon as the scope recognises the
command, before its arguments are extracted, so a lookup that reads the object (a list of the
library's members, say) reads the right one.

**The same path serves a command named exactly**, by the agent or the user: `count_overdue`
typed on Dune runs on the library. The nearest scope wins, so a command hides one of the same
name farther out. A qualified name (`Library/count_overdue`) must name a context on the path.

## A planning command of a context

<!-- file: fastworkflow/examples/library/_commands/Book/loan_report.py -->
```python
"""loan_report: who has the open book on loan, and until when.

A planning command of the Book context: it is found only while the workflow stands on a
book. loans lists the book's loans; __result__ names each borrower with the due date.
"""
from fastworkflow.planning import full_list

from .list_loans import Signature as ListLoans


class Signature:
    """Say who has this book on loan and when each loan is due back."""

    plain_utterances = ["who has this book", "when is it due back"]


class PlanGenerator:
    def loans(self) -> ListLoans.Output:
        return [full_list("list_loans")]

    def __result__(self, loans):
        if not loans.borrowers:
            return "No one has it on loan."
        due = (f"{b} has it until {d}" for b, d in zip(loans.borrowers, loans.due_dates))
        return "; ".join(due) + "."
```

It lives in `_commands/Book/`, so it is offered only while the workflow stands on a book, as
`shelf_report` (page 10) is:

- **on Dune**, "who has this book?" is `loan_report`'s, the planning command of the nearest
  scope and the one written for the book the workflow stands on, and it runs on Dune:

  ```text
  Phase loans
    1. list_loans (full list)
    code runs: list_loans
    facts: list_loans.Output(borrowers=['Ann Lee', 'Raj Patel'], due_dates=['2026-11-02', '2026-10-20'], total=2)
  answer written by __result__:
    Ann Lee has it until 2026-11-02; Raj Patel has it until 2026-10-20.
  ```

- **on the library**, `loan_report` is not on the path, and "who has Dune?" goes to
  `book_loans` (page 12).

## How each scope answers

In each scope, matching answers one question over that scope's commands of both kinds: which
command, if any, is this request for? The answer is a command, a short list it is unsure
between, "not here", or a bare value (a reply such as "Dune" that names no command).

- **Execution commands** are answered by a small classifier per context, which knows the
  context's own commands, the four core commands (`go_up`, `reset_context`, `what_can_i_do`,
  `what_is_current_context`) and a "not here" label. "Not here" is trained on a few
  `plain_utterances` of every other command of the workflow, planning commands included. So
  write `plain_utterances` for a planning command as you would for an execution command:
  they also teach every other context that the request is not theirs.
- **Planning commands** are chosen by the agent as it plans, from its list of commands, where
  it sees each one with its `Signature` and its template (page 1), and uses it as the next
  section shows. With `JEV_API_KEY` in `fastworkflow.passwords.env`, the Jev decision model
  first picks the planning command that fits the request, by its description and
  `plain_utterances`, scope by scope along the same path, and the agent is shown that one
  first. Jev is TypeSafe's service, so this sends the request and the planning commands'
  descriptions to TypeSafe. The key is optional: without it the agent finds the planning
  command itself, and the `verify` option checks nothing (page 9).
- **An unsure answer that includes "not here"** keeps its candidates and the walk goes on;
  the user is asked once, at the end, among all the candidates kept, each named with its
  scope.

`fastworkflow train` trains one classifier per context. Each context's "not here" class holds
a few seeds of every other command, chosen once, when that command first joins the
workflow. So adding or removing a command retrains every classifier, while editing a
command's seeds retrains only its own; `fastworkflow train --regenerate-utterances` chooses
the seeds again. A workflow trained by a release before scoped lookup does not load, and
says to retrain with `fastworkflow train`.

## How the agent plans a request

In agent mode a request is split into its parts, and each part gets a run of the agent of
its own. **The run's first step is the agent's plan**: it calls its `plan` tool with the
steps it means to run, and until it has, the only other thing it can do is ask the user a
question. The plan is a step like any other, so it is in the turn's trace with the steps it
led to, and the agent revises it with another `plan` when a step fails or the user's answer
changes what is needed.

For a planning command that matches a part, the agent writes one of three plans:

- **The one step that calls it**, when it covers the whole part. "Who wrote Dune?" is the
  plan `book_by_title title=Dune`. The runtime runs that step itself, right after the plan,
  and the answer `__result__` writes is the answer the user reads, as written. Every page so
  far runs this way.
- **The call with steps before or after it**, when it covers only some of the part. "Who
  wrote Dune, and which branch holds it?" is `book_by_title title=Dune`, then `show_book`,
  which runs on Dune because the call left the workflow there (page 5). The call runs with
  its own phases, checks and `__result__`; the agent runs `show_book`, and writes the answer
  from both.
- **An adapted copy of its steps**, when the request needs some of them dropped or changed.
  "Open Raj Patel's record" can be the first two steps of `member_report`'s template,
  `find_member query=Raj Patel` and `open_member with the uid of the Raj Patel match`, with
  the rest left out. The agent runs the copy as ordinary steps, so `member_report`'s phases
  and `__result__` do not run.

The agent prefers the first plan, then the second: both run the command's own code. In
each, the command's arguments arrive as any command's do (page 9).

## The checks on a plan

Code checks every plan the agent writes against what your workflow declares, and puts its
notes in the plan step's result, for the agent to read. A note never stops a plan: the agent
revises it with another `plan`, or goes on. There are three checks:

- **A producer that no earlier step runs.** `open_member`'s `member_uid` names its producer,
  `find_member` (page 5), so a plan that opens Raj Patel's record before it has looked him up
  gets `Step 1 runs open_member, whose member_uid comes from find_member; no earlier step
  runs find_member.`
- **A context the plan does not reach.** Code follows the plan from where the workflow
  stands, moving it where each step's `descends_to` says (page 5). A step whose command
  cannot run where the plan then stands gets a note that names the command entering its
  context: `Step 2 runs list_member_loans while the plan is in Library; enter a member first
  (open_member).`
- **A matched planning command's steps left out.** When Jev matched a planning command and
  the plan neither calls it nor runs each of its steps, the note names the steps the plan
  lacks: `This request matches member_report. The plan neither calls it nor runs its steps
  list_member_loans and show_notes: call member_report, or add the steps the request needs.`
  For "what has Ann Lee borrowed" the agent adds `list_member_loans` and leaves out
  `show_notes`, which `member_report` runs only when notes are asked for. Code names the
  steps; only the agent can tell which of them the request needs.

So the declarations of page 5 do two jobs: they tell the agent where values and contexts come
from, and they check its plans. A `from_` or a `descends_to` you leave out is a check that
never runs.

## What the agent is shown

The agent writes each plan from the list of commands its run is shown: for a part of a
request a planning command covers, as above, and for one no planning command covers, from
execution commands alone. No plan is kept for the next request. The agent keeps the same list
while it runs the plan.

**While the workflow is small, the list holds every command.** The library's commands fit
whole, so for "list Mia Chen's loans and note that she was reminded", which `member_report`
covers only in part, the agent is shown every command of the workflow in full, each planning
command with its template, and writes the plan:

```text
member_report member=Mia Chen loans=true
add_note text=she was reminded
say which books Mia Chen has on loan, and that the note was added
```

`member_report` leaves the workflow on Mia Chen (page 5), so `add_note` runs on her record.

**A larger workflow gets a shortlist.** Past a fixed size, listing every command in full makes
the prompt of every step of the run long and the agent's choices worse, so the run gathers a
shortlist for the request instead. For the same request it would hold:

1. **The commands that match**, in every scope on the path and in every context the library
   can navigate into through a declared `descends_to` (`Book`, `Member`, and their base
   `Record`): `find_member`, `list_member_loans`, `add_note`, and `member_report`, the
   planning command that covers part of it, with its template.
2. **What those need**, added until nothing new comes in: the producers of their inputs,
   read from producer `from_` edges (`open_member`'s `member_uid` comes from
   `find_member`); the commands that navigate to their contexts (`open_member` enters
   `Member`); the callees of each planning command on the list; and the core commands.

The agent sees the shortlist in full, each planning command on it with its template, and
every other context as one line. The templates count toward the size at which the shortlist
starts. The list follows the workflow: when a step moves it into another context, the list
is shown again from there, and a later plan gathers the shortlist again.

So in a large workflow, **the edges you declare decide what the agent's plan can reach**: a
producer `from_` on an input brings its producer onto the shortlist, and a `descends_to` on
a navigation command brings in the way to its context. A navigation command that moves the
workflow without declaring `descends_to` is reported the first time it does it (page 5). A
command neither edge reaches is still listed, as one line under its context, and the agent
sees it in full once its run enters that context.

When the agent writes a planning command as a step, that command's arguments are checked
against the request and what the turn's earlier steps found, not against the step the agent
wrote: a value the agent made up is in neither, and is refused.

If the agent keeps writing the same plan for one kind of request, that is the hint to write a
command for it: an execution command if code can decide every step, and a planning command if
the agent should run, judge or recover part of it (page 1). Its plans are the plan steps in
the turn's trace, so they are there to read.

## Assistant mode

`fastworkflow run --assistant` runs one execution command per message, with no agent, so
nothing is planned. A request there that needs a planning command is refused with a pointer
to agent mode, never handed to the agent silently, and naming one gets
`member_report is a planning command; it runs in agent mode.`

## What you learned

A request is matched one scope at a time: the current context, its bases, each enclosing
record and its bases, then the global commands; the first scope that recognises it wins, and
the command runs on that scope's object. Planning commands belong to contexts like execution
commands. Every part of a request gets a run of the agent, whose first step is its plan,
written from a list that shows each planning command with its template: the agent calls one
whole for a request it covers, with other steps for one it covers in part, or adapts a copy
of its steps, with no key needed; `JEV_API_KEY` lets Jev show it first the one that fits. A
request no planning command covers is planned from execution commands alone. Code checks
each plan against your producer `from_` and `descends_to` declarations, and the agent reads
the notes. The list holds every command while the workflow is small, and a shortlist that the
same declarations complete once it is large.
