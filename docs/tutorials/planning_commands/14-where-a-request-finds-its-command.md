# 14. Where a request finds its command

**Goal:** know which commands a request can reach from where the workflow stands, and how it
is planned, with a planning command that covers it or without one. Contexts exist so that
each model that matches a request chooses among few commands, and that holds for both kinds
of command.

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

For a request in natural language, the global commands are asked **first**. If the global
scope names one command with confidence, that command runs, bound to the root object, and the
lookup stops there. Otherwise the scopes above are asked in order, as the table shows. An exact
command name (below) is not affected: it always takes the nearest scope.

A bare value (a reply such as "Dune" that names no command) ends the lookup: no enclosing scope
is asked about it.

**A command found in an enclosing scope runs on that scope's object**, and the workflow does
not move. "How many loans are overdue?", said on Dune, is recognised by the `Library` scope
and runs `count_overdue` on the library; the workflow is still on Dune afterwards, and the
run is filed as the library's. The object is chosen as soon as the scope recognises the
command, before its arguments are extracted, so a lookup that reads the object (a list of the
library's members, say) reads the right one.

**The same path serves a command named exactly**, by the agent or the user: `count_overdue`
typed on Dune runs on the library. The nearest scope wins, so a command hides one of the same
name farther out. A qualified name (`Library/count_overdue`) must name a context on the path.
Re-issuing a command that opens the record the workflow already stands on runs it in place; it is
not refused as "already in context".

**For authors:** the first word of a request is matched against command names along the whole
path. A one-word command name in an enclosing context can therefore capture a request that only
starts with that word. Give commands names that a sentence about something else does not begin
with, or name the context in the request.

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

    plain_utterances = [
        "who has this book",
        "when is it due back",
        "is this book out, and until when",
        "who is holding this one right now and what's the return date",
        "tell me who borrowed this book and when it comes back",
        "is this one on loan to anyone",
        "when will this copy be returned, and to whom",
        "who's got this book at the moment",
    ]


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
  `what_is_current_context`) and a "not here" label. "Not here" is trained on one hand-written
  seed of every other command of the workflow, planning commands included. So write
  `plain_utterances` for a planning command as you would for an execution command: its seed also
  teaches every other context that the request is not theirs.
- **Planning commands** are labels too. Each classifier whose scope holds a planning command has
  that command as one of its labels, so a request can reach it the way it reaches an execution
  command, and its seed teaches the classifier what it is for. The classifier proposes; the
  planner then writes the plan, and the agent runs it (below). With `JEV_API_KEY` in
  `fastworkflow.passwords.env`, the Jev decision model first picks the planning command that fits
  the part, by its description and `plain_utterances`, scope by scope along the same path, and the
  agent is shown that one first. Jev is TypeSafe's service, so this sends the request and the
  planning commands' descriptions to TypeSafe. The key is optional: without it the planner finds
  the planning command in its list, and the `verify` option checks nothing (page 9).
- **An answer that is unsure between commands** stops the walk: the user is asked which one
  is meant. **A "not here" answer** moves the walk on to the next scope.

`fastworkflow train` trains one classifier per context. Each context's "not here" class holds
one hand-written seed of every other command, planning commands included. The seed is chosen
once, when that command first joins the context, and the classifier records which seed it used.
So editing one command's seeds retrains only the classifiers that hold that command (and any
that recorded its seed), while adding or removing a command retrains every classifier.
`fastworkflow train --regenerate-utterances` chooses the seeds again. A workflow trained by an
earlier release does not load, and says to retrain with `fastworkflow train`.

The first train after an upgrade to this release **generates** training utterances for each
planning command from its `plain_utterances`, because planning commands are now labels of the
classifiers. That costs LLM calls, which earlier releases did not make. Later trains reuse them.
The library example gives each command eight seeds, which is the count to follow.

## How a request is planned

In agent mode a request is split into its parts, and each part gets a run of the agent of
its own. **Before the run, the planner writes its plan**: a numbered list of steps, written
from the list of commands described below. The agent then runs those steps. The plan is in
the turn's trace (`fw.planner.plan`), and the planner writes the rest of it again
(`fw.planner.replan`) when a step's arguments are refused or the user's answer to a question
changes what is needed.

**Before anything runs, the parts are reviewed.** A request with several parts, or a part with no confident
match, is shown back as a proposed plan. Each part has its goal, its inputs (a value, or `from sub-task N`) and
its means: a planning command, a plain command, or build from scratch. Reply `accept` to run the plan, `no` to
stop, or change it with `change 1: means=plain:find_member`, `change 2: inputs=query=from sub-task 1` or
`drop 3`. A reply in your own words is read into those changes by one model call, and shown back for you to
confirm. A part with one confident match, or an exact command name, runs at once with no review. A review
admits 8 replies or 15 minutes. An unattended turn, or a host that is not a person (MCP, REST, another agent),
runs the proposed plan as it stands and marks it `not_reviewed`.

With `JEV_API_KEY` set, Jev first decides whether a planning command fits the part, one
scope at a time along the lookup path above. A confident match is planned in one of two ways:

- **A typed call**, the usual way. The planner fills the command's typed inputs from the
  request, reading a shorter list in which each command is its name, the first sentence of
  its description and its parameter names. Code checks the values and expands the command's
  steps for them, so the planner never rewrites them. When the request asks for more than the
  command does, the planner adds steps before or after the command's own. "Who wrote Dune?"
  is the call `book_by_title title=Dune` alone. "Who wrote Dune, and which branch holds it?"
  is the same call with `show_book` after it, which runs on Dune because the call left the
  workflow there (page 5). The call runs with its own phases, checks and `__result__`; when
  it is the whole plan, the answer `__result__` writes is the answer the user reads, as
  written.
- **Its template, adapted**, when the typed call fails. The planner gets the request with
  the command's template after it (`A job like this is usually done as: ... Adapt this to
  the request: drop steps it does not need and add steps it asks for.`) and writes ordinary
  steps. "Open Raj Patel's record" can be the first two steps of `member_report`'s template,
  `find_member query=<member>` and `open_member with the uid of the <member> match`, with
  `<member>` read as Raj Patel and the rest left out. The agent runs them as ordinary steps, so `member_report`'s phases and
  `__result__` do not run.

In each, the command's arguments arrive as any command's do (page 9).

**A value the request does not give is asked for.** When the checks refuse an argument of
the typed call (one the request does not name, or a placeholder, a pronoun or a kind of
record standing in for a name), an attended turn stops on a question and runs no step:
`I can't run member report yet: your request does not say which member to use. Which member
should I use?` The reply runs the call again with the value. An unattended turn never asks:
the planner plans from the template instead.

Without `JEV_API_KEY` no planning command is matched first: the planner plans every part from
the listing (the full list, or a large workflow's shortlist), where each planning command is shown with its
template, and may call one as a
step or copy its steps. An INFO line naming `JEV_API_KEY` says so, once per workflow.

So the declarations of page 5 tell the planner and the agent where values and contexts come
from: a `from_` or a `descends_to` you leave out is a path neither of them is shown.

## What the planner is shown

The planner writes each plan from the list of commands it is shown: for a part of a
request a planning command covers, as above, and for one no planning command covers, from
execution commands alone. No plan is kept for the next request.

The list shows the commands of the enclosing records and the global commands as callable
here, each with its scope, so the plan never names a command the executor would refuse. A
step that names one of them runs it on the record that owns it, as an exact name does.

**While the workflow is small, the list holds every command.** The library's commands fit
whole, so for "list Mia Chen's loans and note that she was reminded", which `member_report`
covers only in part, the planner is shown every command of the workflow in full, each planning
command with its template, and writes the plan:

```text
member_report member=Mia Chen loans=true
add_note text=she was reminded
say which books Mia Chen has on loan, and that the note was added
```

`member_report` leaves the workflow on Mia Chen (page 5), so `add_note` runs on her record.

**A larger workflow is shortlisted.** The planner is shown every command of the workflow, each planning
command with its template, while the full list is 7000 characters or less. A larger workflow that has trained
heads is shown a shortlist instead, built like this for a request:

1. **The commands that match**, in every scope on the path and in every context the library can
   navigate into through a declared `descends_to` (`Book`, `Member`, and their base `Record`):
   `find_member`, `list_member_loans`, `add_note`, and `member_report`, the planning command that
   covers part of it, with its template.
2. **What those need**, added until nothing new comes in: the producers of their inputs, read from
   producer `from_` edges (`open_member`'s `member_uid` comes from `find_member`); the commands that
   navigate to their contexts (`open_member` enters `Member`); the callees of each planning command
   on the list; and the core commands.

In that case the planner sees the shortlist in full, each planning command on it with its template,
and every other context as one line. A workflow with no trained heads is listed in full at every size. A part
bound to a plain command in a reviewed plan is shown that command, its producers and its navigation, not the
whole list. The list follows the workflow: when a step moves it into
another context, the list is shown again from there.

So in a large workflow, **the edges you declare decide what the plan can reach**: a producer `from_`
on an input brings its producer onto the shortlist, and a `descends_to` on a navigation command brings
in the way to its context. A navigation command that moves the workflow without declaring
`descends_to` is reported the first time it does it (page 5). A command neither edge reaches is
still listed in the full list, and it is the shortlist that would leave it out.

When a plan has a planning command as a step, that command's arguments are checked against
the request and what the turn's earlier steps found, not against the step as written: a value
the planner made up is in neither, and is refused.

If the planner keeps writing the same plan for one kind of request, that is the hint to write a
command for it: an execution command if code can decide every step, and a planning command if
the agent should run, judge or recover part of it (page 1). Its plans are the `fw.planner.plan`
spans in the turn's trace, so they are there to read.

## Assistant mode

`fastworkflow run --assistant` runs one execution command per message, with no agent. A planning
command named exactly, as `count_books` or `/count_books`, runs there when it is mechanical: code runs
every step and nothing needs the agent. A planning command that needs the agent, such as
`member_report`, is refused, with the reasons and what to change, and the message points to agent
mode. A request that is not an exact name is not matched to a planning command in this mode. Direct
API and MCP calls refuse a planning command.

## What you learned

A request is matched one scope at a time: the current context, its bases, each enclosing
record and its bases, then the global commands; the first scope that recognises it wins, and
the command runs on that scope's object. Planning commands belong to contexts like execution
commands. Every part of a request gets a plan from the planner, then a run of the agent that
carries it out. With `JEV_API_KEY` set, a planning command that fits the part is a typed call:
the planner fills its inputs, code checks them and writes its steps, and the planner adds the
steps the request asks for beyond it; a value the request leaves out is asked for in an
attended turn. When the call fails, the planner adapts the command's template. Without the key,
or for a part no planning command covers, the planner plans from the list of commands, which
shows each planning command with its template. The planner is shown the full list of commands up to 7000
characters; above that, the shortlist of a large workflow, which your producer `from_` and `descends_to`
declarations complete.
