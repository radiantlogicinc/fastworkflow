# 7. Words the runtime reads

**Goal:** write every step code acts on with a helper, so that a value is never misread and a
typo is a Python error instead of a step that quietly does something else. Then: answers that
read well for any number of rows, "2 books (Emma, Persuasion)", "1 book", "none (0 books)".

## Steps

A step is a line of text the agent reads. Code reads some lines too, and acts on their words:
it runs a command whose values are all written, reads every page of a listing, picks a row by
name, or leaves a decision to the agent. Each of those has a helper in `fastworkflow.planning`
that writes the words for you:

```python
run("find_book", query=input.title)                          # find_book query=Dune
run("list_loans", all_pages=True)                            # list_loans (full list)
run("open_book", named=input.title)                          # open_book with the uid of the Dune match
when("the member asked for them", run("show_notes"))         # only when the member asked for them, show_notes
do_not_run("add_note")                                       # do not run add_note
```

| written | what the runtime does with it |
|---|---|
| `run(command, ...)`, every required value given | code runs it |
| `run(command, ..., all_pages=True)` | code runs it, then each later page, up to 20 pages (`CODE_MAX_PAGES`) |
| `run(command, named=...)` | code opens the named row of a listing the call read, or its only row; else the agent picks (page 4) |
| `when(condition, step)` | the agent decides whether to run the step |
| `do_not_run(command)` | nothing runs: code passes the step over, and the agent is told not to |
| a plain string | the agent does what it says |

- **The rule:** a plain string is the agent's, and everything code does is a helper call.
- **Decide in Python what Python can decide.** A condition on the input or on the facts is an
  `if` in the phase method (page 4), decided on the real values. `when` is for a judgement
  only the agent can make while it runs.
- **Values are written whole.** `run` writes each value so that the step
  parser reads it as it is: a value that holds a bracket, an `=`, a `;`, a `, `, a quote or
  `...` is quoted. A value of `None` is left out, and a keyword the command's `Input` does
  not have fails the load.
- **A note** is text in parentheses after a step, for the agent; it is never sent to the
  command. Add one with `.note(...)`: `run("find_book", query="Dune").note("the first page")`
  writes `find_book query=Dune (the first page)`. Code reads a note only for the words that
  ask for every page (`full list`, `every page`, `all pages`, `until complete`,
  `page through`), which `all_pages=True` writes as `(full list)`. A note that holds a condition
  (`(only when the member asked)`) loads, but `preview --check` warns: code does not read notes, so the step would run every
  time. Write `when(...)`, or an `if`. Bare `only` in a note (`first page only`) is not a
  condition word for this warning.
- **Typed by hand, the same words still run**, as workflows written before the helpers have
  them, but `fastworkflow command preview --check` warns about each and names the helper to
  write instead. A misspelt helper fails where you can see it; a misspelt phrase did not.

## Pages

`find_book`, `list_loans` and `list_member_loans` are **paged**: their `Input` has an integer
`start`, the first row of the page. A field of another type with one of those names, such as a
`start` that is a date, is an ordinary field. A paged command's `Output` must declare
`total`, how many rows there are on every page, and the load fails without it; its list
fields are the rows of the page it read. That is all the runtime needs:

- `run(..., all_pages=True)` reads page after page until the rows read reach `total` (at most
  20 pages, `CODE_MAX_PAGES`), and the phase's facts hold every row, with the one `total`;
- when rows remain after a page, the response the framework writes for the command ends with
  `Rows 1-10 of 14; next page: start=10.`, so the agent can read on too;
- a step that names a paged command with `run` and no `all_pages` reads one page. The preview
  says which steps read every page and which read one.

## The answer

<!-- doctest -->
```pycon
>>> count_and_names_read(["Emma", "Persuasion"], 2, "book")
'2 books (Emma, Persuasion)'
>>> count_and_names_read([], 0, "book")
'none (0 books)'
>>> count_and_names_read(["Dune"], 12, "book")
'12 books (Dune; 1 of 12 shown)'
>>> count(1, "book"), count(3, "copy", "copies")
('1 book', '3 copies')
```

- **`count_and_names_read(names, total, noun)`** writes a list fact with its total: "N books (a, b)",
  "none (0 books)", and "K of N shown" when the page read holds fewer rows than the total.
  Pass the list and the total from the same command's `Output` (`loans.titles`,
  `loans.total`), so the note is true.
- **`count(number, noun, plural=None)`** writes "1 book", "2 books", "3 copies". It is public
  and safe on a placeholder: use it rather than an inline `n == 1`, which breaks the preview.

They are for the text `__result__` returns (page 3), and for any step that names a number of
rows. A wording several commands share goes in a `_` helper, as `_wording.py`'s `copies`
wraps `count` (page 4).

## Where the sample uses them

| command | helper | what it writes |
|---|---|---|
| `count_books` (page 3) | `count(find.total, 'book')` in `__result__` | Persuasion matches 1 book. |
| `book_by_title` (page 4) | `run("open_book", named=input.title)`, and `copies(book.copies)` in `__result__` | ... the library owns 1 copy. |
| `member_report` (page 6) | `run("list_member_loans", all_pages=True)`, `count_and_names_read(loans.titles, loans.total, 'book')` | Raj Patel has 2 books (Dune, Emma) on loan. |
| `member_report` (page 6) | `count_and_names_read(notes.notes, notes.total, 'note')` | Raj Patel's record holds 1 note (prefers email reminders). |
| `author_books` (page 8) | `copies(b.copies)` in `__result__` | Emma (2 copies); Persuasion (1 copy). |
| `shelf_report` (page 10) | `run("list_loans", all_pages=True)`, `count_and_names_read(loans.borrowers, loans.total, 'borrower')` | ... on loan to 2 borrowers (Ann Lee, Raj Patel). |
| `loan_report` (page 14) | `run("list_loans", all_pages=True)` | Ann Lee has it until 2026-11-02; Raj Patel has it until 2026-10-20. |

## What the agent plans from

The step helpers write the same words in the template as at run time:
`run("list_loans", all_pages=True)` reads `list_loans (full list)` there too, and `run` writes a
placeholder as it is (`find_book query=<title>`). On a placeholder each answer helper writes
the placeholder in its place, with the plural, since the number is not known:
`count_and_names_read(loans.titles, loans.total, 'book')` reads `<loans.total> books (<loans.titles>)`
in the preview's answer.

## What you learned

Write every step code acts on with a helper (`run`, with `all_pages=True` or `named=`, `when`,
`do_not_run`), and decide in Python whatever Python can decide; a note, added with
`.note(...)`, is for the agent and never holds a condition. A command paged by an integer
`start` declares `total` in its `Output`. Write answer facts with `count_and_names_read` and
`count`, so the sentence is right for zero, one or many rows.
