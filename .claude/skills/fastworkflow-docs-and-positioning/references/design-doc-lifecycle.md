# House design-doc lifecycle: skeletons to copy

> TEAM-PRIVATE: embeds content from uncommitted internal docs. Do not commit or publish this skill without the developer'sexplicit approval.

Extracted from the TurnResult saga (docs/turn_result_design*.md, epic `fix-vof`,
2026-06). This is the proven house process for any risky or large change. Verified
2026-07-09 against v2.22.2 (commit c33b9a5); exemplar commits: 42e9b3a (resolution),
9031942 (architecture review), 31b2d20 (final spec), afcbe01 (implementation).

Reminder before you start: **no git commits without the developer'sexplicit request in that
turn** — every "one commit per resolution" step below happens only when Dhar asks.
See `fastworkflow-change-control` for gating.

## Stage 1 — Design doc (`docs/<feature>_design.md`)

Skeleton:

```markdown
# <Feature> — Design

## 1. The symptom
<Verbatim transcript / API response showing the user-visible bug or need.
Lead with the concrete failure, not the abstraction.>

## 2. Root cause
<Narrative with file:line evidence for every assertion. Name what was ruled
OUT and how (the TurnResult doc proved producer/mapping/handler were all
correct before indicting the framework; the deterministic '/'-path NOT having
the bug was the diagnostic tell).>

## 3. Design
<Types, data flow, contracts. Distinguish what ships now vs roadmap.>

## N. Decision log (quick reference)
| # | Decision | Rationale |
|---|----------|-----------|
| 1 | <chosen option, and the rejected alternatives by name> | <why> |

## N+1. Open questions
<NEVER write "none remaining". The TurnResult doc did; the 48-finding review
found four internal contradictions and the section was publicly rewritten as
an admission. List real open questions or say what review would falsify.>
```

## Stage 2 — Adversarial review (`docs/<feature>_design_review.md`)

Review the design against the ACTUAL working tree at a pinned commit — re-verify the
design's factual baseline, don't just critique its logic (finding R37 existed because
the design's claimed baseline about existing persistence was factually wrong).

```markdown
# <Feature> Design — Critical Review
Reviewed against commit <sha>.

## Severity index
| Finding | Severity | One-line summary |
|---|---|---|
| R1 | critical | ... |

### R1. <Title stating the defect, not the topic>
<What the design says; what the code/requirements actually are; the concrete
failure scenario; file:line evidence.>
**Status:** open | resolved by A<#>
```

## Stage 3 — Beads epic, one child per finding

```
bd create "<feature> design review resolutions" --type epic   # e.g. fix-vof
# one child per finding, review-only:
bd create "R1: <title>" --parent <epic>
```

Work one finding at a time with the human. Review-only — no implementation inside the
epic. Discipline (bd memory `beads-flakiness-observed-2026-06-11...`): one bd write at
a time, verify `.beads/issues.jsonl` after each. `bd close --reason` was long held to
silently fail to persist; retested 2026-08-13 on bd 1.1.2 it persists 12/12 (see
change-control Rule 2). Verify the JSONL either way.

## Stage 4 — One docs commit per resolution

Exact shape (from `git show 42e9b3a`):

```
docs: resolve R<#> - <decision in one line, credit the decider>

Decision: <2–6 line summary of what was decided and what it supersedes/restores>.

- design doc: Amendment A<#>
- review doc: R<#> marked resolved
- beads: <epic>.<n> closed (epic <k>/<total>)
```

Each commit touches exactly three things: the design doc (append the amendment), the
review doc (mark the finding resolved), `.beads/issues.jsonl` (child closed).

Amendment format, appended to the design doc — never edit history:

```markdown
### A<#> — <Decision title> (resolves R<#>; supersedes decision <n>; corrects §<x>) — <date>
<The decision and its rationale. Traceability tags [A#]/[X#] are how later
docs cite this.>
```

## Stage 5 — Second-order architecture review (`docs/<feature>_architecture_review.md`)

After all R-findings are resolved, review the FINALIZED design with parallel
per-concern agents asking "what did the amendments themselves miss". The TurnResult
pass used ten concerns: performance, reliability, scalability, code size, readability,
observability, manageability, testability, ease-of-use, deployment. Findings numbered
X1..Xn. This pass caught, e.g., X1 (a "no write-time index" decision that meant
full-keyspace Redis SCANs — "the worst under-priced decision", independently flagged
by three of ten agents) and X6 (shrank the implementation slice from ~2.5k lines/120
files to ~500 lines/6 files).

## Stage 6 — Consolidated final spec (`docs/<feature>_design_final.md`)

Open with the supremacy clause (copy from turn_result_design_final.md lines 1–8):

```markdown
**Status: authoritative implementation spec.** This document consolidates
`<design>.md` (original design + Amendments A1–A<n>), `<review>.md` (<n> resolved
findings), and `<arch_review>.md` (cross-cutting fixes X1–X<n>, all adopted <date>).
Where this document conflicts with any of those, **this document wins**; they
remain the rationale archive. Traceability tags like `[A7]`/`[X3]` link back to
decisions.
```

End with a traceability table (final.md §18 style): every spec section → the A#/X#/R#
that produced it. Clearly fence "ships now" from "vN roadmap" — and remember future
readers will treat roadmap sections as unbuilt until code proves otherwise.

## Stage 7 — Learning checklist + teaching/pushback session

Two artifacts:

1. `docs/<feature>_learning_checklist.md` — legend
   `[ ]` not yet · `[~]` in progress · `[x]` mastered. Header: "Living doc. We check
   items off only after you've *demonstrated* understanding (explained it back or
   answered a quiz). Nothing is 'done' just because we discussed it."
2. `docs/<feature>_design_feedback.md` — the human pushback session transcript/record,
   re-verifying the SHIPPED code against the spec. This stage is not ceremony: the
   TurnResult session caught that the shipped v2.21 slice did NOT fix the user-visible
   bug (fixed in v2.21.1), and code-reading during review R43 found a live CLI hang
   (fix-5fv).

the developer'sreview protocol (bd memory `dhar-s-working-preference-for-the-turn-result...`):
ELI5-style explanation BEFORE decision questions; one finding at a time; no commits
until approval.

## Why this process earns its cost

- The adversarial review overturned a "no open questions" design (4 contradictions).
- The architecture pass cut the implementation 5x and reversed a resolved decision (X1
  reversed A23.1).
- The teaching session caught a shipped non-fix.
- Four-way traceability (R# → bead → commit → A# → final §) means every "why is it
  like this?" has a citable answer 13 months later — this reference file is proof.
