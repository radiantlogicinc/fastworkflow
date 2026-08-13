---
name: fastworkflow-change-control
description: >
  Load this skill BEFORE making any change to the fastWorkflow repo that touches git, beads
  (bd), version numbers, tests, or documents in docs/. Triggers: "commit this", "push",
  "bump the version", "close the issue", "bd create/update/close", "delete this test",
  "clean up docs/", "should this get a design doc?", "is this a patch or a minor?",
  "can I commit the skills library?". Also load when you see symptoms like a bd close that
  didn't stick, an epic that looks shipped but shows open, or untracked team-private docs
  in git status. Do NOT load for debugging runtime failures (use
  fastworkflow-debugging-playbook), for running/operating the app (fastworkflow-run-and-operate),
  or for how-to-test mechanics (fastworkflow-validation-and-qa).
---

# fastWorkflow Change Control

> TEAM-PRIVATE: embeds content from uncommitted internal docs. Do not commit or publish this skill without the developer'sexplicit approval.

How changes are classified, gated, reviewed, versioned, and released in this repo — and the
four non-negotiable rules, each earned through a real incident. This is policy inferred from
what actually happened in git history and the beads tracker, not aspiration.

**Jargon, defined once:**
- **bd / beads** — the git-backed issue tracker used here (`bd` CLI; source of truth is `.beads/issues.jsonl`). Issue IDs look like `fix-vof`, children like `fix-vof.1`.
- **Epic** — a bd issue of type `epic` with child issues.
- **tau-bench / tau2-bench (τ²-bench)** — public customer-service agent benchmarks this project reports scores on. "Parity" = running under standard, unmodified benchmark conditions.
- **Design doc of record** — a markdown file in `docs/` that a change was reviewed against (e.g. `docs/turn_result_design_final.md`).
- **Adversarial review** — a structured pass where reviewers (here: parallel AI review agents) attack a design and every finding gets a written resolution.

---

## 1. When to use / when NOT to use

Use this skill when deciding **whether and how a change may land**: commit/push etiquette, bd workflow, design-review gating, version bumps, test removal, and what may never enter the public repo.

| Your actual need | Use instead |
|---|---|
| Why the architecture is shaped this way; invariants | `fastworkflow-architecture-contract` |
| Triage a runtime/build/train failure | `fastworkflow-debugging-playbook` |
| History of past investigations and dead ends | `fastworkflow-failure-archaeology` |
| What counts as test evidence; adding tests | `fastworkflow-validation-and-qa` |
| Env vars / CLI flags catalog | `fastworkflow-config-and-flags` |
| Running the CLI / server / examples | `fastworkflow-run-and-operate` |
| tau-bench mechanics and our published claims | `fastworkflow-taubench-reference` |
| Executing the tau2 E0–E25 campaign itself | `tau2-reliability-campaign` |
| Doc style, publication boundaries, external claims | `fastworkflow-docs-and-positioning` |

---

## 2. The Four Non-Negotiables

Each rule exists because something went wrong. Cite the incident when enforcing the rule.

### Rule 1 — NEVER `git commit` or `git push` without the developer's explicit request *in that turn*

- **Rule:** No agent commits or pushes on its own initiative. A session-close protocol, a "wrap up" instruction, or a finished task does NOT override this. Explicit request, that turn, every time. Files written into the repo tree may still be private/team-only.
- **Rationale:** The repo is PUBLIC on GitHub. Anything pushed is published to the world instantly; team-private strategy documents live *inside the working tree* as untracked files, so blanket staging (`git add -A`, `git add .`) is a publication event.
- **Incident:** 2026-07-08 — a private planning doc was auto-committed and pushed to the public repo by an agent following a session-close routine. Recovery required a **git history rewrite** on a public repository. Recorded as a bd memory (`bd memories --json`, key `never-git-commit-or-push-anything-without-dhar`).
- **Corollaries:** stage files by explicit path only, never `-A`/`.`; check `git status` for untracked private docs before any commit you *were* asked to make (see §7 for the do-not-commit list).

### Rule 2 — One bd write at a time; verify the JSONL after each

- **Rule:** Serialize all bd write commands. After every write, confirm it landed: `grep '"id":"<issue-id>"' .beads/issues.jsonl` and check the field you changed. **Verify by reading the file, not by trusting the command's output** — that is the durable half of this rule and it has never stopped being necessary.
- **Rationale:** The bd backend (a Dolt SQL server; repo-root `config.yaml` is its config, port 3307) can wedge into a state where every command re-imports `issues.jsonl` into an empty DB and **async exports race and clobber each other**. `.beads/issues.jsonl` is the source of truth; when bd is wedged, patching the JSONL directly is safe because every command re-imports it.
- **Incident:** observed 2026-06-11 during the fix-vof review epic; recorded as bd memory `beads-flakiness-observed-2026-06-11-in-the`. Likely collateral: epic `fix-7kp` is a duplicate (created seconds apart) of the closed `fix-2jo` — same title, one closed, one open.
- **`--reason` — retested 2026-08-13 on bd 1.1.2, and it persists.** This rule used to say `bd close <id> --reason "..."` reports success but silently fails, and to add the close note through a separate verified update instead. Twelve consecutive closes with `--reason` (the fix-dzs epic and its eleven children) all landed, verified by reading `close_reason` out of the JSONL afterwards. So the workaround is no longer required. The 2026-06-11 observation is left standing rather than deleted: it was made on an older bd, and if the Dolt backend wedges again the symptom may return. Verify the write either way.
- **Corollary, still true and cheaper to get wrong:** a `bd` write updates the Dolt DB, not `.beads/issues.jsonl`. Export with `bd export -o .beads/issues.jsonl` (the `-o` is required since 1.1.2; bare `bd export` writes to stdout) or the close is invisible to git. See `.cursor/skills/beads-workflow/SKILL.md`.
- **What a stale-`old_value` chain in `.beads/interactions.jsonl` means:** the flakiness leaves a fingerprint — repeated identical transitions each reporting a prior state that is already out of date (e.g. `fix-16d`: `in_progress` four times, then `open` four times, then `closed` four times). A 2026-08-13 audit found 225 such discontinuities, clustered on 2026-06-11, 2026-08-02 and 2026-08-03, and **none** after that. They are harmless: `old_value` is a breadcrumb nothing consumes, and every actual status in `issues.jsonl` was correct. Do not "repair" that log — it is a truthful record of what bd emitted, and rewriting it would erase the only evidence those episodes left.

### Rule 3 — Never let tests or experiments wipe trained example models

- **Rule:** Nothing may delete or retrain-in-place `fastworkflow/examples/*/___command_info` (the trained intent-model artifacts: `tinymodel.pth`, `largemodel.pth`, `threshold.json`, ...). Tests and experiments that need training must `copytree` the workflow into a temp dir (`tmp_path_factory`) and train the copy.
- **Rationale:** These artifacts are gitignored (`.gitignore` lines 2–5), exist only because a developer ran `fastworkflow train` locally with real API keys, and other tests depend on them silently. Destroying them costs a paid retraining run and breaks unrelated tests with confusing `FileNotFoundError`s.
- **Incident:** **fix-0hb**, fixed in commit `fa97b48` (2026-06-15). A module-scoped fixture in `tests/test_train_modern_stack.py` ran `shutil.rmtree(<workflow>/___command_info)` on the REAL bundled `hello_world` example at setup AND teardown. The *next* full-suite run lost `threshold.json` and 4 `test_fastapi_service.py` tests failed. Fix: train into a temp copy; suite went 478 passed / 0 failed and is now idempotent.

### Rule 4 — tau-bench benchmark parity is sacred

- **Rule:** Never modify tau-bench / tau2-bench tools or tasks for benchmark runs. Any nonstandard trade must be **disclosed in the results write-up, never silent**.
- **Rationale:** The project's headline external claim is small-model parity on tau-bench; a silently modified harness would invalidate the claim and the credibility behind it. The team-private tau2 plan encodes this as a design constraint, e.g. (`docs/tau2_retail_reliability_implementation_plan.md`): line 43 — "**Tool layer** | The Sierra/τ-bench tool implementations in `tools/*.py`. Defines benchmark parity — we do not modify these."; line 199 — "Pinning sim temperature/seed raises the ceiling but changes standard benchmark conditions — **a disclosed trade if we make it**."; line 630 — new search commands must be "fastWorkflow command-layer additions over `load_data()`, benchmark-parity-safe", with "the τ² tools themselves are untouched".
- **Incident:** none yet — this is the one rule established *preventively*, because the tau2 reliability program (experiment cards E0–E25) will create constant temptation. Keep it that way.

---

## 3. Change classification: which gate does my change need?

There is no written gating policy; this table is **de-facto policy inferred from git history**, stated here so it becomes explicit. When in doubt, ask Dhar — he is the only approver (421 of 450 commits on main).

| Tier | What qualifies | Required artifacts | Precedent |
|---|---|---|---|
| **T0 — plain fix** | Bug fix, dependency bump, perf fix, cache/staleness fix; no public contract change; roughly ≤ ~200 lines across ≤ ~4 source files | bd issue (before code), tests, patch version bump in the same commit, `fix: vX.Y.Z — ...` subject | `b5747df` v2.22.1 (fingerprint cache invalidation, 186 insertions, 4 files, no design doc); `033132f` v2.21.5 (session eviction, fix-04r); `0e68284` v2.22.2 (dep bumps) |
| **T1 — designed change** | New subsystem or concurrency/persistence semantics; new files; minor version bump; but contract already settled | Design doc committed to `docs/` alongside code; bd **epic** with step-scoped children; verification evidence in close notes | `cf3eeae` v2.22.0: `fastworkflow/run_fastapi_mcp/turns.py` (424 lines) + `docs/fastworkflow_turns_async_execution_design.md` (538 lines) + epic fix-85g with 13 children, shipped Step 1 only; close notes record "full suite 457 passed" |
| **T2 — adversarial-review change** | Changes the **data model, public wire contract, or system-of-record semantics** — anything many future decisions will sit on | The full **fix-vof pipeline** (below) | TurnResult redesign, v2.21 |

### The fix-vof model (T2), as a checklist

The TurnResult redesign is the repo's reference process for high-stakes changes. Gate rationale and cost/benefit: [references/fix-vof-case-study.md](references/fix-vof-case-study.md) (which points to the reusable procedure in `fastworkflow-research-methodology` and the full chronicle in `fastworkflow-failure-archaeology`).

1. [ ] **Design doc** capturing origin bug, root cause, type algebra, decision log, amendments — `docs/turn_result_design.md` (85,376 bytes / 1424 lines; 22-entry decision log; Amendments A1–A47).
2. [ ] **Adversarial review** producing numbered findings — `docs/turn_result_design_review.md` (110 KB, 48 findings R1–R48, severity-indexed).
3. [ ] **One bd child per finding** under a review-only epic (`fix-vof`, 48 children, "review-only — no implementation. Work one finding at a time").
4. [ ] **One docs-commit per resolution**, each a decision record. Verified format of `42e9b3a` ("docs: resolve R1"): decision + rationale in body, then a trailer block: `- design doc: Amendment A7 / - review doc: R1 marked resolved / - beads: fix-vof.1 closed (epic 7/48)`.
5. [ ] **Second-order review** of the *resolved* design by ten parallel review agents — `docs/turn_result_architecture_review.md` (concerns X1–X12). This stage reversed one ratified decision (X1) and found a two-amendment contradiction (X3).
6. [ ] **Consolidated final spec** that supersedes everything — `docs/turn_result_design_final.md` ("where this conflicts with any of those, this document wins").
7. [ ] **Human pushback/teaching session** — `docs/turn_result_design_feedback.md` (354 lines). This stage caught that the shipped v2.21 slice did NOT actually fix the user-visible bug.
8. [ ] **Learning checklist** for mastery tracking — `docs/turn_result_learning_checklist.md`.
9. [ ] Only then: implementation, sliced minimal (X6 shrank the plan from ~2.5k lines/120 files to ~500 lines/6 files, user-approved).

**Gate triggers for T2** (any one suffices): breaking wire change (fix-qtq is explicitly labeled "a BREAKING WIRE CHANGE" and is held at T2-pending); new persistent record type or serialization format; semantics of success/failure reporting; anything the tau2 campaign will report externally.

**Review-interaction rule (bd memory, established 2026-06-10, R11):** in design reviews with Dhar, ALWAYS give an ELI5 plain-language explanation of each finding BEFORE asking decision questions via AskUserQuestion.

---

## 4. Beads discipline

**Issue before code.** Every T0+ change gets a bd issue first; discovered work gets a linked issue (`--deps discovered-from:<parent-id>`). `AGENTS.md` mandates bd for ALL tracking — no markdown TODOs, no duplicate trackers. Close notes must carry verification evidence (the house pattern: "Step 1 implemented & verified (full suite 457 passed)").

**Doc-rot note:** `.cursor/rules/dev_workflow.mdc` and `taskmaster.mdc` (both `alwaysApply: true`) still prescribe **Task Master**. Beads is the live system; the taskmaster rules are stale. Do not adopt Task Master.

### Command reference (verified against this repo's bd)

| Action | Command | Sharp edge |
|---|---|---|
| Ready work | `bd ready --json` | — |
| Create | `bd create "Title" --description="..." -t bug\|feature\|task\|epic -p 0-4 --json` | one write at a time (Rule 2) |
| Claim | `bd update <id> --status in_progress --json` | verify JSONL after |
| Close | `bd close <id>` then verify | **do not use `--reason`** (Rule 2); record the close note via a verified update instead |
| Inspect | `bd show <id> --json` / `bd list --status open` / `bd stats` | read-only, always safe |
| Memories | `bd memories --json` | `bd memories show <name>` is invalid ("accepts at most 1 arg"); use `--json` or a search term |
| Verify a write | `grep '"id":"<id>"' .beads/issues.jsonl` | JSONL is source of truth |

**Flakiness protocol** (from the 2026-06-11 memory): one bd write at a time → verify `.beads/issues.jsonl` → if the Dolt server wedges (every command re-imports into an empty DB), restarting helps only partially; patching `issues.jsonl` directly is safe. Note `.beads/issues.jsonl` currently sits staged-but-uncommitted — per Rule 1 it awaits the developer'scommit; do not commit it yourself.

### The stale-epic problem — needs owner adjudication, do NOT adjudicate yourself

These epics look shipped-but-open. The code demonstrably exists on main, yet children show open. Closing them requires a decision only Dhar can make (residual scope vs. forgotten bookkeeping):

| Epic | Status shown | Evidence code shipped | Why it may be *intentionally* open |
|---|---|---|---|
| `fix-yy1` (v2.21 TurnResult capture) | open, children open | `fastworkflow/turn.py`, `process_turn` at `workflow_execution_context.py:484`, shipped `afcbe01` | Its own notes say "Tasks left open until approval+commit" — approval happened, closure forgotten? **Ask Dhar.** |
| `fix-qtq` (TurnOutput wire cutover) | open, 9 children | `turns.py` types `WorkFn` as returning `TurnOutput` | Genuinely incomplete: `run_fastapi_mcp/utils.py:427` still serves the old shape; breaking wire change pending. Likely legitimately open. |
| `fix-7kp` (speedict removal) | open | fix-2jo (identical title) CLOSED, shipped v2.21.4 | Duplicate from the flaky-bd era — or an intentional extension (speedict remains in `cache_matching.py`, `workflow.py`, `session_state_store.py`). **Ask Dhar.** |

---

## 5. Version and release conventions

- **Source of truth:** `pyproject.toml:12` (`version = "2.22.2"`). There is **no CHANGELOG file**; the git log *is* the changelog.
- **Bump ships inside the release commit** — every release commit touches `pyproject.toml` (verified on `0e68284`, `b5747df`, `033132f`).
- **Subject format:** `fix: vX.Y.Z — <lowercase summary>` for patches, `feat: vX.Y.Z — <summary>` for minors. Em dash (`—`), not hyphen. Append the bd issue in parens when there is one: `fix: v2.21.5 — auto-evict abandoned Workflow session state (fix-04r)`, `feat: v2.22.0 — turns-based async execution for run_fastapi_mcp (fix-85g Step 1)`.
- **Docs-only commits:** `docs: <summary>`, no version bump. Test-only: `test: <summary>` (`fa97b48`).
- **Patch (x.y.Z)** = bug fix, dependency/CVE bump, perf/staleness fix, no new public capability (v2.21.1–.6, v2.22.1–.2). **Minor (x.Y.0)** = new capability or subsystem, usually with a design doc and epic (v2.21.0 TurnResult, v2.22.0 turns engine). v2.17.x accumulated 35 patches — long patch trains on one minor are normal here. **Major**: v3.0 is reserved for the breaking wire/model collapse per `turn_result_design_final.md` §14–15 (designed, unbuilt).
- **Tags lag and are best-effort:** tags exist through `v2.22.0` but `v2.22.1`/`v2.22.2` are untagged. Never treat tags as the release record; use pyproject + commit subjects.
- **Publishing:** `make publish` (poetry build + publish to PyPI); `make publish-testpypi` for rehearsal. Both depend on `gen-env` (see §7 foot-gun). Per Rule 1, releases happen only on the developer'sexplicit request.
- **Agent trailer:** commits authored with Claude Code end with `Co-Authored-By: Claude ... <noreply@anthropic.com>` (verified on `42e9b3a`).

---

## 6. Test-removal prohibition

From `.cursor/rules/testing_rules.mdc` (`alwaysApply: true`), verbatim:

> - Don't use Mock fixtures. All our tests are integration tests
> - Do not remove pytest tests without explicit user approval

Operational meaning:

- **Never delete, `@pytest.mark.skip`, or comment out a test to make a change pass.** If a test blocks you, that is a finding — file a bd issue and ask. Permanent skips that exist today (11 markers, e.g. `tests/test_workflow_training.py:74,137` "takes too long") were each user-approved decisions.
- **Reality check (doc-rot, documented, not license):** 11 test files already import `unittest.mock`/`MagicMock` (mostly to stub LLM/dspy calls while keeping workflows real). The no-mocks rule is aspirational in places — that is NOT permission to add new mocks; prefer the real test workflows `tests/example_workflow/`, `tests/hello_world_workflow/` (and note CLAUDE.md omits the on-disk `tests/todo_list_workflow/` naming — `tests/example_workflow` IS the todo-list app).
- Changing a test's *assertion* to match new intended behavior is a normal T0/T1 change; removing its *coverage* is what needs approval.
- CI runs only `tests/test_train_modern_stack.py` (single workflow `.github/workflows/train-modern-stack.yml`) — so a removed test will NOT be caught by CI. The full suite is local-only: `source .venv/bin/activate && python -m pytest`. It is **1803 tests and ~49 minutes** as of 2026-08-13 / v3.1.0 (was 495 at v2.22.2). Run it before release commits; Rule 3 applies to how training tests isolate themselves.

---

## 7. The public-repo boundary

The GitHub repo is **public** (Apache 2.0). Treat every commit as a press release.

**NEVER commit or push (without the developer'sexplicit, per-item approval):**

| Item | Where it sits | Why |
|---|---|---|
| tau2 plan | `docs/tau2_retail_reliability_implementation_plan.md` (untracked) | team-private strategy, experiment cards E0–E25 |
| RSI harness report | `docs/rsi_harness_agent_report.md` (untracked) | marked team-private |
| Forge spec | `docs/forge_meta_workflow_spec.md` (untracked) | marked team-private |
| Article PDFs 1–3 + Analysis docx | `docs/Article*.pdf`, `docs/Analysis and Recommendations...docx` (untracked) | pre-publication drafts |
| **The team-private skills only** | `.claude/skills/{fastworkflow-proof-and-analysis-toolkit, fastworkflow-taubench-reference, tau2-reliability-campaign, fastworkflow-research-frontier}/` | These four embed the docs above and are **gitignored by exact path** (`.gitignore:219-237`) as of 2026-08-13. The rest of the library — 14 skill dirs plus `README.md` — was deliberately committed and is PUBLIC. Know which one you are editing: anything you add to a tracked skill is published. |
| Secrets | `passwords/` (gitignored), any real `*.passwords.env` | API keys |

**Sharp edges verified:**

- **The `git add -A` mechanism described here has been fixed, and by something better than tracking.** When this skill was written, `.claude/skills/` and the private `docs/` files were untracked-but-NOT-gitignored, so `git add -A` would have staged them. As of 2026-08-13 every team-private artifact is **gitignored by exact path** — the four skills at `.gitignore:219-237`, and the tau2 plan, RSI harness report and Forge spec verified gitignored under `docs/`. So `-A` can no longer publish them. Stage by explicit path anyway: it costs nothing, it survives someone adding a new private file before adding its ignore line, and it is what the 2026-07-08 incident actually argues for. Re-verify with `git check-ignore -v <path>` rather than assuming either state.
- Untracked ≠ private-safe: the 2026-07-08 incident (Rule 1) was exactly an untracked private doc getting auto-committed.
- `env/.env` IS tracked — deliberately: it holds model-name config only (verified: zero key/token/secret lines). Real keys belong in gitignored `passwords/`. The bundled template `fastworkflow/examples/fastworkflow.passwords.env` is tracked but contains a dead placeholder key.
- **`gen-env.sh` foot-gun:** `make gen-env` (a dependency of `make publish`) merges **every `*.env` in the tree** — including gitignored password files — into root `.env`. Root `/.env` is gitignored, but never copy/rename/commit that merged file, and never widen the `.gitignore` negations (lines 141–143) without review.
- There IS a sanctioned path for *public* skills: `fastworkflow/docs/integrate-chat-agent/SKILL.md` was deliberately committed in `1d3a6aa`. Public skill ⇒ that directory, with approval. Team-private skill ⇒ `.claude/skills/`, uncommitted.
- Public claims (benchmark numbers, DOI — note README and the Article PDFs currently disagree on the paper DOI) are governed by `fastworkflow-docs-and-positioning`; do not add or edit external claims as a side effect of a code change.

**No routing around change control:** nothing in this skill authorizes bypassing a gate because it is inconvenient. If a gate seems wrong, that is a conversation with Dhar, not an exception you grant yourself.

---

## 8. Provenance and maintenance

Facts verified 2026-07-09 against v2.22.2 (commit c33b9a5). This skill also embeds content from uncommitted team-private docs (tau2 plan) and bd memories, both volatile.

Re-verification one-liners (run from repo root):

| Volatile fact | Re-verify with |
|---|---|
| Current version 2.22.2 | `grep -n '^version' pyproject.toml` |
| The three bd memories (Rules 1–2, ELI5) | `bd memories --json` |
| fix-0hb fix commit `fa97b48` | `git log --oneline -1 fa97b48` |
| Parity language lines 43/199/630 | `grep -n "benchmark parity" docs/tau2_retail_reliability_implementation_plan.md` |
| Release-subject convention | `git log --oneline -20 \| grep -E '^\w+ (fix\|feat): v'` |
| Tags lag (last tag v2.22.0) | `git tag -l 'v2.*' \| sort -V \| tail -3` |
| Epic statuses fix-yy1/fix-qtq/fix-7kp still open | `for i in fix-yy1 fix-qtq fix-7kp fix-2jo; do bd show $i --json \| python3 -c "import json,sys; d=json.load(sys.stdin); d=d[0] if isinstance(d,list) else d; print(d['id'],d['status'])"; done` |
| `bd close --reason` persistence | non-destructive: close something you were going to close anyway with `--reason`, then read `close_reason` back out of `.beads/issues.jsonl`. Working as of bd 1.1.2 / 2026-08-13 |
| AGENTS.md still recommends `--reason` (doc-rot) | `grep -n 'close bd-42' AGENTS.md` |
| Test-removal rule text | `sed -n '1,12p' .cursor/rules/testing_rules.mdc` |
| `.claude/skills` not ignored; only rules file tracked | `git check-ignore -v .claude/skills/x; git ls-files .claude` |
| Team-private docs still untracked | `git status --porcelain docs/ \| grep '^??'` |
| env/.env tracked and secret-free | `git ls-files env/.env; grep -c -iE 'key\|token\|secret' env/.env` (expect 0) |
| gen-env merges all *.env | `grep -n 'find . -name "\*.env"' gen-env.sh` |
| fix-vof artifact sizes/line counts | `wc -c docs/turn_result_design.md; wc -l docs/turn_result_design_review.md docs/turn_result_design_final.md` |
| R1 decision-record commit format | `git show 42e9b3a --stat --format='%s%n%b'` |
| CI runs one test file only | `grep -n 'pytest' .github/workflows/*.yml` |
| Mock-usage doc-rot count (11 files) | `grep -rlE 'unittest.mock\|MagicMock' tests --include='*.py' \| wc -l` |

Gate rationale and cost/benefit case study: [references/fix-vof-case-study.md](references/fix-vof-case-study.md).
