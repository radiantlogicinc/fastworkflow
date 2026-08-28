---
name: fastworkflow-docs-and-positioning
description: Load when writing, editing, or citing any fastWorkflow documentation or public claim — README changes, design docs, articles, benchmark numbers, DOIs, "which doc is authoritative?", "can we say/publish X?", "the README/CLAUDE.md says...", starting a new design doc or design review, or handling the team-private docs in docs/. Triggers include "article" (there are TWO colliding series), "DOI", "Tau Bench results", "positioning", "doc rot", "commit message style". Do NOT load for running the code (use fastworkflow-run-and-operate), the statistical evidence bar for experiments (fastworkflow-research-methodology), tau-bench mechanics (fastworkflow-taubench-reference), or the change-gating process itself (fastworkflow-change-control).
---

# fastWorkflow Docs and Positioning

> TEAM-PRIVATE: embeds content from uncommitted internal docs. Do not commit or publish this skill without the developer'sexplicit approval.

This skill is the map of every documentation surface in the repo, its authority level,
its known rot, the house design-doc process, and the rules for what may be claimed
publicly. Facts verified 2026-07-09 against v2.22.2 (commit c33b9a5).

## When to use / when NOT to use

**Use when you are:**
- Editing README, CLAUDE.md, AGENTS.md, anything under `docs/`, or the article series
- Deciding whether a doc is authoritative, stale, or a ghost (spec for unbuilt feature)
- Making, checking, or citing any external claim (benchmark numbers, paper citation, DOI)
- Starting a design doc, design review, or decision record (use the house template here)
- Touching or referencing the untracked team-private docs in `docs/`
- Writing commit messages for docs/design work

**Do NOT use for — go to the sibling skill instead:**

| Need | Sibling skill |
|---|---|
| Change gating, the non-negotiable rules and their incidents | `fastworkflow-change-control` |
| Evidence bar for a new experimental result | `fastworkflow-research-methodology` |
| tau-bench / tau2-bench harness mechanics, pass@1 vs pass^k definitions | `fastworkflow-taubench-reference` |
| Running the CLI/server, artifact locations | `fastworkflow-run-and-operate` |
| The tau2 E0–E25 campaign execution | `tau2-reliability-campaign` |
| Env var / flag catalog | `fastworkflow-config-and-flags` |
| What counts as test evidence | `fastworkflow-validation-and-qa` |

---

## 1. The docs map, by authority level

"Doc of record" = the file you are allowed to cite as the current truth for its domain.
Everything else is rationale archive, marketing, or rot. Full per-file inventory with
notes: [references/docs-inventory.md](references/docs-inventory.md).

| Surface | Authority | Audience | Status |
|---|---|---|---|
| `README.md` (603 lines) | Doc of record for public positioning + quickstart | Public | Live; benchmark provenance gap (§4) |
| `CLAUDE.md` | Doc of record for agent discipline | Agents | Live; 3 known inaccuracies (§3) |
| `AGENTS.md` | Doc of record for beads (`bd`) issue tracking + venv rule | Agents | Live |
| `.claude/rules/command-authoring.md` | Doc of record for writing `_commands/*.py` | Agents | Live; path-scoped to `**/_commands/**` and `**/context_inheritance_model.json` — loads only when touching those paths (moved out of CLAUDE.md in c33b9a5) |
| `.cursor/rules/testing_rules.mdc` | Doc of record for testing rules | Agents (Cursor) | Live, `alwaysApply: true` |
| `fastworkflow-article-1..4.md` (repo root) | Published TUTORIAL series | Public | Live; see name-collision warning (§2) |
| `docs/*.md` (git-tracked) | Design docs; per-doc supremacy clauses | Team | Mixed — some are ghosts (§3) |
| `docs/turn_result_design_final.md` | Authoritative spec for TurnResult; explicit "this document wins" clause (lines 1–8) | Team | Live; v3.0 half unimplemented |
| `fastworkflow/skills_for_coding_fastworkflows/` (integrate-chat-agent + 9 more skills; moved 2026-08-25 from `fastworkflow/docs/integrate-chat-agent/`) | Wheel-shipped coding-agent skill library; README calls integrate-chat-agent "the fastest path for a non-trivial app" | Public + downstream devs | Live, ships in the wheel (untracked in git as of the move — commit pending) |
| `docs/` untracked files | TEAM-PRIVATE strategy docs (§5) | Team only | NEVER commit without the developer'sexplicit approval |
| `SECURITY_VULNERABILITY_REPORT.md` | Point-in-time CVE remediation report (2026-01-28) | Team | Historical; one open item (ecdsa CVE-2024-23342, no patch exists). NB the report's own bead reference `fastworkflow-d8f` (line 86) is absent from `.beads/issues.jsonl` as of 2026-07-09 — the report has its own rot |
| `WEC_learning_checklist.md`, `docs/turn_result_learning_checklist.md` | Mastery checklists (house style, §6) | Team | Living docs |
| `.cursor/rules/{taskmaster,dev_workflow}.mdc`, `.windsurfrules`, `.roo/`, `.roomodes`, `.taskmaster/` | NONE — stale editor-config sprawl | Nobody | Dead since beads adoption; cleanup pending owner decision (§3) |

**Rule of thumb (verified repeatedly in this repo): a spec doc in `docs/` is NOT proof
the feature exists.** `docs/probabilistic_response_generation.md` has zero implementation
on main (`grep -rl probabilistic fastworkflow/ --include='*.py'` is empty). Check the code.

## 2. LOUD WARNING: two "Article" series with colliding names

There are two completely different article series. Confusing them has real consequences —
one is public, one is team-private.

| | Root tutorial series | docs/ failure series |
|---|---|---|
| Files | `fastworkflow-article-1.md` … `-4.md` (repo root, git-tracked) | `docs/Article 1 - The Setup.pdf`, `Article 2 - The Failure Taxonomy.pdf`, `Article 3 - Mitigations and a Path Forward.pdf` (UNTRACKED) |
| What | 4-part published tutorial: hello_world → stateful classes → inheritance (`context_inheritance_model.json`) → parent-child hierarchies (`context_hierarchy_model.json`) | 3-part study of an RLM memory layer on τ²-bench retail: 15 hand-picked hard tasks, 2/15 → 5/15 after mitigations, failure taxonomy |
| Authors | Dhar Rawal / core team | Rakshit Pandhi (principal), Sanchit Satija, Sharvil Oza; mentored by Dhar; funded by Radiant Logic |
| Visibility | PUBLIC | TEAM-PRIVATE until Dhar decides (dated Jul 7 2026) |
| Citation hygiene | article-4 link text says `github.com/fastworkflow/fastworkflow` but href is `radiantlogicinc/fastworkflow` (line 637) — org migration status unknown | DOI + arXiv citation errors, see §4 |

When someone says "Article 2", ALWAYS ask or determine which series. "The Failure
Taxonomy" = the private PDF; "From Functions to Classes" = the public tutorial.

Glossary for the failure series: **RLM** = the memory-retrieval module that answers
`recall_from_memory(query)` by writing and executing Python code against archived tool
outputs, terminating with `SUBMIT(answer)` (derived from "Recursive Language Models",
arXiv 2510.04871). **τ²-bench** = the dual-control conversational-agent benchmark
(arXiv 2506.07982). The RLM code lives in a fork, NOT on main (zero grep hits for
`recall_from_memory`); publishing it is work item E19 in the tau2 plan.

## 3. Doc-rot inventory (verified 2026-07-09)

These are documented inaccuracies. When your reading of the code contradicts a doc
below, the code wins. Do not "fix" tracked docs without going through
`fastworkflow-change-control` — but do not propagate the errors either.

| Location | What is wrong | Evidence |
|---|---|---|
| `CLAUDE.md:7,30` | Says intent models are trained "via scikit-learn" / pipeline built on "DSPy, scikit-learn, and Pydantic" | Training is torch/transformers: `fastworkflow/model_pipeline_training.py:3-17` imports `AutoModelForSequenceClassification`, `torch`, `AdamW`; sklearn supplies only PCA, metrics, split, LabelEncoder |
| `CLAUDE.md:33` | "sklearn classifier identifies the target command" | Same — it is a fine-tuned BERT-family classifier (see `fastworkflow-nlu-pipeline-reference`) |
| `CLAUDE.md:13` | Names only `tests/example_workflow/` and `tests/hello_world_workflow/` | Third real test workflow `tests/todo_list_workflow/` is used by 10+ test files (`grep -rl todo_list_workflow tests/`) |
| `fastworkflow/run_fastapi_mcp/README.md` | Launch command `uvicorn services.run_fastapi.main:app` (lines 31, 63, 75) — that package was deleted; request examples use `channel_query` (lines 314, 337, 344) but the model field is `user_query`; documents 504-on-timeout (line 360) but v2.22.0 switched to 202-deferral | File itself + `git log --oneline -- services/` + `fastworkflow/run_fastapi_mcp/turns.py` |
| `fastworkflow/run_fastapi_mcp/mcp_specific.py:60-63` vs older docs | MCP prompt registration NOT supported (fastapi-mcp 0.4.0 limitation, commented out) | The code comment says so explicitly |
| `redoc.html` (repo root) | Stale generated API doc, dated 2025-10-11 | `ls -la redoc.html` |
| `pyproject.toml:41` | Comment references `model_pipeline_training._load_tokenizer`, a function that no longer exists | `grep -rn _load_tokenizer fastworkflow/` is empty |
| `docs/implementing_tau_bench_retail.md` (Sep 2025) | Original τ-bench integration spec; its queue-based monkey-patch adapter is obsolete post-Topology-B (tau2 plan Phase 0 mandates redesign). Also uses a NONSTANDARD "Pass^1…Pass^4 = success within k turns/retries" — not the standard pass^k (all k independent trials pass). Never quote its metric definitions; use `fastworkflow-taubench-reference` |
| `docs/probabilistic_response_generation.md` | Spec ghost — feature never merged | grep above |
| `fastworkflow-article-4.md:637` | Link text/href org mismatch (see §2) | The line itself |

**Editor-config sprawl (dead, cleanup-pending-owner-decision — do NOT delete without
Dhar):** `.taskmaster/` (state frozen 2025-06-16/23), `.cursor/rules/taskmaster.mdc` and
`dev_workflow.mdc` (both `alwaysApply: true`, still prescribing Task Master to any Cursor
session — directly contradicting AGENTS.md's "bd for ALL issue tracking"), `.windsurfrules`
and `.roo/rules/` (13 stale docs; `context-model-design.md` references the deleted
`fastworkflow/context_model.py`), `.roomodes`. Only the `.cursor/rules/*.mdc` files,
`.cursor/skills/`, and `.taskmaster/templates/example_prd.txt` are git-tracked.
Special hazard: `.cursor/skills/landing-the-plane/SKILL.md` says "Work is NOT complete
until `git push` succeeds" — this is OVERRIDDEN by the never-commit rule (§5). Beads
(AGENTS.md) is the only live tracking system.

## 4. External-claim discipline

**Jargon:** *pass@1* = fraction of tasks solved in one attempt. *pass^k* = probability
all k independent runs of the same task succeed (a reliability metric — see
`fastworkflow-taubench-reference`). *DOI* = permanent Digital Object Identifier for a
publication.

### What may be claimed publicly today

Exactly one class of external claim is currently backed: **the published CAIS '26 paper,
with its citation** (README.md:101): Satija, Bhatt, Jani, Rawal 2026, *fastWorkflow:
Closing the Performance Gap Between Small and Frontier Language Models for Conversational
Agents*, CAIS '26. The README's positioning ("small models punch above their weight AND
frontier models get more reliable", README.md:6-12; the four structural failure modes
table, README.md:69-74) is marketing framing of that paper.

### Known citation defects — do not propagate

| Defect | Detail | Status |
|---|---|---|
| DOI discrepancy | README.md:101 and Article 1 PDF cite `10.1145/3786335.3813158`; Article 2 and Article 3 PDFs cite `10.1145/3738331.3738402`. Verified by extracting text from all three PDFs. One is wrong. | OPEN — tau2 plan Appendix E (line 933) says "verify the single correct ACM DOI". Until resolved, cite the README form and flag it |
| arXiv RLM citation | The Article PDFs cite arXiv 2512.24601 for Recursive Language Models; correct ID is 2510.04871 | Acknowledged erroneous, "being fixed" (tau2 plan line 931) |

### The benchmark-provenance gap (flagged, unresolved)

The README's headline charts (`fastWorkflow - Tau Bench Retail.jpg` at README.md:88,
`fastWorkflow - TauBench Airline.jpg` at README.md:92; top numbers Retail 86.95%
Mistral Small 24B, Airline 96.00% GPT-OSS 20B, labeled Pass@1 in the images) have **no
in-repo provenance**: no run counts, no harness config, no task subset, no record of
whether leaderboard baselines were re-run or quoted. The paper is the citation; the
repo documents nothing. `grep -in -e 'pass@1' -e trial README.md` returns nothing.

**Standard for every FUTURE public number (non-negotiable):** before any new benchmark
claim ships in README or an article, the repo must contain, committed: (1) run count k
and per-run results, (2) exact harness/fork commit and config, (3) task subset (full
split vs enriched subset — the failure-series 15-task set is adversarially enriched,
never present it as representative), (4) any nonstandard trade **disclosed, never
silent** (e.g., pinning user-simulator temperature), (5) statistical treatment per
`fastworkflow-research-methodology` (confidence intervals, paired tests). Benchmark
parity is sacred: NEVER modify tau-bench tools/tasks for benchmark runs.

### What is NOT yet publicly claimable

- Anything from the failure-series PDFs or the tau2 plan (2/15 → 5/15, the failure
  taxonomy, Option N v2, pass^3 targets) — team-private until Dhar approves and E19
  (publish the fork + artifacts) lands.
- Any pass^k / reliability number — none has been measured under the E0 harness (E0 is
  unbuilt as of 2026-07-09; no `eval/` dir exists).
- "1-shot adaptation from intent-detection mistakes" (README.md:373) is implemented
  (`fastworkflow/cache_matching.py:53 store_utterance_cache`) but its test coverage and
  cross-session persistence claims are unaudited — treat elaborations beyond the README
  sentence as unverified.

Full claims register with per-claim status: [references/external-claims-register.md](references/external-claims-register.md).

## 5. Team-private docs and the confidentiality boundary

The repo is PUBLIC (github.com/radiantlogicinc/fastworkflow). These files sit untracked
in the working tree and must stay that way until Dhar explicitly says otherwise:

| File | What it is | Marker |
|---|---|---|
| `docs/tau2_retail_reliability_implementation_plan.md` (949 lines, Draft v2.0, 2026-07-09) | THE program plan: experiment cards E0–E25, Option N v2 "Governed Determinism"; §2.5 governs on conflicts | References team-private docs; treat as team-private |
| `docs/rsi_harness_agent_report.md` | E21 RSI-for-reliability design | Line 4: "Team-private report — do not commit or publish without explicit approval" |
| `docs/forge_meta_workflow_spec.md` (714 lines, Draft v1.0) | Forge self-hosting meta-workflow spec | Line 4: "team-private, not for publication" |
| `docs/Article 1-3 *.pdf` | The failure series (§2) | Untracked; unpublished |
| `docs/Analysis and Recommendations for Improving fastWorkflow.docx` | Intern research ideas absorbed into the plan (→ E17, E3, E5/E10, E14) | Untracked |

**THE RULE (bd memory `never-git-commit-or-push-anything-without-dhar`, established
2026-07-08):** "NEVER git commit or push anything without the developer'sexplicit request in that
turn — the session-close protocol does NOT override this. Documents, plans, and analyses
may be private/team-only even when written into the repo tree; the fastworkflow repo is
PUBLIC. Established 2026-07-08 after a private planning doc was auto-committed and pushed,
requiring a history rewrite."

Practical consequences: `git add`/`commit`/`push` require the developer'sexplicit same-turn
request — always, for everything, including "just docs". A blanket "land the plane" or
"wrap up" instruction is NOT that request. When writing new strategy docs, put the
team-private status line at the top on day one (copy the RSI report's line 4). This
skill library itself is team-private and uncommitted.

## 6. House design-doc lifecycle (the TurnResult template)

For any risky or large change, the house process — proven on the TurnResult saga
(docs/turn_result_design*.md, ~4,300 lines total, epic `fix-vof`, 48 findings, all
resolved) — is:

1. **Root-cause narrative + design doc** in `docs/` — symptom transcript first, then
   root cause with file:line evidence, then design with a numbered **decision log** table
   (`turn_result_design.md` §12).
2. **Adversarial review** against the ACTUAL working tree at a pinned commit — numbered
   findings R1..Rn with a severity index (`turn_result_design_review.md`). The original
   design declared "Open questions: none remaining"; the review found four internal
   contradictions — the overclaim was publicly rewritten as an admission (design.md §13).
   A design is not done until adversarially reviewed against the codebase.
3. **One beads epic, one child per finding, review-only** (no implementation), worked
   one finding at a time with the human.
4. **One docs commit per resolution**: subject `docs: resolve R# - <decision>`, body =
   decision summary + the traceability triple (design-doc Amendment A#; review doc R#
   marked resolved; beads child closed with epic counter). Exemplar: commit 42e9b3a.
5. **Second-order architecture review** of the finalized design by parallel per-concern
   agents — findings X1..Xn (`turn_result_architecture_review.md`, ten concerns).
6. **Consolidated final spec** with an explicit supremacy clause ("Where this document
   conflicts with any of those, this document wins; they remain the rationale archive")
   and a traceability table (`turn_result_design_final.md`).
7. **Learning checklist + human teaching/pushback session** — items checked `[x]` only
   after explained-back verification; the TurnResult session caught that the shipped
   v2.21 slice did NOT actually fix the user-visible bug (`turn_result_design_feedback.md`).

**Never edit design history — append Amendments** with traceability tags (`[A7]`,
`[X3]`). Every decision must be traceable four ways: R# → beads child → resolution
commit → A# → final-spec section.

Copy-pasteable skeletons for all seven stages (doc headers, finding format, amendment
format, commit template, checklist legend):
[references/design-doc-lifecycle.md](references/design-doc-lifecycle.md).

## 7. Writing style guide (observed, verified in git log and files)

**Commit messages:**
- Type prefixes dominate: `docs:`, `fix:`, `feat:`, `test:` (44 of the last 60 commits
  are `docs:` from the fix-vof epic).
- Release commits: `fix: vX.Y.Z — <root cause in plain words>` (e.g. b5747df
  `fix: v2.22.1 — fingerprint-based invalidation of stale command caches`); bump
  `pyproject.toml` version in the same commit; cross-reference the bead ID; include an
  accepted-limitations note for partial fixes (cf3eeae "(fix-85g Step 1)").
- Design-resolution commits: the §6 stage-4 template.

**README voice:** lead with a concrete failure narrative before features; failure-mode
tables (README.md:69-74); GitHub callouts (`> [!NOTE]` etc. — 6 uses); one mermaid
diagram; bold one-line claims.

**Tutorial voice** (fastworkflow-article-1..4.md): before/after code diffs with "Key
differences"/"Key points" bullets; transcript walkthroughs; "Next up" links; closing
"Key Takeaways".

**Internal-doc voice:** symptom → root cause → fix narration with file:line evidence;
mastery checklists with legend `[ ]` not yet · `[~]` in progress · `[x]` mastered
(explained back); "Living doc" banner; supremacy clauses; amendments over edits.

**Strategy-doc voice** (tau2 plan): "Written for a college-freshman-level intern",
glossary-first, a "How to Read This Document" section, explicit governance clauses
("where §2.5 and the original card text disagree, §2.5 governs").

## Provenance and maintenance

Facts verified 2026-07-09 against v2.22.2 (commit c33b9a5). Re-verify volatile facts:

| Fact | Re-verification command |
|---|---|
| CLAUDE.md scikit-learn rot (lines 7, 30, 33) | `grep -n -e sklearn -e scikit CLAUDE.md; sed -n 3,7p fastworkflow/model_pipeline_training.py` |
| CLAUDE.md test-workflow omission (line 13) | `grep -n workflow CLAUDE.md; ls -d tests/*workflow*` |
| Which docs/ files are team-private (untracked) | `git status --porcelain docs/` |
| Team-private markers still present | `grep -n -i "team-private" docs/rsi_harness_agent_report.md docs/forge_meta_workflow_spec.md` |
| README claim/citation line numbers (88, 92, 99, 101, 179, 373, 407) | `grep -n -e doi.org -e .jpg -e integrate-chat-agent -e "Adaptive intent" README.md` |
| DOI discrepancy across PDFs | extract text: `python3 -c "import re,zlib,sys;d=open(sys.argv[1],'rb').read();print(sorted(set(re.findall(rb'10\\.\\d{4,5}/[0-9.]+',d))))" "docs/Article 2 - The Failure Taxonomy.pdf"` |
| run_fastapi_mcp README rot | `grep -n -e services.run_fastapi -e channel_query -e 504 fastworkflow/run_fastapi_mcp/README.md` |
| probabilistic_response_generation still a ghost | `grep -rl probabilistic fastworkflow/ --include='*.py'` (expect empty) |
| pyproject stale comment | `grep -rn _load_tokenizer fastworkflow/ pyproject.toml` |
| Task Master still dead / beads live | `stat -c '%y' .taskmaster/tasks/tasks.json; grep -n "bd (beads)" AGENTS.md` |
| Never-commit rule text | `bd memories never-git --json` |
| beads close --reason hazard (retested working 2026-08-13, bd 1.1.2) | `bd memories flakiness --json`; change-control Rule 2 |
| Supremacy clause intact | `sed -n 1,8p docs/turn_result_design_final.md` |
| Resolution-commit exemplar | `git show --stat 42e9b3a` |
| ecdsa item still open / bead still missing | `grep -n ecdsa SECURITY_VULNERABILITY_REPORT.md; grep -c fastworkflow-d8f .beads/issues.jsonl; bd search ecdsa --json` |
| E0/eval harness still unbuilt | `ls eval/ docs/experiments/ 2>&1` (expect not found) |
| Current version | `grep -n '^version' pyproject.toml` |

Unverified/open items carried in this skill: the correct ACM DOI (open); whether the
Article PDFs are pre-publication drafts (unknown — no placeholder text found in
extracted PDF text, so the earlier "placeholder links" report is unconfirmed); the
fastworkflow.org vs radiantlogicinc org migration (unknown); disposition of the
editor-config sprawl (owner decision pending).
