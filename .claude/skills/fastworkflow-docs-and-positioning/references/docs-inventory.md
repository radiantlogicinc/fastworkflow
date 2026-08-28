# Full docs inventory with per-file notes

> TEAM-PRIVATE: embeds content from uncommitted internal docs. Do not commit or publish this skill without the developer'sexplicit approval.

Facts verified 2026-07-09 against v2.22.2 (commit c33b9a5). Re-verify the file list with
`git ls-files docs/ '*.md' && git status --porcelain docs/`.

## Root-level docs (git-tracked)

| File | Purpose | Authority / status |
|---|---|---|
| `README.md` | Public positioning, benchmark charts, quickstart, FastAPI/MCP usage, env contract, dependency posture | Doc of record for what the project publicly IS. 603 lines. Benchmark provenance gap — see external-claims-register.md |
| `CLAUDE.md` | Claude Code agent instructions (project overview, testing philosophy, CLI, three-phase architecture, Topology B, env vars, beads pointer) | Doc of record for agent discipline, ALWAYS loaded. Three verified inaccuracies (SKILL.md §3). Deliberately slim since c33b9a5 ("trim derivable content") — resist re-fattening it; path-scoped material belongs in `.claude/rules/` |
| `AGENTS.md` (90 lines) | venv-activation rule + full beads (`bd`) command reference: mandatory for ALL tracking, `--json` always, discovered-from links, priorities 0–4 | Doc of record for issue tracking |
| `SECURITY_VULNERABILITY_REPORT.md` | 2026-01-28 CVE sweep: 22 vulns across 11 packages, 21 fixed, each mapped to a bead ID. Open: ecdsa CVE-2024-23342 (no patch exists). Caution: its bead reference `fastworkflow-d8f` (line 86) is absent from `.beads/issues.jsonl` as of 2026-07-09 (`bd search ecdsa` empty) — the report's bead mappings cannot all be trusted | Historical snapshot, not living |
| `WEC_learning_checklist.md` | Topology A→B migration mastery checklist (`[ ]`/`[~]`/`[x]`, checked only after explain-back) | Living doc; exemplar of the checklist house style |
| `fastworkflow-article-1.md` | Tutorial 1: intro, hello_world, hand-built messaging_app_1 | Published tutorial series |
| `fastworkflow-article-2.md` | Tutorial 2: functions → classes/state (`workflow.root_command_context`, `command_context_for_response_generation`) | " |
| `fastworkflow-article-3.md` | Tutorial 3: inheritance via `context_inheritance_model.json` (`base` key, `*` = available anywhere) | " |
| `fastworkflow-article-4.md` | Tutorial 4: parent-child hierarchies via `context_hierarchy_model.json`, `_<ClassName>.py` `Context.get_parent`, `db_lookup`, `startup_action.json` | " — note line 637 link text/href org mismatch |

## `docs/` git-tracked design docs

Grouped by saga. Remember: **a spec here is not proof the feature exists.**

### TurnResult saga (the house-style exemplar — see design-doc-lifecycle.md)

| File | Lines | Role |
|---|---|---|
| `turn_result_design.md` | 1424 | Origin bug narrative, root cause, type algebra, 22-entry decision log (§12), Amendments A1–A47 (from line 781). §13 contains the rewritten "none remaining" overclaim admission |
| `turn_result_design_review.md` | 1577 | 48 adversarial findings R1–R48 with severity index (lines 16–46) |
| `turn_result_architecture_review.md` | 297 | Second-order review, ten parallel per-concern agents, findings X1–X12 |
| `turn_result_design_final.md` | 620 | AUTHORITATIVE consolidated spec with supremacy clause (lines 1–8) and traceability table (§18). §14–15 = v3.0 roadmap, UNIMPLEMENTED as of v2.22.2 (no stores/ package, no ConversationTurnStore, no admin CLI) |
| `turn_result_design_feedback.md` | 354 | Human teaching/pushback session — caught that the shipped v2.21 slice did not fix the user-visible bug |
| `turn_result_learning_checklist.md` | 51 | Staged mastery tracking |

### Execution-core / FastAPI / MCP

| File | Role / status |
|---|---|
| `fastworkflow_turns_async_execution_design.md` | "Final, ready to implement" design for the v2.22.0 turns engine (fix-85g). Step 1 shipped (cf3eeae); Step 2 (TTL eviction, `/turns` polling endpoint) designed but NOT wired — the `/initialize` docstring already tells clients to poll a `GET /turns` route that does not exist |
| `understanding_execution_core_refactor.md` | Learning checklist for the transport-free execution core (WEC owns its own cme_workflow; ContextVar active-workflow stack; CommandCancelledError subclasses BaseException) |
| `workflow_execution_context_migration.md` | Embedder migration guide: use WEC instead of ChatSession when you own HTTP/session |
| `learning_trajectory_serialization.md` | Learning checklist: trajectory serialization + Topology B |
| `fastworkflow_fastapi_spec.md` / `fastworkflow_fastapi_architecture.md` | FastAPI service spec + derived architecture doc (spec → architecture pairing is a house pattern) |
| `fastworkflow_mcp_server_spec.md` / `fastworkflow_mcp_server_spec_summary.md` | HTTP-only MCP server spec + key-decisions summary. Check against `run_fastapi_mcp/mcp_specific.py` before trusting details (prompt registration is NOT supported in fastapi-mcp 0.4.0 despite older spec language) |
| `invoke_agent_stream_implementation.md` | Implementation summary for `/invoke_agent_stream` (SSE/NDJSON) |
| `agent_mode_live_command_traces.md` | Design spec for live command traces in agent mode |
| `agentic_functionality_integration.md` | Spec for integrating agentic functionality into core (historical) |

### Build/refine pipeline

| File | Role / status |
|---|---|
| `build_postprocessing_using_dspy.md` | Original spec for GenAI build post-processing (born inline in v2.11.0, split into `refine` in v2.13.0) |
| `genai_postprocessor_readme.md` | The refine tool's LibCST additive/idempotent editing contract ("never overwrites existing field metadata") |
| `extending_existing_workflows.md` | How custom workflows reuse/override base template workflows |
| `DSPY_CACHE_GUIDE.md` | How to clear DSPy's disk/memory LLM cache when refine/agent outputs seem frozen |

### Benchmarks / ghosts

| File | Role / status |
|---|---|
| `implementing_tau_bench_retail.md` (Sep 2025) | Original τ-bench retail integration spec. The adapter lives in the tau-bench FORK (`tau_bench/agents/fastworkflow_adapter.py` etc. per its lines 101–136), not this repo; only `fastworkflow/examples/retail_workflow/` is in-tree. OBSOLETE adapter design post-Topology-B, and uses a nonstandard Pass^k definition (success within k retries). Historical reference only |
| `probabilistic_response_generation.md` | GHOST — spec on main, implementation never merged (`grep -rl probabilistic fastworkflow/ --include='*.py'` empty) |
| `chats/` | Empty directory, no git history — repo-hygiene backlog |

## `docs/` UNTRACKED team-private files (never commit without Dhar)

| File | What it is |
|---|---|
| `tau2_retail_reliability_implementation_plan.md` | 949-line Draft v2.0 (2026-07-09): E0–E25 experiment cards, Option N v2 "Governed Determinism", phasing, dependency spine, Appendix E reference errata. §2.5 governs on any conflict with card text |
| `rsi_harness_agent_report.md` | E21 design: bounded RSI loop for reliability (frontier proposer / gpt-oss-20b runtime, typed editable surface, 5-stage eval cascade, pass^k promotion objective). Marked team-private at line 4 and line 217 |
| `forge_meta_workflow_spec.md` | 714-line Draft v1.0: Forge self-hosting meta-workflow (gates, invariants G0/FP-1/SEC-1/PROV-1, ForgeSandbox §16). Marked team-private at line 4 and line 714 |
| `Article 1 - The Setup.pdf` (8pp), `Article 2 - The Failure Taxonomy.pdf` (12pp), `Article 3 - Mitigations and a Path Forward.pdf` (7pp) | The RLM failure series (Rakshit Pandhi et al., Jul 7 2026). Raw traces are public at github.com/Programiz-007/fastworkflow_RLM_Traces, but the PDFs themselves are unpublished/untracked |
| `Analysis and Recommendations for Improving fastWorkflow.docx` | Intern research ideas; absorbed into the plan as E17, E3 tiers, E5/E10, E14 |

Disposition: undecided by owner. The tau2 plan supersedes the PDFs' analysis where they
conflict; E19 (publish fork + artifacts) is the planned path to making the RLM work
publicly citable.

## Package-shipped docs (inside the wheel)

`fastworkflow/docs/` was dissolved 2026-08-25: the skill moved into the new public
coding-skills library `fastworkflow/skills_for_coding_fastworkflows/`, the PRD to
repo-level `docs/`.

| File | Role |
|---|---|
| `fastworkflow/skills_for_coding_fastworkflows/integrate-chat-agent/` (SKILL.md + reference.md) | Coding-agent skill for downstream app developers (added 1d3a6aa, 2026-06-16; moved + revised 2026-08-25). README positions it as the recommended integration path. Mandates hand-written command files over `fastworkflow build`, and beads tracking for downstream devs. Known gap: does not document `context_hierarchy_model.json` or `_<ContextName>.py` context classes |
| `docs/context_modules_prd.txt` (repo docs/, no longer wheel-shipped) | PRD for per-context callback modules (`_<ContextName>.py` with `Context.get_parent`) — a live feature whose only in-tree example is `tests/todo_list_workflow/_commands/TodoItem/_TodoItem.py`, and which CLAUDE.md and command-authoring.md do not mention |

## Agent-rules surfaces

| File | Status |
|---|---|
| `.claude/rules/command-authoring.md` | LIVE. Path-scoped frontmatter: `**/_commands/**`, `**/context_inheritance_model.json`. Single-file command pattern, two-key context model entries, NOT_FOUND ask-don't-guess |
| `.cursor/rules/testing_rules.mdc` | LIVE, alwaysApply. No mocks, real test workflows, never delete tests without approval |
| `.cursor/rules/cursor_rules.mdc`, `self_improve.mdc` | Meta-rules about writing Cursor rules; low stakes |
| `.cursor/rules/taskmaster.mdc`, `dev_workflow.mdc` | STALE but alwaysApply:true — still prescribe Task Master to Cursor sessions, contradicting AGENTS.md. Cleanup pending owner decision |
| `.cursor/skills/beads-workflow/SKILL.md` | bd usage rules for Cursor |
| `.cursor/skills/landing-the-plane/SKILL.md` | Session-close protocol whose "Work is NOT complete until `git push` succeeds" (lines 8, 29) is OVERRIDDEN by the 2026-07-08 never-commit rule. Do not follow its push steps |

## Stale editor-config sprawl (cleanup pending owner decision — read-only until Dhar rules)

| Path | Tracked? | Evidence of death |
|---|---|---|
| `.taskmaster/` | Only `templates/example_prd.txt` | `tasks/tasks.json` mtime 2025-06-23, `state.json` 2025-06-16 — frozen >12 months while beads is used daily |
| `.windsurfrules` | No | Same Task Master dev_workflow text |
| `.roo/rules/` (13 docs) + `.roomodes` | No | Dated ~2025-06; `context-model-design.md` references deleted `fastworkflow/context_model.py` (current module: `command_context_model.py`) |

Other hygiene backlog (do not delete unilaterally): root `redoc.html` (stale, Oct 2025),
`fastworkflow/run_fastapi_mcp/redoc_2_standalone_html.py` (broken for the current layout),
root `examples/` gitignored fetch copies, `docs/chats/` empty dir,
`___user_conversations/` legacy runtime residue. (The empty tracked root
`passwords.env` was deleted 2026-08-26 — the "keeps gen-env merging working"
rationale recorded here was wrong: gen-env.sh merges by glob.)
