# fastWorkflow Skill Library

> **TEAM-PRIVATE.** Several skills embed content from uncommitted internal docs
> (tau2 reliability plan, RSI harness report, Forge spec, the Article PDFs).
> Do **not** commit or publish this directory without the developer'sexplicit approval.
> (The never-commit-unasked rule applies to the whole repo regardless — see
> `fastworkflow-change-control`.)

Authored 2026-07-09 against v2.22.2 (commit c33b9a5) as a continuity library: it exists so
that a mid-level engineer or a Sonnet-class agent with zero context can debug, extend,
validate, and advance this project at the standard it was held to. Every skill was
ground-truth-verified against the repo, then adversarially reviewed (factual / doctrine /
usability) and fixed. Each skill ends with a **Provenance and maintenance** section whose
one-line commands re-verify anything that may have drifted — run them before trusting
volatile facts (line numbers, issue statuses, flags).

## Inventory

### Core — how to work here

| Skill | One line |
|---|---|
| `fastworkflow-change-control` | The four incident-backed non-negotiables, the T0/T1/T2 change gates (fix-vof adversarial review as the T2 model), beads discipline + flakiness protocol, release conventions, the public-repo boundary. Load BEFORE any commit/close/version/test-removal decision. |
| `fastworkflow-build-and-env` | Recreate the dev environment from scratch: Python/poetry/extras, secret provisioning (env/.env + passwords/.env + gen-env.sh foot-gun), the golden hello_world trained artifact, known fresh-clone traps. |
| `fastworkflow-run-and-operate` | CLI anatomy for all six subcommands, the bundled-examples learning progression, artifact conventions (`___command_info` etc.), the FastAPI+MCP server surface and its trusted-network-only security posture. |
| `fastworkflow-config-and-flags` | Catalog of every env var and CLI flag: default, consumer, production/experimental/dead status, sharp edges (get_env_var precedence, subcommands that ignore env files), how to add a config axis. |
| `fastworkflow-validation-and-qa` | What counts as evidence: the real state of the 1803-test suite (env gating, skips, no-ops), CI truth, golden test workflows, how to add tests without violating the no-mock/no-removal rules. |
| `fastworkflow-diagnostics-and-tooling` | Measure instead of eyeball: working scripts (`inspect_command_info.py`, `check_cache_freshness.py`, `trace_turn.py`, `collect_model_metrics.py`) with interpretation guides, plus existing observability. |
| `fastworkflow-docs-and-positioning` | Docs-of-record map with authority levels, the TWO colliding article series, house design-doc lifecycle template, external-claim discipline (benchmark provenance, DOI discrepancy, confidentiality boundary). |

### Core — how the system works

| Skill | One line |
|---|---|
| `fastworkflow-architecture-contract` | Load-bearing design decisions and WHY: WEC vs ChatSession, CME/wildcard pipeline, BaseException suspension, the invariants (single-writer, iteration_counter=-1, finalize chokepoint), known-weak points, the v3.0 designed-but-unbuilt boundary. |
| `fastworkflow-nlu-pipeline-reference` | Domain pack for the NLU stack as implemented here: two-tier BERT intent models, thresholds and their (undocumented) provenance, synthetic utterance generation, DSPy parameter extraction, runtime matching layers, litellm routing. |
| `fastworkflow-taubench-reference` | Domain pack for tau-bench/tau2-bench as they apply here: pass@1 vs pass^k, the user-simulator ceiling (with the math), our published claims and their provenance status, benchmark-parity discipline, the RLM experiment dossier. |
| `fastworkflow-debugging-playbook` | Symptom → triage table for runtime/build/train failures, each trap with its story and a discriminating experiment; the incident index. First stop when something misbehaves. |
| `fastworkflow-failure-archaeology` | The chronicle: every major investigation, revert, dead end, and abandoned feature as symptom → root cause → evidence → status, so no one re-fights a settled battle. Canonical home of the incident narratives. |

### Advanced — how to advance the project

| Skill | One line |
|---|---|
| `tau2-reliability-campaign` | THE executable, decision-gated campaign for the hardest live problem (E0–E25, "Governed Determinism", 15/15 pass^3 target): phases, gates with expected numbers, ranked solution menu, fenced-off wrong paths, promotion protocol through change control. |
| `fastworkflow-proof-and-analysis-toolkit` | First-principles recipes with worked examples from this repo: pass^k/variance arithmetic, simulator-ceiling derivation, blind dual-rater attribution, adversarial design review, root-cause discipline, calibration analysis, pre-registration. |
| `fastworkflow-research-frontier` | The four ranked frontiers (pass^k reliability discipline; RSI-for-reliability; Forge self-hosting; small-model parity extended): why SOTA fails, our specific assets, first three concrete steps in this repo, falsifiable milestones. |
| `fastworkflow-research-methodology` | The constitution: the evidence bar (one mechanism explains ALL observations; survives assigned refutation; pre-registered numbers), the idea lifecycle from hunch to adopted change or documented retirement, where good ideas historically came from. |
| `fastworkflow-intent-training-convergence` | The loop for growing intent-detection quality: why the reported training F1 measures memorisation, why seeding did **not** make two runs comparable, how to measure your own run-to-run noise floor before believing anything, the minimum-detectable-effect table, growing seeds/personas one axis at a time, paired McNemar scoring on both routing and escalation (`scripts/score_benchmark.py`), and the stop rule. Added 2026-08-01; re-pointed 2026-08-02 at the in-package held-out evaluation and artifact versioning. |

## Suggested onboarding order

1. `fastworkflow-change-control` (the rules that protect you)
2. `fastworkflow-build-and-env` → `fastworkflow-run-and-operate` (get it running)
3. `fastworkflow-architecture-contract` (how it works)
4. `fastworkflow-validation-and-qa` + `fastworkflow-debugging-playbook` (working on it safely)
5. Domain packs as needed; the Advanced four before touching the reliability program.

## Maintenance

- Facts are date-stamped 2026-07-09. Anything volatile has a re-verification one-liner in
  its skill's Provenance section — trust those over the prose after significant releases.
- When a skill's area changes (new release, epic closed, doc committed), update the skill
  in the same working session and refresh its date stamp.
- One home per fact: if you find the same explanation in two skills, the cross-referenced
  home wins; trim the copy.
