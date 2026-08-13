---
name: fastworkflow-research-methodology
description: Load when you are about to propose, run, review, accept, or retire a research idea or experiment in this repo — trigger phrases include "I have a hypothesis", "let's try adding X and see if the score improves", "the numbers went up so it works", "should we pin the simulator temperature", "write up this experiment", "is this result real", "kill this idea", or any request to design an E-card / pre-registration / adversarial review. Do NOT load for executing the tau2 campaign step-by-step (use tau2-reliability-campaign), for statistics recipes and pass^k math (use fastworkflow-proof-and-analysis-toolkit), for picking WHICH open problem to work on (use fastworkflow-research-frontier), or for ordinary bug fixing (use fastworkflow-debugging-playbook).
---

> TEAM-PRIVATE: embeds content from uncommitted internal docs. Do not commit or publish this skill without the developer'sexplicit approval.

# fastWorkflow Research Methodology

The discipline that turns a hunch into an accepted result in this repo — or into a documented retirement. This is the constitution; the `tau2-reliability-campaign` skill is the worked application of it.

The house summary, from the plan's own documentation standard: **"if it isn't written here with a number and an interval, it didn't happen"** (`docs/tau2_retail_reliability_implementation_plan.md`, Appendix G).

## When to use / when NOT to use

| Situation | Skill |
|---|---|
| You have an idea and want to know how to make it an accepted result here | **this skill** |
| You are running the E0–E25 retail reliability experiments in order | `tau2-reliability-campaign` |
| You need the actual variance / pass^k / McNemar / CI math with worked examples | `fastworkflow-proof-and-analysis-toolkit` |
| You want to know which open problems are worth attacking | `fastworkflow-research-frontier` |
| You need tau-bench mechanics (harness, simulator, pass@1 vs pass^k, parity rules) | `fastworkflow-taubench-reference` |
| You need what counts as test evidence / how to add pytest tests | `fastworkflow-validation-and-qa` |
| Your change is code, not research — how it gets classified/gated/reviewed | `fastworkflow-change-control` |
| You want the history of every dead end and abandoned feature | `fastworkflow-failure-archaeology` |
| You are writing a design doc or deciding what may be claimed publicly | `fastworkflow-docs-and-positioning` |

## Glossary (define once, use everywhere)

| Term | Meaning |
|---|---|
| **pass@1 / pass^k** | pass@1 = fraction of tasks passing one run. pass^k = fraction passing **all k** independent runs. pass^k measures *reliability*; pass@1 measures capability. |
| **τ²-bench retail** | Public benchmark: agent plays a retail customer-service rep against a *simulated user* (another LLM). Scoring is binary and conjunctive on final DB state — one wrong write = 0 for the task. |
| **The 15 hard tasks** | Hand-picked τ²-retail task IDs that failed under memory pressure (plan Appendix F). An adversarial sample AND our months-long tuning set — which is why anti-p-hacking rules exist. |
| **bd (beads)** | The issue tracker (`bd` CLI, source of truth `.beads/issues.jsonl`). Mandatory for ALL tracking (AGENTS.md). |
| **E-card** | Experiment card: the plan's unit of proposed work (E0–E25 in `docs/tau2_retail_reliability_implementation_plan.md` §7). Hypothesis with predicted numbers, design, metrics, gate criteria, dependencies. |
| **Pre-registration** | Writing hypothesis + config + metrics + analysis plan, and committing it, **before** the run. |
| **Adversarial review** | A reviewer (human or agent) explicitly tasked to *refute* a design against the actual working tree at a pinned commit — the fix-vof pattern. |
| **McNemar / Clopper-Pearson / Wilson** | Paired before/after significance test; exact/approximate confidence intervals on proportions. Recipes in `fastworkflow-proof-and-analysis-toolkit`. |
| **Ablation** | Measuring one change at a time so effects are attributable. |
| **RSI** | Recursive self-improvement — here always *bounded* and for *reliability* (E21), never free-running. |

---

## 1. The evidence bar

Five rules. Each exists because of a specific failure. A claim that does not clear all five is labeled "open" / "candidate" / "unverified" — never asserted.

### Rule 1 — One mechanism must explain ALL observations, including the negatives

A root cause is accepted only when it predicts every observation: the failures AND the paths that did *not* fail.

The canonical case (`docs/turn_result_design.md`, §"Origin"): xray's `/invoke` always returned `payload=None` in agent mode. Producer, mapping layer, and handler were each verified correct. The accepted mechanism (DSPy ReAct tools return text-only observations + `_finalize_agent_output` synthesized a fresh answer-only output) was accepted *because* it also explained the negative: the deterministic `/`-prefixed path did NOT have the bug — the diagnostic "tell". A candidate mechanism that couldn't explain why the deterministic path worked would have been wrong.

Corollary from the Articles (Article 1 PDF, methodology): attribute failures to a layer with **exact log lines**, never inferred from the final score. E0 hardens this into blind dual-rater re-attribution with Cohen's κ reported (plan §6, item 5).

### Rule 2 — Claims survive assigned adversarial refutation

Design-level claims are not "done" when the author is satisfied. They are done when an independent reviewer, explicitly tasked to refute them against the actual codebase at a pinned commit, has run out of findings — and every finding has a recorded resolution.

The proof this is necessary: the original TurnResult design declared "Open questions: none remaining." The systematic review (`docs/turn_result_design_review.md`, verified against the working tree at commit `dc250b3`) found **48 findings including four internal contradictions**, and the design doc's section 13 was publicly rewritten as an admission (`docs/turn_result_design.md:769-777`). The full reusable procedure — the fix-vof template — is in [references/adversarial-review-runbook.md](references/adversarial-review-runbook.md).

### Rule 3 — Measured k-run numbers with variance accounting; never single runs; never eyeballs

The founding incident: Article 3 reported 2/15 → 5/15 after mitigations. The critical review dissolved it (plan §2.4): n=15, single runs, unfixed temperature, and **the two originally-passing tasks were NOT among the five new passes** (plan §2.2) — statistically indistinguishable from noise. The whole reliability program's first deliverable (E0) exists to make this class of claim impossible to repeat.

The standing bar for any measured claim (plan §6):
- k ≥ 5 runs per configuration; report all runs, not the best.
- Confidence intervals (Clopper-Pearson/Wilson) on every proportion; McNemar's exact test for paired before/after.
- Measure and report the **irreducible variance floor** (k runs of an *identical* config) under every number — provider nondeterminism is real.
- pass^k math is unforgiving: at per-task p=0.98, a 15/15 pass^3 sweep succeeds only ~40% of the time (plan §3.1). "100% pass^3 is a determinism claim, not a capability claim."

### Rule 4 — Pre-registered predictions: the hypothesis states numbers BEFORE the run

Every E-card's hypothesis is a falsifiable numeric prediction written before any measurement, e.g. E3: "deterministic pre-commit invariants convert **≥4 of the 10** post-mitigation failures from wrong-writes into blocked-then-repaired successes"; E1: "recovers **≥1 task** at zero design cost"; E2: "serves **≥95%** of recall queries" (plan §7). Appendix G makes pre-registration section (1) of every experiment doc, "committed **before** the run."

The extreme form is the Gate-3 protocol (plan §3.3): the final 15/15 pass^3 attempt is **frozen, pre-registered, run once, reported** — with an honest prior success estimate (25–40%) stated in advance, and always paired with the full ~115-task retail split so the 15 read as the stress subset, not the headline.

### Rule 5 — Disclosed trades only

Any deviation from standard conditions is stated in the report, never silent. The two live examples:
- **Simulator temperature/seed pinning** raises the reliability ceiling but "changes standard benchmark conditions — a disclosed trade if we make it" (plan §3.2, item 1).
- **Benchmark parity is sacred**: tau-bench tools/tasks are never modified for benchmark runs. E3's invariants live in `validate_extracted_parameters` / Pydantic validators with the "Tool layer untouched (benchmark parity)" note written into the card itself (plan §7, E3).

Also disclosed in every report: the frontier/runtime role separation — dev-time tooling (RSI proposer, E24 generator) may be frontier-class, the runtime under test stays gpt-oss-20b (`docs/rsi_harness_agent_report.md`).

---

## 2. The idea lifecycle

```
hunch → bd issue → experiment card (E-card) → gated experiment
      → adversarial review (design-level changes only)
      → ADOPT (versioned release + docs)  or  DOCUMENTED RETIREMENT
```

### Stage 1 — bd issue before code

Issue-before-code is mandatory (AGENTS.md: "bd for ALL issue tracking... Do NOT use markdown TODOs"). Link discovered work with `discovered-from` dependencies. The plan itself mandates it for experiments: "Track all work items as bd issues under a single epic; claim before starting, close with reasons" (plan §9).

Discipline rules, learned the hard way (bd memory `beads-flakiness-observed-2026-06-11`):
- **One bd write at a time; verify `.beads/issues.jsonl` after each write.**
- **`bd close --reason=...`** was observed reporting success while silently failing to persist (2026-06-11, older bd); retested 2026-08-13 on bd 1.1.2 it persists 12/12. Verify the JSONL either way — that half never stops applying.
- Patching `issues.jsonl` directly is safe (every command re-imports it as source of truth).
- Likely collateral of ignoring this: `fix-7kp` is a duplicate of `fix-2jo` created 9 seconds apart; `fix-2jo` closed, `fix-7kp` still open.

Status check (verified 2026-07-09): **no E0–E25 bd issues exist yet** — `.beads/issues.jsonl` has 93 issues, none matching the E-catalog. Filing that epic is the first paper-trail act of Phase 0.

### Stage 2 — write the experiment card

The E-card is the unit of proposal. Card format (plan §7, line 299): **ID. Name** | Effectiveness toward the goal (High/Med/Low, *with mechanism*) | Complexity (XS–XL ≈ person-days/weeks/months) | Dependencies | Parallelism — then Hypothesis (numeric, falsifiable), Design, Metrics, Failure modes before → after, Contribution. Full template with a worked example (E3) and the Appendix G experiment-doc standard: [references/experiment-card-and-preregistration.md](references/experiment-card-and-preregistration.md).

Two card-level obligations people skip:
- **Metrics must include the ways your idea could be net-negative** (E3 mandates reporting false-positive blocks "prominently — a validator that blocks correct actions is worse than none"; E4 mandates watching turn-budget exhaustion).
- **Dependencies are real gates**: E0 before any measured claim; nothing enters the measured stack before its dependencies (plan §8.3's serial list).

### Stage 3 — run it gated

- **E0-first**: no measurement infrastructure → no science. Until the harness exists, every score is an anecdote.
- **The integration-queue rule** (plan §8.4): development branches merge into the measured stack **one experiment at a time, in a declared order, with a paired k=5 before/after evaluation (McNemar) at each step**. The baseline ladder *is* the ablation table. "Nothing enters the stack unmeasured; nothing is measured while entangled."
- Every experiment lands a `docs/experiments/EXX_<name>.md` with the 8 mandatory Appendix G sections *as it completes* — lint yours with [scripts/lint_experiment_doc.py](scripts/lint_experiment_doc.py). (`docs/experiments/` does not exist yet; E0 creates it.)

### Stage 4 — adversarial review for design-level changes

Anything that changes architecture, public types, persistence, or wire contracts goes through the fix-vof template before implementation. Summary (full runbook in [references/adversarial-review-runbook.md](references/adversarial-review-runbook.md)):

1. Root-cause narrative + design doc in `docs/` with a numbered decision log.
2. Independent adversarial review against the ACTUAL working tree at a pinned commit; numbered findings (R1…Rn) with a severity index. Run a second, deeper pass — R37–R47 came from pass 2 and found an existing subsystem the design ignored.
3. One bd epic, one child per finding, **review-only** ("no implementation. Work one finding at a time").
4. One docs commit per resolution with the standard message shape (`docs: resolve R# — <decision>`; body lists the amendment tag, review-doc mark, bd closure, epic counter — see commit `42e9b3a`).
5. Second-order review of the *finalized* design by parallel per-concern agents (X1–X12 pattern) asking "what did the amendments themselves miss." This reversed decisions the first review had accepted (X1 reversed A23.1).
6. Consolidate into a single authoritative spec with a supremacy clause ("where this document conflicts with any of those, this document wins" — `docs/turn_result_design_final.md:3-8`) and a traceability table.
7. Post-implementation human teaching session that re-verifies shipped code against the spec. This step caught v2.21 about to ship with the original bug still user-visible (`docs/turn_result_design_feedback.md`, Topic 5: "Don't merge v2.21 believing the payload bug is fixed") — wired same-day at the WEC finalize chokepoint.

### Stage 5a — adopt

Adoption = versioned release whose commit message records the root-cause narrative, the bd epic, and the **accepted limitations** (e.g. `cf3eeae` v2.22.0 names fix-85g and "the three accepted limitations"; the design doc `docs/fastworkflow_turns_async_execution_design.md` ships in the same release), plus docs updated, plus bd epic closed with reasons. Release/commit mechanics and the NEVER-commit-without-Dhar's-explicit-request rule (established 2026-07-08 after a private doc was auto-pushed to the public repo, forcing a history rewrite): see `fastworkflow-change-control`.

### Stage 5b — documented retirement

Killing an idea is a first-class outcome — Appendix G section (5): "negative results are results." A retirement must leave: a bd close with reason, a status line in the relevant doc, and (for shipped features) removal commits that say why.

The historical failure mode this rule exists to prevent: **probabilistic response generation died silently.** Its spec (`docs/probabilistic_response_generation.md`) is on main (commit `ddab25c`); the implementation exists only on abandoned branch `origin/cursor/implement-probabilistic-response-generation-and-pass-tests-7586` (commit `7c7d1cf`); `fastworkflow/probabilistic_response.py` does not exist on main; nothing anywhere says it was abandoned. Consequence every reader must internalize: **spec docs on main can describe unimplemented features.** Contrast with a retirement done adequately via commit messages, though spread across releases: the command dependency graph (built `55a8668` v2.10.0, tests removed `ed21d77` v2.15.8, dependency excised `ad68d9b` v2.17.19, whose message calls the feature "now unused"). The full chronicle of dead ends lives in `fastworkflow-failure-archaeology`.

Pre-committed kill criteria are part of the card where feasible — E21 states its own: "if after ~4 weeks the loop's promotions plateau below Gate 2, stop and return to human-led Phase-3 work. Kill criterion stated now, per the plan's discipline" (`docs/rsi_harness_agent_report.md`).

---

## 3. Where good ideas historically came from

Mined from git history and the docs of record. Watch these five wells; they are where the next ideas will come from too.

| Source | Example | Evidence |
|---|---|---|
| **Production bug post-mortems** | xray payload-loss bug → the entire TurnResult type algebra, turn accumulator, and v3.0 persistence roadmap | `docs/turn_result_design.md` origin section; shipped v2.21 (`afcbe01`) |
| **Benchmark failure analysis** | The three Article PDFs' 15-task failure taxonomy → the whole tau2 reliability program (E0–E25, Governed Determinism) | plan §2.3–§2.5; `docs/Article 2 - The Failure Taxonomy.pdf` |
| **Adversarial review findings themselves** | R-findings spawned Amendments A1–A47; second-order X-review reversed A23.1 (write-time index) and shrank v2.21 scope 5× (X6: ~2.5k lines/120 files → ~500 lines/6 files) | `docs/turn_result_design_review.md`; `docs/turn_result_architecture_review.md` |
| **Teaching / pushback sessions** | The feedback session caught v2.21's "fix" not fixing the user-visible bug pre-merge; the turn_key identity fight produced the recorded stronger v3.0 alternative | `docs/turn_result_design_feedback.md` Topics 1, 3, 5 |
| **Dogfooding pain** | Prose SKILL.md guidance for building workflows = "level-5 enforcement (signs)" — same failure classes as the retail agent, one meta-level up → Forge, the self-hosting meta-workflow | `docs/forge_meta_workflow_spec.md` §1.1 |

Outside ideas enter through the same discipline: the Analysis docx's four intern research ideas were absorbed and *retargeted* into cards (Idea 1 → E17 conditional; Idea 2 → E3's deterministic tiers; Idea 3 → E5/E10; Idea 4 → E14 rescoped after the frozen-suffix-replay validity objection) rather than adopted as proposed (plan cards E3/E5/E14/E17). Retargeting-with-credit is the norm, not a slight.

---

## 4. The learning-checklist practice (the mastery gate for humans)

Big changes ship with a staged learning checklist, and a human has not "learned" an item until they **explain it back** (or pass a quiz) — including "the why behind the why." Two live artifacts:

- `WEC_learning_checklist.md` (repo root) — Topology A→B migration. Header rule: "We check items off only after you've *demonstrated* understanding... Nothing is 'done' just because we discussed it." Legend: `[ ]` not yet · `[~]` in progress · `[x]` mastered.
- `docs/turn_result_learning_checklist.md` — TurnResult. Same legend; notes record *how* mastery was demonstrated (e.g. 2.2: "Mastery demonstrated via the critique itself" — the learner's pushback was the proof).

Two operational implications:
1. **The checklist is honest state, not ceremony.** As of 2026-07-09 the TurnResult checklist has unmastered items (2.5–2.8, 2.10, all of Stage 3) — those areas are exactly where an onboarding engineer should expect thin shared understanding.
2. **The teaching session doubles as verification** (Stage 4 step 7 above): explaining the system to a skeptical human re-verifies the code against the spec, and has caught shipping bugs.

Related standing preference (bd memory, established 2026-06-10 during R11): in collaborative reviews with Dhar, ALWAYS give an ELI5 plain-language explanation of a finding BEFORE asking decision questions.

When onboarding a new team member to a subsystem, create a checklist in this exact format and work it explain-back style. It is the project's proven mechanism for transferring deep context.

---

## 5. Anti-patterns — fenced off

Each of these was actually committed (by us or in the Articles) or is a live temptation. Do not do them; if you see one in a draft, refute it citing this table.

| # | Anti-pattern | The incident | The rule that replaced it |
|---|---|---|---|
| 1 | **Bundled mitigations** — shipping several changes and measuring once | Article 3 bundled four mitigations with no ablation; one of them (full tool-output view for the planner) silently reintroduced the context bloat the memory layer existed to solve (plan §2.4) | Integration-queue rule (§8.4): one experiment at a time, paired k=5 McNemar at each step |
| 2 | **Declaring victory from disjoint pass sets / single runs** | 2/15 → 5/15 where the original 2 were not among the 5 — noise reported as progress (plan §2.2) | Rule 3: k≥5, CIs, variance floor under every number |
| 3 | **Tuning on the eval set** | The 15 hard tasks have been the tuning set for months; re-running pass^3 and patching until a sweep lands is test-set optimization | Gate-3 protocol: freeze, pre-register, run ONCE, report, paired with the full ~115-task split (plan §3.3); the RSI loop is permanently firewalled from the Gate-3 run (rsi report) |
| 4 | **Silent methodology trades** | Temptations on record: pinning sim temperature/seed; editing tau-bench tools "just for the run" | Rule 5: parity is sacred; any nonstandard condition is a disclosed trade in the report, never silent |
| 5 | **Skipping the bd paper trail** | Undocumented work is unrecoverable after context loss; duplicate issues (fix-7kp/fix-2jo) came from unverified writes | Issue-before-code; one bd write at a time; verify `issues.jsonl` |
| 6 | **"Open questions: none remaining"** — self-certifying a design | The TurnResult design said exactly that; the adversarial review found 48 findings incl. 4 contradictions, and the section was rewritten as a public admission (`docs/turn_result_design.md:769-777`) | Rule 2: a design is not done until adversarially reviewed against the actual codebase |
| 7 | **Silent retirement** — letting an idea die on a branch | probabilistic-response-generation: spec on main, implementation abandoned, no death certificate | Stage 5b: bd close with reason + doc status line; negative results are results |
| 8 | **Fixing the symptom the checklist would have caught** — merging on "we discussed it" | v2.21 nearly shipped with the origin bug still user-visible; only the explain-back teaching session caught it | Stage 4 step 7: post-implementation re-verification session before believing a bug is fixed |

---

## 6. Pre-claim checklist (copy-paste before asserting any result)

```
[ ] bd issue exists, claimed, discovered-from linked; issues.jsonl verified after every write
[ ] E-card (or equivalent) written: numeric hypothesis, design, metrics incl. net-negative
    metrics, deps, gate criteria — committed BEFORE the run
[ ] Measured with the harness (E0 once it exists): k>=5, temperature/seeds controlled,
    all runs reported
[ ] CIs on every proportion; McNemar vs. the ladder predecessor; variance floor reported
[ ] One mechanism explains all observations INCLUDING the negatives (exact log lines,
    not score inference)
[ ] Entered the measured stack alone (integration queue), not entangled with other changes
[ ] Any nonstandard condition (sim pinning, etc.) disclosed in the report
[ ] Design-level change? fix-vof adversarial review completed, findings resolved,
    final spec has supremacy clause
[ ] docs/experiments/EXX_*.md has all 8 Appendix G sections
    (scripts/lint_experiment_doc.py passes)
[ ] Negative/killed? bd closed with reason + doc status line — no silent retirement
[ ] NOT committed/pushed without the developer'sexplicit request in this turn
```

---

## Provenance and maintenance

Facts verified 2026-07-09 against v2.22.2 (commit c33b9a5). The tau2 plan, RSI report, Forge spec, Article PDFs, and Analysis docx are **untracked team-private files** — they can change or vanish without git history; re-check them first.

| Volatile fact | Re-verify with |
|---|---|
| Plan sections cited (§2.2–2.5 at lines ~90–177, §3 at 179, §6/E0 at 274, card format at 299, §8.4 at 793, §9 at 799, Appendix G at 943) | `grep -n "^##" docs/tau2_retail_reliability_implementation_plan.md` |
| "None remaining" rewrite at `docs/turn_result_design.md:769-777` | `sed -n '769,777p' docs/turn_result_design.md` |
| Review pinned at commit dc250b3; 48 findings; 4 contradictions | `head -20 docs/turn_result_design_review.md` |
| Final-spec supremacy clause | `head -8 docs/turn_result_design_final.md` |
| Feedback Topic 5 near-miss + same-day resolution | `grep -n "Topic 5\|RESOLUTION" docs/turn_result_design_feedback.md` |
| Resolution-commit message shape | `git show --stat 42e9b3a` (also `e05652a`, `920d0de`) |
| bd memories (flakiness, ELI5, never-commit) | `bd memories --json` |
| No E0–E25 bd issues filed yet | `python3 -c "import json,re;print([json.loads(l)['title'] for l in open('.beads/issues.jsonl') if re.search(r'\bE[0-9]{1,2}\b',json.loads(l).get('title',''))])"` |
| probabilistic_response absent on main, spec present | `ls fastworkflow/probabilistic_response.py; git log --oneline -1 -- docs/probabilistic_response_generation.md` |
| Command-dependency-graph retirement commits | `git log --oneline -1 55a8668 && git log --oneline -1 ed21d77 && git log --oneline -1 ad68d9b` |
| Learning-checklist unmastered items | `grep -n "^- \[ \]" docs/turn_result_learning_checklist.md WEC_learning_checklist.md` |
| `docs/experiments/` still absent (E0 not started) | `ls docs/experiments 2>&1` |
| Current version | `grep '^version' pyproject.toml` |
| AGENTS.md bd mandate | `grep -n "ALL issue tracking" AGENTS.md` |
