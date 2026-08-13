# The fix-vof adversarial review runbook

> TEAM-PRIVATE: embeds content from uncommitted internal docs. Do not commit or publish this skill without the developer'sexplicit approval.

This is the reusable, step-by-step procedure for design-level changes, reconstructed from the TurnResult epic (`fix-vof`, 2026-06-10/11) — the repo's flagship example of paper-first change control. All artifact paths are on main and git-tracked.

The cast of artifacts, in creation order:

| Artifact | Role | Scale |
|---|---|---|
| `docs/turn_result_design.md` | Design doc: origin bug, root cause, type algebra, 22-entry decision log, Amendments A1–A47 | 1424 lines |
| `docs/turn_result_design_review.md` | Adversarial review: findings R1–R48 with severity index | 1577 lines |
| `docs/turn_result_architecture_review.md` | Second-order ten-concern review of the *finalized* design: X1–X12 | 297 lines |
| `docs/turn_result_design_final.md` | Consolidated authoritative spec with supremacy clause + traceability table | 620 lines |
| `docs/turn_result_design_feedback.md` | Post-implementation human teaching/pushback session | 354 lines |
| `docs/turn_result_learning_checklist.md` | Staged explain-back mastery tracking | living doc |

## Step 1 — Root-cause narrative + design doc

Write the design doc in `docs/`. Mandatory content:
- **The origin story with the negative observations.** The TurnResult doc opens with the bug (xray `/invoke` returned `payload=None`), what was verified correct (producer, mapping, handler), the two actual drop points, and the "tell" — the deterministic path that did NOT fail. A mechanism that does not explain the negatives is not accepted (evidence-bar Rule 1).
- **A numbered decision log** (the TurnResult doc has 22 entries) so reviewers can attack decisions individually.
- Honest open-questions section. Do NOT write "none remaining" — that exact phrase preceded 48 findings and was publicly rewritten as an admission (`docs/turn_result_design.md:769-777`).

Commit shape: design doc + review created together in one commit is fine (`e05652a` "docs: TurnResult design + critical review (48 findings); beads epic fix-vof") — but remember: **no git commit or push without the developer'sexplicit request in that turn** (bd memory, established 2026-07-08).

## Step 2 — The adversarial review pass(es)

Task an independent reviewer (agent or human) explicitly to **refute** the design. Non-negotiables, all evidenced in the review doc's header:

- **Pin the baseline**: "All code references were verified against the working tree at the time of review (v2.20.0, commit `dc250b3`)." Claims about the codebase must cite file:line at that commit.
- **Number every finding** (R1…Rn) and build a severity index table up front.
- **Run a second, deeper pass.** R37–R47 came from pass 2 and "uncovered — among other things — an existing turn-persistence subsystem the design does not account for." Pass 2 must re-check the design's *factual baseline*, not just its logic.
- The verdict may be positive overall while still listing blocking findings. The fix-vof verdict: diagnosis correct, core design sound, AND four internal contradictions (R1, R2, R37, R42) plus blocking gaps.

## Step 3 — bd epic, one child per finding, review-only

Create one epic with one child issue per finding. The fix-vof epic description reads "review-only — no implementation. Work one finding at a time." 48 children (`fix-vof.1`–`.48`), all closed one at a time.

bd discipline while doing this (memory `beads-flakiness-observed-2026-06-11`): one write at a time, verify `.beads/issues.jsonl` after each. (`bd close --reason` persists on bd 1.1.2, retested 2026-08-13; the 2026-06-11 failure was an older bd.)

Working preference on record (memory, established 2026-06-10 during R11): give Dhar an ELI5 plain-language explanation of each finding BEFORE asking decision questions.

## Step 4 — One docs commit per resolution, standard message shape

Each finding's resolution is a single docs-only commit. The canonical shape, from `git show 42e9b3a`:

```
docs: resolve R1 - ask_user exchanges modeled as command executions (the developer'sdesign)

<decision summary paragraph: what was decided and why, in past tense>

- design doc: Amendment A7
- review doc: R1 marked resolved
- beads: fix-vof.1 closed (epic 7/48)
```

Mechanics:
- The decision is recorded as a numbered **Amendment** (A1–A47) appended to the design doc; amendments supersede the original sections wherever they conflict.
- The review doc's finding row is marked `RESOLVED <date> — <one-line outcome>`.
- The body carries the running epic counter (`epic 7/48`).
- Resolutions ran from `fe3981d` (R37, first) to `920d0de` ("docs: resolve R34-R36, R47, R48 - epic fix-vof complete (48/48)").

## Step 5 — Second-order review of the FINALIZED design

After all findings resolve, review the *amended* design again — this time with **parallel per-concern reviewers** asking "what did the amendments themselves miss or under-price?" The fix-vof instance (`docs/turn_result_architecture_review.md`, commit `9031942`): ten parallel review agents, one each for performance / reliability / scalability / code-size / readability / observability / manageability / testability / ease-of-use / deployment, producing findings X1–X12.

This step is not ceremony — it reversed accepted decisions:
- **X1 reversed A23.1** (no write-time index): Redis `SCAN MATCH` walks the entire keyspace; three of the ten agents independently flagged it as "the worst under-priced decision."
- **X3 found the A10×A28 contradiction**: strict serialization rejection × best-effort record writes = deterministic silent permanent record loss.
- **X6 shrank the implementation 5×**: from A14's plan of ~2.5k lines/120 files to ~500 lines/6 files, user-approved 2026-06-11.

## Step 6 — Consolidate into one authoritative spec

Write the final spec (`docs/turn_result_design_final.md`) that:
- Opens with a **supremacy clause**: "Where this document conflicts with any of those, **this document wins**; they remain the rationale archive."
- Carries **traceability tags** (`[A7]`, `[X3]`) linking every requirement back to its decision, plus a traceability table (final.md §18).
- Records what is deliberately deferred (the v3.0 roadmap sections) so absence is documented scope, not oversight.

## Step 7 — Post-implementation teaching session (the verification that has caught shipping bugs)

After implementation, a human explain-back session re-verifies the shipped code against the spec (`docs/turn_result_design_feedback.md`). This is where the near-miss was caught:

**Topic 5 (2026-06-12):** the reviewer verified against code that `_finalize_agent_output` was UNCHANGED from the buggy version — v2.21 as staged only *captured* turns into a dormant TurnResult and did not surface payloads. The origin bug would have shipped "fixed" while still user-visible. Recorded warning: "Don't merge v2.21 believing the payload bug is fixed." Resolved the same day by wiring the projection at the WEC finalize chokepoint (the single point both CLI and FastAPI flow through), with the cleaner transport-edge shape deferred to v3.0.

Other outcomes of the same session, showing pushback is generative, not adversarial theater:
- **Topic 1 (turn_key identity fight):** reviewer argued turn_key is redundant vs (conversation_id, ordinal); evidence-gathering (grepping the runtime for locks/leases) established single-writer-per-channel is an UNENFORCED assumption; the reviewer's stronger alternative ("add a lease, delete the UUID") is recorded for v3.0.
- **Topic 3 (process_message deprecation contested):** resolved by showing the reviewer's own Topic-2 position logically forces the deprecation. Rule extracted: never a silent return-type swap on a public method — deprecate + add the new entry point.
- Success semantics went through **two rejected interims** before landing on the fully orthogonal status/failure_reason/success model — the rejects are recorded (`feedback.md:317-326`), which is what "documented retirement" looks like at decision granularity.

Track the human-mastery state in a learning checklist (see SKILL.md §4).

## Scaling the template down

Not every change needs 48 findings and six documents. The invariant kernel for ANY design-level change:
1. Written design with decision log, in `docs/`.
2. Someone explicitly tasked to refute it against a pinned commit, findings numbered.
3. Every finding resolved with a written, traceable decision (bd + doc).
4. One authoritative final statement of what won.
5. Post-implementation re-verification by someone explaining it back.

## Re-verification

Facts verified 2026-07-09 against v2.22.2 (commit c33b9a5).

```bash
head -20 docs/turn_result_design_review.md            # pinned commit, 48 findings, pass-2 note
sed -n '769,777p' docs/turn_result_design.md          # the rewritten overclaim
head -8 docs/turn_result_design_final.md              # supremacy clause
head -6 docs/turn_result_architecture_review.md       # ten parallel agents
grep -n "Topic 5" docs/turn_result_design_feedback.md # the near-miss
git show --stat 42e9b3a                               # resolution commit shape
git show -s --oneline e05652a fe3981d 920d0de 9031942  # bookend commits
bd show fix-vof                                       # epic: closed, 48 children
```
