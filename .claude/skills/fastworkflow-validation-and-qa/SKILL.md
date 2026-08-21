---
name: fastworkflow-validation-and-qa
description: >
  What counts as evidence in fastWorkflow: the real state of the 1803-test pytest
  suite, which tests skip/no-op and why, the golden test workflows and trained
  artifacts you must never destroy, how to add integration tests, and the
  acceptance bar before a release tag. Load this when you are about to run the
  test suite, interpret pass/fail/skip results, add or modify tests, decide
  whether "CI is green" means anything (it barely does), or judge whether a
  change has enough evidence to ship. Symptoms that trigger it: "tests pass but
  should they?", "why did 4 FastAPI tests suddenly fail with FileNotFoundError
  on threshold.json", "can I mock this?", "is this test flaky?", "what do I run
  before tagging a release?". Do NOT load it for debugging a failing feature
  (fastworkflow-debugging-playbook), for running the app itself
  (fastworkflow-run-and-operate), or for pass^k reliability measurement design
  (tau2-reliability-campaign / fastworkflow-proof-and-analysis-toolkit).
---

# fastWorkflow Validation and QA — what counts as evidence

> TEAM-PRIVATE: embeds content from uncommitted internal docs. Do not commit or publish this skill without the developer'sexplicit approval.

Facts verified 2026-07-09 against v2.22.2 (commit c33b9a5). Re-verification commands are in the final section — run them before trusting any volatile number.

## When to use / when NOT to use

| You want to... | Use |
|---|---|
| Run/interpret the pytest suite, add tests, protect golden artifacts, judge shipping evidence | **this skill** |
| Triage a runtime/build/train failure by symptom | `fastworkflow-debugging-playbook` |
| Set up venv, keys, `env/.env` + `passwords/.env` from scratch | `fastworkflow-build-and-env` |
| Launch the CLI / FastAPI+MCP server and see artifacts | `fastworkflow-run-and-operate` |
| Design pass^k experiments for the tau2 retail program (E0-E25) | `tau2-reliability-campaign` |
| Variance / pass^k math, attribution, calibration recipes | `fastworkflow-proof-and-analysis-toolkit` |
| The evidence lifecycle for research claims (hunch -> accepted result) | `fastworkflow-research-methodology` |
| Rules on what changes need approval; the non-negotiables | `fastworkflow-change-control` |
| Measure instead of eyeball (diagnostic scripts) | `fastworkflow-diagnostics-and-tooling` |

Jargon used below, defined once:
- **Golden workflow** — a small committed application in `tests/` or `fastworkflow/examples/` that tests run against instead of mocks.
- **`___command_info/`** — the directory of trained artifacts (BERT model dirs `tinymodel.pth`/`largemodel.pth`, `threshold.json`, `label_encoder.pkl`, plus `command_directory.json`/`routing_definition.json`) that `fastworkflow train` writes inside a workflow folder. Gitignored (`.gitignore` lines 2-5).
- **Env gating** — tests that `pytest.skip` themselves unless local secret files exist.
- **pass@1 / pass^k** — probability a task passes one run / passes ALL of k independent runs. pass^k is the reliability metric of the tau2 program.
- **bd (beads)** — the issue tracker; `bd show <id>` is read-only and safe.

## 1. Test-suite reality

**Run command** (the only sanctioned way; AGENTS.md line 3 requires the venv):

```bash
cd /path/to/fastworkflow
source .venv/bin/activate
python -m pytest
```

| Fact | Value (verified 2026-07-09) |
|---|---|
| Collected tests | **1860** (re-measured 2026-08-20 at v3.1.1; 1803 at v3.1.0, 495 at v2.22.2 — this number moves fast, re-run rather than trusting it) |
| pytest / python | pytest 9.0.3 on venv Python 3.12.2 |
| pytest config file | **None.** No `pytest.ini`/`setup.cfg`/`tox.ini`; `pyproject.toml` has no `[tool.pytest.ini_options]`. Markers `integration`/`slow` are registered dynamically in `tests/conftest.py:101-108` |
| Makefile test target | **None** (targets: gen-env, lint, audit, audit-json, publish-testpypi, publish) |
| Coverage | `pytest-cov` is installed; no coverage is configured or ever reported |
| Last recorded full-suite baseline | **1846 passed / 14 skipped / 0 failed / 0 errors, 1860 collected, 50:03** — measured 2026-08-20 at v3.1.1 on the owner's provisioned WSL2 box (bd fix-86c / fix-0xn / fix-11r / fix-63k). Supersedes the v3.1.0 figure (1788/15/0 at 1803 collected, 48:56) and the older 478/0 at fa97b48 (2026-06-15). **Caveat: this still does not meet the two-consecutive-runs bar below.** Two full runs were made that day, both 0 failures, at 51:40 and 50:03 — but on ADJACENT TREES (the first with only fix-86c landed, the second with all four), so they establish that wall time is stable and say nothing about idempotence. The v3.1.0 checksum finding was not re-checked: 10 build-cache JSONs (`command_directory.json`, `routing_definition.json` in five example workflows) rewritten by the build tests, no `.pth`, no `threshold.json`, nothing deleted — the suite is not strictly byte-idempotent, though not in the way Rule 3 cares about |
| Fixed overhead | **Fixed 2026-08-13 — this row's old claim is obsolete.** It said an autouse fixture sleeps 0.5 s after *every* test (≈4 min of dead time at 495 tests, which would be ~15 min at 1803) and that the underlying thread-lifecycle issue was unfixed. `tests/conftest.py:151-169` now joins only when a `ChatWorker` is actually alive and falls back to the 0.5 s sleep just for a stubborn one; its own comment records that the blanket sleep cost ~4 minutes and that only a handful of tests start those threads. Do not reintroduce a blanket sleep |
| Global-state warning | `tests/conftest.py:33-58` injects `fastworkflow/examples/{retail_workflow, simple_workflow_template, hello_world}` into `sys.path` for the whole session; tests also mutate `fastworkflow._env_vars` and drop `./___workflow_contexts` in CWD. Never assume a clean `os.environ` |

### Environment gating map — what runs where

| Test group (files) | Gate | Behavior when gate fails |
|---|---|---|
| `test_fastapi_service.py`, `test_fastapi_topology_b.py`, `test_fastapi_turns_async.py`, `test_probes.py`, `test_mcp_token_generation.py` | Repo-local `env/.env` AND `passwords/.env` exist (both gitignored) | `pytest.skip` (guards at test_fastapi_service.py:27-36 and equivalents) |
| 4 tests in `test_fastapi_service.py` (initialize-with-startup, invoke-assistant, session-not-found, concurrent) | Pre-trained `fastworkflow/examples/hello_world/___command_info/global/threshold.json` on disk (gitignored — a fresh clone does NOT have it) | **FAIL** with FileNotFoundError, not skip. Restore: `fastworkflow train ./fastworkflow/examples/hello_world ./env/.env ./passwords/.env` (recipe recorded in `bd show fix-0hb`) |
| `tests/test_train_modern_stack.py` (2 tests) | `datasets` package importable AND a real (non-placeholder) `LITELLM_API_KEY_SYNDATA_GEN` | `pytest.skip` (guards at test_train_modern_stack.py:113-121). The BUNDLED `fastworkflow/examples/fastworkflow.passwords.env` key is a dead placeholder — only repo-local `passwords/.env` works |
| `test_react_available_commands.py::test_react_e2e_with_commands_injection` | permanently `@pytest.mark.skip` (line 99); also gated on `OPENAI_API_KEY` | skip |
| ~6 runtime test files using `tests/example_workflow` (`test_command_router.py`, `test_command_routing.py`, `test_python_utils.py`, `test_what_can_i_do_command.py`, `test_what_is_current_context_command.py`, `test_wildcard_inheritance.py`) | `tests/example_workflow/` is **gitignored** (`.gitignore:7`) — only 6 files (application/*.py, startup_action.json, __init__.py) are tracked; `_commands/` exists only on provisioned machines | errors/failures on a fresh clone. Known trap; adjudication of the fix belongs to Dhar |
| Everything else (~375 run-time tests + 116 `tests/test_build/` tests) | none — pure local | runs anywhere |

Run the gate report before interpreting any suite result:

```bash
bash .claude/skills/fastworkflow-validation-and-qa/scripts/qa_env_report.sh
```

### CI truth: green CI is almost no evidence

There is exactly **one** CI workflow: `.github/workflows/train-modern-stack.yml`. It runs exactly **one** test file (`tests/test_train_modern_stack.py`) on Python 3.11, and if the `LITELLM_API_KEY_SYNDATA_GEN` secret is not configured **the test skips and the job still passes** (stated in the workflow's own header comment). The other ~493 tests never run in CI.

Consequences you must internalize:
- A PR that breaks build-time or run-time tests will still show green CI. **Reviewers must run the suite locally.**
- "CI passed" is evidence only that training on the modern stack didn't regress — and only when the secret was present.
- Whether the missing full-suite CI job is deliberate (cost/secrets) or just unbuilt is an **open question for the owner**.

## 2. The no-mocks philosophy vs reality

The rule of record (`.cursor/rules/testing_rules.mdc`, restated in CLAUDE.md):
> Don't use Mock fixtures. All our tests are integration tests. Do not remove pytest tests without explicit user approval.

Reality, verified: **11 test files import `unittest.mock`/`MagicMock`** (`test_command_executor_delegation, test_execution_context_agent, test_genai_postprocessor{,_integration,_integration_updated,_updated}, test_parameter_extraction_error_regression, test_session_state_serialization, test_turn_result_capture, test_what_can_i_do_command, test_wildcard_inheritance`), and more use `monkeypatch`. `requests-mock` is a declared test dependency (`pyproject.toml:104`) with **zero** usages.

**The de-facto policy** (state this, don't pretend the written rule is practiced literally):
1. **Workflows and framework components are real.** Tests instantiate real `Workflow`/`WorkflowExecutionContext` objects against real golden workflows. Nobody mocks the command directory, routing, or context model.
2. **LLM calls are sometimes stubbed** (patching `dspy.LM`, `fastworkflow.get_env_var`, or agent objects) so tests are cheap, keyless, and deterministic. E.g. `test_genai_postprocessor.py` patches `dspy.LM`; `test_litellm_proxy_routing.py` verifies `get_lm()` resolution "without making actual LLM calls" by writing `fastworkflow._env_vars` directly.
3. **The preferred cost-avoidance pattern is not mocking at all** — it is bypassing the LLM stage: `test_fastapi_turns_async.py` "dispatches a command directly by name, bypassing NLU/intent detection" and controls timing via env hooks (`FW_TEST_ADD_CALL_LOG`, `FW_TEST_ADD_SLEEP_SECONDS`) implemented inside the golden app `tests/hello_world_workflow/application/add_two_numbers.py`, not inside the framework.

Whether the 11 mock-using files are sanctioned exceptions or drift to be cleaned up is **not yours to decide — adjudication belongs to Dhar**. When writing NEW tests, follow the de-facto policy: real workflows always; stub only the LLM boundary, and prefer the bypass pattern over stubs.

## 3. Golden inventory — the artifacts tests depend on

| Golden asset | What it is | Git status | Who depends on it |
|---|---|---|---|
| `tests/hello_world_workflow/` | Minimal add-two-numbers app with test env hooks | 5 files tracked; `___command_info/` JSONs on disk are untracked | `test_fastapi_turns_async.py` (NLU-bypass tests) |
| `tests/example_workflow/` | Todo-list app (TodoItem/TodoList/TodoListManager contexts) | **Gitignored** (`.gitignore:7`); only 6 files tracked; `_commands/` untracked — fresh-clone trap | ~6 runtime test files + `tests/test_build/` fixtures (which build into `tmp_path`) |
| `tests/todo_list_workflow/` | Fuller todo-list workflow, moved from `fastworkflow/examples/todo_list` in commit d679c38 | 35 files tracked. **CLAUDE.md omits it** (doc rot — CLAUDE.md names only example_workflow and hello_world_workflow) | 12 test files (`test_context_model.py`, `test_what_can_i_do_command.py`, `test_session_state_serialization.py`, ...) |
| `fastworkflow/examples/hello_world/___command_info/` | **Pre-trained** intent model for the bundled example | Untracked/gitignored; exists only where someone ran `fastworkflow train` | 4 tests in `test_fastapi_service.py` — they FAIL (not skip) without it |
| `fastworkflow/examples/retail_workflow/` | The tau-bench retail domain workflow | Source tracked; trained artifacts untracked | conftest sys.path injection; retail tests; the entire tau2 reliability program |

Related doc rot, verified: CLAUDE.md also says intent models are trained "via scikit-learn" — actually torch/transformers fine-tuning (`fastworkflow/model_pipeline_training.py` imports `torch`, `transformers`; sklearn supplies only LabelEncoder/split/metrics/PCA). And `testing_rules.mdc` references `tests/README.md`, which does not exist.

### Protecting golden artifacts — the fix-0hb law

Incident (bd `fix-0hb`, fixed in commit fa97b48, 2026-06-15): the training test's module fixture trained against the REAL `fastworkflow/examples/hello_world` and `rmtree`'d its `___command_info` at setup AND teardown. Result: a full suite run passed, then the NEXT run failed 4 FastAPI tests with `FileNotFoundError: ___command_info/global/threshold.json` — and only on machines with real keys, so CI never saw it.

**The law: any test that trains or deletes generated dirs works on a temp COPY, never the shipped workflow.** The canonical pattern, verbatim from `tests/test_train_modern_stack.py:129-140` (introduced by fa97b48):

```python
workflow_path = str(tmp_path_factory.mktemp("train_hello_world") / "hello_world")
shutil.copytree(
    HELLO_WORLD_PATH,
    workflow_path,
    ignore=shutil.ignore_patterns(
        "___command_info",
        "___workflow_contexts",
        "___convo_info",
        "__pycache__",
    ),
)
_cleanup(workflow_path, env_vars)   # now safe: operates on the copy
```

Warning that follows directly: the two permanently-skipped tests in `tests/test_workflow_training.py` still call `_cleanup_generated_files()` against the REAL `fastworkflow/examples/retail_workflow` and the real installed package. **If you ever un-skip them, port the temp-copy pattern first** or you re-create fix-0hb against retail.

## 4. Known-flaky and no-op tests — do not trust these signals

| Test | Problem | Status |
|---|---|---|
| `test_train_modern_stack.py::test_trained_model_routes_utterance` | Trains a fresh model from LLM-generated synthetic utterances each run, then asserts "add 2 and 3" routes to `add_two_numbers`. Nondeterministic by construction (LLM syndata + training); reported flaky in the 2026-07 discovery review, historically handled by re-running until green | **Open** — retry-until-green vs deflaking is unadjudicated; ask the owner before "fixing" |
| `tests/test_streaming_endpoint.py` | A manual script wearing a pytest name: POSTs to hardcoded `http://localhost:8000` (line 11) with a workflow path `fastworkflow/examples/todo_list` (line 44) that has been **dead since commit d679c38** moved it to `tests/todo_list_workflow`. On any exception it prints and `return`s (lines 54-56, 113-117) — under pytest with no server it **silently passes**. Zero signal, permanently green | Known no-op. Do not cite it as evidence. Converting it to skip-or-assert (or moving it out of `tests/`) requires owner approval — never delete a test unilaterally |
| `test_workflow_training.py` (2 tests, lines 74 and 137) | `@pytest.mark.skip("Skipping because it takes too long.")` — consequently **zero automated coverage of retail_workflow / package training**. Only hello_world training is exercised, and only when keys are present | Open; also carries the un-skip hazard above |
| `test_ambiguous_intent_regression.py` (4 tests) | Asserts on its own local dict simulation of the fix; never imports the code path it claims to protect. Permanently green regardless of regressions | Known no-op signal |
| `test_command_routing.py:36-38` | `except Exception: contexts = {"*": []}  # Fallback to ensure tests don't fail` — deliberately weakened assertions | Weak signal; treat passes as low-value |
| Skip inventory | **11 unconditional `@pytest.mark.skip` markers** total (list + reasons in [references/test-suite-inventory.md](references/test-suite-inventory.md)) | — |

Rule of thumb: a green suite means "the ~375 unguarded tests plus whatever your gates admit passed once." It does not mean the training path works, the streaming endpoint works, or retail training works.

## 5. How to add tests

Checklist (all items verified against existing practice):

1. **Integration-first, real workflows.** Point at `tests/hello_world_workflow`, `tests/example_workflow`, `tests/todo_list_workflow`, or `fastworkflow/examples/retail_workflow`. No mock fixtures for framework components (routing, command directory, context model, Workflow/WEC).
2. **Stub only the LLM boundary, and prefer not stubbing at all.** First choice: the NLU-bypass pattern (dispatch a command by name; put test hooks in the golden app's `application/` code, as `FW_TEST_ADD_CALL_LOG` does). Second choice: patch `dspy.LM` / write `fastworkflow._env_vars` like `test_genai_postprocessor.py` and `test_litellm_proxy_routing.py` do.
3. **Gate, don't fail.** If your test needs keys/env files or trained models, guard with `pytest.skip` exactly like `test_fastapi_service.py:27-36`, so the suite stays green on keyless machines and CI.
4. **Protect golden artifacts.** Anything that trains, writes, or deletes inside a workflow dir: temp-copy pattern (section 3). Excluding `___command_info`, `___workflow_contexts`, `___convo_info`, `__pycache__` in the copytree is part of the pattern.
5. **Isolate global state.** Use the `add_temp_workflow_path` fixture from `tests/conftest.py` for sys.path; use unique user ids (`test_user_{uuid}` pattern in test_fastapi_service.py); call `RoutingRegistry.clear_registry()` if you touch routing caches.
6. **Never remove or permanently-skip an existing test without the developer'sexplicit approval** (`.cursor/rules/testing_rules.mdc`). This includes "cleaning up" the no-op tests in section 4.
7. **Run the affected files, then the full suite** before claiming done: `python -m pytest tests/test_<yours>.py -q` then `python -m pytest`.

## 6. The acceptance-evidence bar

### For ordinary code changes (de-facto, from practice)

- Full local suite on a fully provisioned box (both env files + pre-trained hello_world): the recorded standard is the fa97b48 commit message itself — "Verified: full suite 478 passed / 0 failed ... suite is now idempotent." Match that: **0 failures, and a SECOND consecutive full run also 0 failures** (idempotence is part of the bar because fix-0hb was invisible in single runs). At ~50 minutes a run, two runs is most of a two-hour block — budget it, and note that **neither** the v3.1.0 nor the v3.1.1 release met this bar (v3.1.0: one run; v3.1.1: two runs, but on adjacent trees, which measures wall time rather than idempotence — see the baseline row above). A cheaper partial substitute for the idempotence half, used at v3.1.0, is to checksum `fastworkflow/examples/*/___command_info` either side of a single run.
- CI green is necessary but nearly meaningless (section 1). Do not present it as evidence.
- No formal written release-gate policy exists in the repo (**open** — the makefile `publish` target runs no tests). Until the owner writes one, treat "full suite green twice locally + affected-area tests run in verbose" as the minimum before any release tag.

### For reliability/benchmark claims (tau2 retail program)

Local pytest is NOT evidence of reliability. Defer to the campaign's gates (see `tau2-reliability-campaign`; math and stats recipes in `fastworkflow-proof-and-analysis-toolkit`; lifecycle in `fastworkflow-research-methodology`). The bar, from the internal plan (team-private, §3.3/§6):

- No measured claim before the **E0 harness** exists: k≥5 runs per configuration, temperature 0 + fixed seeds where controllable, Clopper-Pearson/Wilson CIs, McNemar paired tests, blind dual-rater failure attribution with Cohen's κ, pre-registration written **before** the run.
- Staged gates on the 15-task hard set: Gate 1 = 15/15 pass@1 (k=5, report all runs); Gate 2 = ≥13/15 pass^3 (the honest primary endpoint); Gate 3 = 15/15 pass^3, **pre-registered, single-shot, on a frozen stack, attempted once**.
- Anti-p-hacking rule: the 15 tasks are a months-old tuning set; rerun-and-patch until a sweep lands is test-set optimization. Always pair hard-set numbers with the full ~115-task retail split.

### Non-negotiables that bind all QA work (confirmed discipline rules, 2026-07-09)

1. **Never `git commit`/`push` without the developer'sexplicit request in that turn** (established 2026-07-08 after a private doc was auto-pushed to the public repo and history had to be rewritten). Running tests never authorizes committing them.
2. **tau-bench parity is sacred**: never modify tau-bench tools/tasks for benchmark runs; any nonstandard trade (e.g. pinning simulator temperature) must be disclosed in the report, never silent.
3. **One bd write at a time; verify `.beads/issues.jsonl` changed after each.** (`bd close --reason` silently failed to persist on an older bd, 2026-06-11; it persists on 1.1.2, retested 2026-08-13.) QA bookkeeping follows this too.
4. **Never let tests wipe trained example models** — section 3.

## Provenance and maintenance

Facts verified 2026-07-09 against v2.22.2 (commit c33b9a5), on the owner's provisioned WSL2 box (both env files present, hello_world pre-trained); test counts and the full-suite baseline re-measured 2026-08-20 at v3.1.1 (previously 2026-08-13 at v3.1.0). Spot-run evidence: `pytest --collect-only -q` → 1860 collected in 9.4 s (1803 at v3.1.0, 495 at v2.22.2); `pytest tests/test_context_model.py tests/test_command_directory.py -q` → 9 passed, 1 skipped.

**This skill owns the suite's numbers.** `fastworkflow-change-control`, `fastworkflow-build-and-env` and `fastworkflow-debugging-playbook` used to quote the test count and wall time in passing; those mentions were removed on 2026-08-20 and now point here, so a stale figure has only one place to hide.

Re-verify volatile facts:

| Fact | Command |
|---|---|
| Collected-test count (1860 at v3.1.1) | `source .venv/bin/activate && python -m pytest --collect-only -q \| tail -1` |
| No pytest config | `ls pytest.ini setup.cfg tox.ini 2>&1; grep -n '\[tool.pytest' pyproject.toml` |
| 0.5 s sleep fixture lines | `grep -n 'time.sleep(0.5)' tests/conftest.py` |
| Env-gate guards | `grep -n 'pytest.skip' tests/test_fastapi_service.py tests/test_train_modern_stack.py` |
| CI = one workflow, one test file | `ls .github/workflows/; grep -n 'pytest' .github/workflows/train-modern-stack.yml` |
| Mock-importing files (11) | `grep -rln 'unittest.mock\|MagicMock' tests --include='*.py' \| wc -l` |
| Unconditional skips (11) | `grep -rn 'pytest.mark.skip' tests -r --include='*.py' \| grep -v skipif` |
| example_workflow gitignored | `grep -n 'example_workflow' .gitignore; git ls-files tests/example_workflow` |
| todo_list_workflow tracked (35 files) | `git ls-files tests/todo_list_workflow \| wc -l` |
| hello_world trained artifact | `ls fastworkflow/examples/hello_world/___command_info/global/threshold.json` |
| Streaming test dead path | `sed -n '44p' tests/test_streaming_endpoint.py; ls fastworkflow/examples/ \| grep todo_list` |
| fix-0hb narrative + train recipe | `bd show fix-0hb` (read-only) |
| Temp-copy pattern source | `git show fa97b48 -- tests/test_train_modern_stack.py` |
| requests-mock unused | `grep -n 'requests-mock' pyproject.toml; grep -rn 'requests_mock' tests --include='*.py'` |
| Trainer is torch/transformers not sklearn | `grep -n 'import torch\|from transformers' fastworkflow/model_pipeline_training.py \| head -3` |
| Environment gates on THIS machine | `bash .claude/skills/fastworkflow-validation-and-qa/scripts/qa_env_report.sh` |

Full per-file inventory (skips, mocks, gating, golden-asset map): [references/test-suite-inventory.md](references/test-suite-inventory.md).
