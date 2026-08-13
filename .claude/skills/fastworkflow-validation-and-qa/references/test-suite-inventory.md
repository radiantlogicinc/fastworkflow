# fastWorkflow test-suite inventory (detail overflow from SKILL.md)

> TEAM-PRIVATE: embeds content from uncommitted internal docs. Do not commit or publish this skill without the developer'sexplicit approval.

Facts verified 2026-07-09 against v2.22.2 (commit c33b9a5); suite size and the full-run baseline re-measured 2026-08-13 against v3.1.0. Re-verification commands at the bottom of SKILL.md.

> The per-file breakdown below is the v2.22.2 shape and has NOT been re-inventoried — the suite has since roughly tripled. Trust the totals in SKILL.md; treat the file-by-file counts here as historical until someone re-walks them.

## 1. Suite shape

- 1803 tests collected as of 2026-08-13 / v3.1.0. At v2.22.2 it was 495 across 79 `test_*.py` files (verified `find tests -name 'test_*.py' -not -path '*__pycache__*'`): 50 top-level in `tests/`, 25 under `tests/test_build/` (pure-Python AST/codegen — no keys, no network), 4 under `tests/test_simple_workflow_template/`.
- Phase coverage: build-time = `tests/test_build/`; train-time = `tests/test_train_modern_stack.py` (2 tests, key-gated) + `tests/test_workflow_training.py` (2 tests, permanently skipped); run-time = everything else (~375 tests).
- `tests/conftest.py`:
  - lines 33-58: session autouse fixture injects `fastworkflow/examples/{retail_workflow, simple_workflow_template, hello_world}` into `sys.path` ("simulating what Workflow does in production").
  - lines 61-86: `add_temp_workflow_path` opt-in fixture for tmp-dir workflows.
  - lines 89-98: auto-marks node ids containing "integration" as `integration`, "mcp_server" as `slow`. No current filename matches `mcp_server` (only `test_mcp_token_generation.py` exists) — the slow rule is dead code.
  - lines 111-118: function autouse fixture, `time.sleep(0.5)` after every test (ChatWorker daemon-thread drain). ~4 min fixed cost per full run.
- `os.environ.setdefault("PYTEST_RUNNING", "1")` at conftest.py:16 — some framework paths branch on it.

## 2. Unconditional skip inventory (11 markers, verified)

| File:line | Reason string | Coverage consequence |
|---|---|---|
| `tests/test_workflow_training.py:74` | "Skipping because it takes too long." | No automated coverage of training the fastworkflow package/CME workflow |
| `tests/test_workflow_training.py:137` | "Skipping because it takes too long." | **No automated coverage of retail_workflow training at all** |
| `tests/test_react_available_commands.py:99` | "Requires real API key and makes actual API calls" | ReAct end-to-end command injection untested (also gated on `OPENAI_API_KEY` at line 109) |
| `tests/test_build/test_command_stub_generator.py:133,138,143,170` | "Container context handlers no longer generated" | Dead-feature tests kept per no-removal rule |
| `tests/test_build/test_navigator_stub_generator.py:136,259` | "Container context navigation no longer supported" | same |
| `tests/test_build/test_class_analysis_structures.py:106,119` | "Skipping import statement validation as per user request" | import validation of analyzed classes unchecked |

Un-skip hazard (repeat of SKILL.md warning): `test_workflow_training.py`'s `_cleanup_generated_files()` rmtree's `___command_info` of the REAL `fastworkflow/examples/retail_workflow` and the installed package. Port the fa97b48 temp-copy pattern before ever un-skipping.

Conditional skips (env-dependent, not permanent): cyclic-workflow fixture (`test_context_model.py:98`), `extended_workflow_example` presence (`test_workflow_inheritance.py:214`), CME method availability (`test_command_executor_errors.py:53`).

## 3. Mock usage census (the "no mocks" rule vs practice)

The rule (`.cursor/rules/testing_rules.mdc`, alwaysApply): "Don't use Mock fixtures. All our tests are integration tests." It also references `tests/README.md`, which does not exist (doc rot).

11 files import `unittest.mock` / `MagicMock` (verified `grep -rln`):

| File | What is stubbed | Workflow still real? |
|---|---|---|
| test_genai_postprocessor.py | `fastworkflow.get_env_var`, `dspy.LM`, `dspy.context` | yes |
| test_genai_postprocessor_integration.py | LLM boundary | yes |
| test_genai_postprocessor_integration_updated.py | LLM boundary | yes |
| test_genai_postprocessor_updated.py | LLM boundary | yes |
| test_execution_context_agent.py | agent objects (MagicMock at ~line 68) | yes |
| test_command_executor_delegation.py | delegation collaborators | yes |
| test_parameter_extraction_error_regression.py | LLM/extraction boundary | yes |
| test_session_state_serialization.py | selective internals | yes |
| test_turn_result_capture.py | selective internals | yes |
| test_what_can_i_do_command.py | selective internals | yes (todo_list_workflow) |
| test_wildcard_inheritance.py | selective internals | yes (example_workflow) |

Also: `monkeypatch` in `test_command_executor_errors.py`, `test_fastapi_turns_async.py` (env vars, not components). `requests-mock` declared at `pyproject.toml:104`, imported by zero tests (leftover or planned — unknown).

De-facto policy (as stated in SKILL.md §2): real workflows and framework components always; LLM boundary may be stubbed; NLU-bypass preferred over stubbing. Whether the 11 files are sanctioned exceptions is for Dhar to adjudicate — do not "clean them up" or cite the written rule to reject a PR that follows de-facto practice without asking.

## 4. Golden-asset dependency map (expanded)

### tests/hello_world_workflow/
Tracked (5 files): `__init__.py`, `_commands/{README.md, add_two_numbers.py, context_inheritance_model.json}`, `application/add_two_numbers.py`.
On-disk untracked: `___command_info/` (`command_directory.json`, `routing_definition.json`, `command_routing_definition.json`, `global/`), `___workflow_contexts/`, `___convo_info/`.
Special: `application/add_two_numbers.py` implements the test hooks `FW_TEST_ADD_CALL_LOG` (appends one line per invocation to the named file — lets tests assert "executed exactly once") and `FW_TEST_ADD_SLEEP_SECONDS` (simulates a long LLM turn). This is the sanctioned way to make golden apps testable: hooks live in the app, not the framework.

### tests/example_workflow/
`.gitignore:7` ignores the whole directory; only 6 files are tracked (`__init__.py`, `application/{__init__,todo_item,todo_list,todo_manager}.py`, `startup_action.json`). The `_commands/` tree (TodoItem/TodoList/TodoListManager + `context_inheritance_model.json`) exists on provisioned machines but is NOT in git.
Fresh-clone consequence: `test_command_router.py`, `test_command_routing.py`, `test_python_utils.py`, `test_what_can_i_do_command.py`, `test_what_is_current_context_command.py`, `test_wildcard_inheritance.py` reference it outside tmp-path builds and will misbehave. (`tests/test_build/` is safe — its conftest builds into `tmp_path`.)
Also contains committed-era runtime droppings on disk (`___workflow_contexts` RocksDB files) — repo hygiene inconsistency, known.

### tests/todo_list_workflow/
35 tracked files: `_commands/{TodoItem,TodoList,TodoListManager}` + `context_inheritance_model.json` + `startup.py`, `application/`, `startup_action.json`, top-level `context_hierarchy_model.json`. Moved from `fastworkflow/examples/todo_list` by commit d679c38 ("Moved todo_list workflow from the examples folder to the tests folder..."). Used by 12 test files. **CLAUDE.md's testing section omits it** — trust this inventory over CLAUDE.md.
Its on-disk `___command_info/` holds only the two JSONs (no trained BERT models) — tests using it do not need training; the JSONs rebuild cheaply.

### fastworkflow/examples/hello_world/___command_info/ (trained)
Gitignored. Contains `global/{tinymodel.pth, largemodel.pth, threshold.json, tiny_ambiguous_threshold.json, large_ambiguous_threshold.json, label_encoder.pkl}` + directory JSONs after training. Required (FAIL, not skip) by 4 tests in `test_fastapi_service.py`: `test_initialize_with_startup_command_in_request`, `test_invoke_assistant_endpoint`, `test_session_not_found_errors`, `test_concurrent_request_handling`.
Restore recipe (from bd fix-0hb): `fastworkflow train ./fastworkflow/examples/hello_world ./env/.env ./passwords/.env`. The bundled `fastworkflow/examples/fastworkflow.passwords.env` carries a placeholder key that yields `litellm AuthenticationError` — only repo-local `passwords/.env` works. `test_train_modern_stack.py:45-47` codifies placeholder rejection (`_looks_like_real_key` rejects values containing `<` or `your-`).

### fastworkflow/examples/retail_workflow/
The tau-bench retail domain. conftest injects it into sys.path for the whole session. Its trained artifacts (when present) are untracked. It is the substrate of the tau2 reliability program — treat it as production-grade golden: never modify its tools/tasks for benchmark runs (tau-bench parity rule), never let tests write into it.

## 5. Flaky / no-op detail

### test_trained_model_routes_utterance (tests/test_train_modern_stack.py:177-197)
Chain of nondeterminism: LLM generates synthetic utterances (litellm, real key) → BERT two-tier training on that data → `CommandRouter(model_dir).predict("add 2 and 3")` must contain `add_two_numbers`. Any run's synthetic corpus can produce a model that misroutes the probe utterance. The loop over model dirs (`routed = True; break`) passes if ANY context routes it. Observed handling: re-run until green. Policy question (retry-until-green accepted, or deflake via fixed syndata corpus / seed?) is open — recorded in the 2026-07 discovery review. Any deflaking touches training behavior: route through `fastworkflow-change-control`.

### tests/test_streaming_endpoint.py (dead manual script)
- Line 11: `BASE_URL = "http://localhost:8000"` — assumes a manually started server.
- Line 44: `"workflow_path": "/home/drawal/rl/fastworkflow/fastworkflow/examples/todo_list"` — absolute path to a directory removed by d679c38; wrong on every machine including the owner's.
- Lines 50-56 and 113-117: `except Exception: print(...); return` — pytest sees a function that returns `None` ⇒ pass.
- Has an `if __name__ == "__main__"` block: it was written as a manual smoke script. Under pytest it is a permanently green no-op.
- Correct future states (owner's call): convert to skip-if-no-server integration test, or move out of `tests/`. Do not delete without approval.

### test_ambiguous_intent_regression.py
All four tests build a local `workflow_context = {}` dict and assert on their own `if "command" not in workflow_context` simulation. No fastworkflow code under test is imported besides names never exercised. A regression in the real ambiguous-intent code would not fail these tests.

### test_command_routing.py weak fallbacks
Lines 36-38: `except Exception: contexts = {"*": []}  # Fallback to ensure tests don't fail`; lines 134-136 and 144-146 swallow KeyError/Exception around utterance checks. Passes are weak evidence for routing-definition correctness.

## 6. CI workflow annotated (.github/workflows/train-modern-stack.yml)

- Triggers: push to main, pull_request, workflow_dispatch. Single job, ubuntu-latest, Python 3.11, timeout 30 min.
- Installs `pip install ".[training]"` then pins `transformers>=5.0.0,<6.0.0`, `dspy>=3.0.1,<4.0.0`, `openai>=2.8.0` — deliberately newer than local floors, to prove modern-stack compatibility.
- Env: `LLM_SYNDATA_GEN` (repo var, default `mistral/mistral-small-latest`), `LITELLM_API_KEY_SYNDATA_GEN` (secret).
- Runs `python -m pytest tests/test_train_modern_stack.py -v` and nothing else.
- Header comment states the skip-clean behavior: "without them the regression test skips cleanly (the job still passes)".
- Note the CI/local asymmetry that hid fix-0hb: the destructive fixture only activated where keys existed (local), so CI stayed green while local suites self-destructed. When you add key-gated tests, ask "what does this do on the keyed machine that the keyless machine will never see?"

## 7. Release-evidence ledger (what has actually been recorded)

| Date | Evidence recorded | Where |
|---|---|---|
| 2026-06-15 | "full suite 478 passed / 0 failed", suite idempotent | commit fa97b48 message; bd fix-0hb close reason |
| 2026-01-28 | 22/22 tests passing after CVE sweep (subset run) | SECURITY_VULNERABILITY_REPORT.md |
| 2026-07-09 | 495 collected; spot run 9 passed / 1 skipped (test_context_model + test_command_directory) | this skill's authoring session |
| 2026-08-13 | 1803 collected; **1788 passed / 15 skipped / 0 failed / 0 errors in 48:56**; `___command_info` checksummed either side — 10 build-cache JSONs rewritten, no model weights or thresholds touched, nothing deleted | v3.1.0 release verification (bd fix-dzs) |

Unknowns to close (ask owner / measure): whether full-suite-before-tag is actually practiced for every release; monthly cost of a fully-keyed run. **Closed 2026-08-13:** full-suite wall time (~49 min) and the current full-run pass count (1788/15/0 at 1803 collected) — see the row above. Still open on that run: it was a SINGLE run, so the two-consecutive-clean-runs idempotence bar in SKILL.md was not met.
