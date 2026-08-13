---
name: fastworkflow-build-and-env
description: >
  Load this skill when setting up the fastWorkflow dev environment from scratch or fixing a
  broken one: fresh clone, "which Python / poetry install / pip extras do I need",
  "tests are all skipping", "FileNotFoundError threshold.json", "make gen-env fails",
  "where do API keys go", "env/.env vs passwords/.env vs fastworkflow.env", or re-provisioning
  after the hello_world model was wiped. Do NOT load it for running/operating workflows
  (fastworkflow-run-and-operate), the full env-var catalog (fastworkflow-config-and-flags),
  test-writing policy (fastworkflow-validation-and-qa), or debugging runtime failures
  (fastworkflow-debugging-playbook).
---

# fastWorkflow: build the dev environment from scratch

> TEAM-PRIVATE: embeds content from uncommitted internal docs. Do not commit or publish this skill without the developer'sexplicit approval.

Runbook to go from `git clone` to a machine that can run the test suite, train the
examples, and serve the FastAPI/MCP server. Every command below was executed or
source-verified on 2026-07-09 against v2.22.2 (commit c33b9a5). Repo root is assumed to be
`~/rl/fastworkflow` in examples; substitute your own absolute path.

Jargon used once, defined once:

- **workflow** — a directory of `_commands/*.py` files fastWorkflow can train and run
  (e.g. `fastworkflow/examples/hello_world`).
- **`___command_info/`** — the trained-model artifact directory a `fastworkflow train` run
  writes inside a workflow (BERT intent classifiers + JSON snapshots). Gitignored, excluded
  from the wheel, expensive to regenerate.
- **CME** — `command_metadata_extraction`, fastWorkflow's own internal workflow at
  `fastworkflow/_workflows/command_metadata_extraction/`; it too needs trained models.
- **litellm** — the multi-provider LLM client library; every model is named
  `<provider>/<model>` (e.g. `mistral/mistral-small-latest`) with a per-role API key.

## When to use / when NOT to use

| Situation | Skill |
|---|---|
| Fresh clone, broken venv, missing secrets, all-skipping tests, missing trained example | THIS skill |
| Run the CLI / examples / FastAPI+MCP server, artifact locations | `fastworkflow-run-and-operate` |
| What a specific env var or CLI flag does; adding a new one | `fastworkflow-config-and-flags` |
| Test-suite policy, what counts as evidence, golden inventory | `fastworkflow-validation-and-qa` |
| A runtime/build/train failure on an already-working machine | `fastworkflow-debugging-playbook` |
| Commit/push rules, change gating, incident rationale | `fastworkflow-change-control` |
| NLU stack internals (BERT tiers, DSPy, thresholds) | `fastworkflow-nlu-pipeline-reference` |

## 0. The from-scratch sequence (copy-paste block)

```sh
# --- 1. Clone and create the venv (Python >=3.11,<3.14; team standard is 3.12.x) ---
git clone https://github.com/radiantlogicinc/fastworkflow.git
cd fastworkflow
python3.12 -m venv .venv
source .venv/bin/activate

# --- 2. Editable install with all dev/test deps and both extras ---
pip install --upgrade pip
poetry install --all-extras          # poetry 1.8.x; uses the existing .venv
                                     # (dev+test groups are on by default; add --with aws for Bedrock)

# --- 3. Provision secrets (see section 3; NEVER put keys in env/.env — it is git-tracked) ---
mkdir -p passwords
cp fastworkflow/examples/fastworkflow.passwords.env passwords/.env
"${EDITOR:-nano}" passwords/.env     # replace every <...> placeholder with a real key
                                     # one free Mistral key can fill every LITELLM_API_KEY_* slot

# --- 4. Sanity: collection should report ~1803 tests (v3.1.0), ~15 s ---
python -m pytest --collect-only -q | tail -1

# --- 5. The golden artifact: train hello_world ONCE (needs the real key from step 3) ---
fastworkflow train ./fastworkflow/examples/hello_world ./env/.env ./passwords/.env

# --- 6. Full local suite (no makefile target for this; CI does NOT run it for you) ---
python -m pytest
```

Expected outcomes on a fully provisioned box: step 4 prints `1803 tests collected in ~14s`
(measured 13.94 s on the reference WSL2 box); step 6 baseline is **478 passed / 0 failed**
(fix-0hb verification run, 2026-06-15; the remainder are deliberate skips). Full-suite wall
time is not recorded anywhere — expect tens of minutes (there is a fixed ~4 min of per-test
0.5 s thread-drain sleeps alone). Without secrets you will see mass skips, not failures —
see section 6.

Doctor script for any existing machine:

```sh
bash .claude/skills/fastworkflow-build-and-env/scripts/check_env.sh
```

## 1. Prerequisites

| Requirement | Ground truth | Evidence |
|---|---|---|
| OS | Linux or macOS; on Windows use WSL (reference dev box is WSL2) | README.md:139; `uname -r` → `...-microsoft-standard-WSL2` |
| Python | `>=3.11,<3.14` (package constraint). Local convention: 3.12.x in-project `.venv` | pyproject.toml `python = ">=3.11,<3.14"`; `.venv/pyvenv.cfg` → 3.12.2 |
| Poetry | 1.8.x generated poetry.lock; `virtualenvs.in-project=true` and `virtualenvs.create=true` in the owner's user config | poetry.lock header; `poetry config virtualenvs.in-project` |
| Venv activation | **Always** `source .venv/bin/activate` before tests or scripts — this is a repo rule | AGENTS.md line 3 |
| GPU | Not required. Training runs on CPU; runtime inference is CPU milliseconds | README.md "no GPU at runtime" note |

The reference `.venv` was created with plain `python3 -m venv .venv` (from a miniconda
3.12.2 base — see `.venv/pyvenv.cfg`), then populated by `poetry install`. Either
`python -m venv` + poetry, or letting poetry create the in-project venv itself, works;
what matters is that the venv lives at `<repo>/.venv` (tests, `make audit`, and AGENTS.md
all assume that path).

## 2. Install paths

### 2a. Repo developer (this is you)

```sh
source .venv/bin/activate
poetry install --all-extras            # main + dev + test groups, extras: training, server
# poetry install --all-extras --with aws   # add boto3 if using AWS Bedrock models
```

- `poetry install` performs an **editable** install (a `fastworkflow.pth` pointing at the
  repo). Code changes are live immediately.
- Foot-gun: the dist-info metadata is stamped at install time and goes stale — on the
  reference box `importlib.metadata.version('fastworkflow')` returns `2.21.5` while the
  code is 2.22.2 (pyproject.toml). Trust `pyproject.toml`, never `pip show fastworkflow`,
  for the working version.
- Dependency groups: `dev` (isort, black, flake8, pylint, mypy, bandit) and `test`
  (pytest, pytest-cov, requests-mock) are **non-optional** — plain `poetry install`
  includes them. Only `aws` (boto3) is optional.

### 2b. Package user / downstream app (pip)

```sh
pip install fastworkflow                 # core: build + run, plain litellm client
pip install "fastworkflow[training]"     # + datasets (required for `fastworkflow train` to
                                         #   actually train models; without it train only
                                         #   regenerates the JSON snapshots)
pip install "fastworkflow[server]"       # + uvicorn/fastapi/fastapi-mcp/pyjwt/... for
                                         #   `fastworkflow run_fastapi_mcp`
```

Exact extras defined in pyproject.toml `[tool.poetry.extras]`: `training`, `server`, and
`fastapi` (a backward-compatible alias of `server`, pre-2.18.2 name). There is no other
extra — anything else in older docs is rot.

Pip-user consequence of packaging: the wheel **excludes** `**/___command_info` and
`**/___workflow_contexts` (pyproject.toml `exclude`). So a pip install ships **no trained
models at all**, including for the internal CME workflow — the first
`fastworkflow train <your-workflow>` trains CME first (several extra minutes). See the
CME trap in section 6.

## 3. Secret provisioning — read this twice

There are THREE distinct env-file conventions. Confusing them is the #1 setup failure.

| File(s) | Purpose | Git status | Contains keys? |
|---|---|---|---|
| `<workflow>/fastworkflow.env` + `<workflow>/fastworkflow.passwords.env` | Per-workflow config the CLI loads (train/run/run_fastapi_mcp default to these paths inside the workflow dir) | Only the bundled templates under `fastworkflow/examples/` are tracked | passwords file: yes (yours) |
| `env/.env` + `passwords/.env` (repo-local, for tests and example training) | What the key-gated tests and the golden-artifact train command read | `env/.env` **IS TRACKED IN GIT** (public!); `passwords/.env` is gitignored (`.gitignore` line 1: `passwords/*`) | `env/.env`: **NEVER**. `passwords/.env`: yes |
| root `.env` | Generated aggregate written by `gen-env.sh` (used by `make publish*` for PyPI tokens) | gitignored (`/.env`) | yes — it's a plaintext merge of everything, including `passwords/*.env` |

### The loud warning

`env/.env` is a **git-tracked, public** file (verify: `git ls-files env/.env`). It holds
model strings and framework settings only. If you paste an API key into it, the next
commit publishes that key to GitHub. Keys go **only** into `passwords/.env`
(gitignored) or a per-workflow `fastworkflow.passwords.env` **outside** tracked paths.
Related discipline rule (owner, confirmed 2026-07-08 after a private doc was auto-pushed
to the public repo and required a history rewrite): **never `git commit` or `git push`
unless Dhar explicitly asked for it in that turn.** See `fastworkflow-change-control`.

### gen-env.sh mechanics (verified by reading the script)

`./gen-env.sh` (invoked by `make gen-env`) finds **every `*.env` file in the tree**
(excluding `./.env` itself and `./override/*`, plus any `--exclude dir1,dir2` paths),
concatenates their key=value pairs, processes `override/*.env` last so overrides win,
deduplicates last-occurrence-wins, sorts, and writes the result to the **root `.env`**
with a "Do not edit this file. It is generated!" header. Consequences:

- Root `.env` ends up holding your real LLM keys AND the PyPI publish tokens from
  `passwords/pypi.env` in one plaintext file. It is gitignored, but treat it like a
  password file (don't cat it into logs, transcripts, or LLM context).
- Any stray `*.env` you drop anywhere in the tree silently joins the merge on the next
  `make publish` (which depends on `gen-env`). Name scratch env files `*.env.local` or
  keep them under `override/` deliberately.
- `env/.env` currently contains two duplicate `LLM_*` blocks (groq/gpt-oss-20b, then
  mistral/mistral-small-latest); dotenv and gen-env are both last-wins, so **mistral is
  the effective local model** for every role.

### Minimum viable key set (Mistral free tier)

One free Mistral API key (mistral.ai; README also suggests OpenRouter's free
gpt-oss-20b) pasted into **every** `LITELLM_API_KEY_*` line works for all roles. Minimums
by activity, with the model var each key pairs with:

| Activity | Required key(s) | Notes |
|---|---|---|
| `fastworkflow train` | `LITELLM_API_KEY_SYNDATA_GEN` | Missing key = warning then failed/empty synthetic generation; train also hard-fails without `SPEEDDICT_FOLDERNAME` in the env file |
| `fastworkflow run` (agent mode) | `LITELLM_API_KEY_AGENT`, `LITELLM_API_KEY_PARAM_EXTRACTION`, `LITELLM_API_KEY_PLANNER` | run also probes `LITELLM_API_KEY_SYNDATA_GEN` presence at startup (run/__main__.py:129-132) — keep it set |
| `fastworkflow refine` | `LLM_COMMAND_METADATA_GEN` + `LITELLM_API_KEY_COMMANDMETADATA_GEN` **exported in the OS shell** | refine/build load NO env files (`fastworkflow.init(env_vars={})`); note the spelling: no underscore in `COMMANDMETADATA`; this var pair is absent from the bundled template |
| FastAPI conversation summaries | `LITELLM_API_KEY_CONVERSATION_STORE` | |

Placeholder hygiene: the bundled `fastworkflow/examples/fastworkflow.passwords.env` ships
`<API KEY ...>` placeholders. When running from the repo root, importing fastworkflow
pulls them into `os.environ` — not fastworkflow's doing, but litellm's import-time
`load_dotenv()` over the CWD `.env`, which at the repo root is the gen-env merge and can
carry the template placeholders (verified 2026-07-09: repo-root import added 7
`LITELLM_API_KEY_*` vars, 6 placeholder-valued; importing from a directory with no `.env`
added none). The test suite treats any value containing `<` or `your-` as absent
(`tests/test_train_modern_stack.py` `_looks_like_real_key`). A "key" that still contains
`<` produces `litellm AuthenticationError: Unauthorized` — that exact symptom means you
edited the wrong file or didn't edit at all (fix-0hb post-mortem, `.beads/issues.jsonl`).

Full env-file semantics (loading order, the default-shadows-OS-environment quirk, per-var
catalog): see [references/secrets-and-env-files.md](references/secrets-and-env-files.md)
and the `fastworkflow-config-and-flags` sibling.

## 4. The golden artifact: trained hello_world

Four tests in `tests/test_fastapi_service.py` — `test_initialize_with_startup_command_in_request`,
`test_invoke_assistant_endpoint`, `test_session_not_found_errors`,
`test_concurrent_request_handling` — **silently require** a locally pre-trained
`fastworkflow/examples/hello_world/___command_info/`. Nothing trains it for you; it is
gitignored and wheel-excluded, so a fresh clone does not have it.

Produce it (from repo root, venv active, real key in `passwords/.env`):

```sh
fastworkflow train ./fastworkflow/examples/hello_world ./env/.env ./passwords/.env
```

- This exact command is the blessed recipe (recorded in bead fix-0hb and
  `.vscode/launch.json`'s `train_hello_world` config).
- Because the path contains the substring `fastworkflow`, the internal CME workflow is
  NOT retrained first (`train/__main__.py` `train_main` guard) — you only pay for
  hello_world itself.
- Duration: README claims "~5 min on CPU" for the fetched copy of this example; CI allots
  30 min for the whole train-and-assert job. Not re-measured on 2026-07-09 (training
  writes artifacts and spends LLM quota — deliberately not run while authoring this).
- Success check: `ls fastworkflow/examples/hello_world/___command_info/global/threshold.json`
  exists. Full contents when healthy: `command_directory.json`, `routing_definition.json`,
  `add_two_numbers_param_labeled.json`, and `global/` holding `tinymodel.pth/`,
  `largemodel.pth/`, `label_encoder.pkl`, `threshold.json`, `tiny_ambiguous_threshold.json`,
  `large_ambiguous_threshold.json`. Disk cost: ~276 MB (measured).

### The fix-0hb rule (non-negotiable)

Incident fix-0hb (fa97b48): a training-test fixture wiped this model, poisoning the next
suite run — full story in `fastworkflow-failure-archaeology` entry 5. The standing rule:

> **Never let a test, experiment, or script train into or delete
> `fastworkflow/examples/*/___command_info`.** Always
> `shutil.copytree(example, tmp_path, ignore=ignore_patterns('___command_info',
> '___workflow_contexts', '___convo_info', '__pycache__'))` and train the copy.

If you see the threshold.json FileNotFoundError today: something wiped the model —
re-run the train command above, then find and fix the wiper (see
`fastworkflow-debugging-playbook`).

## 5. Makefile targets (each verified 2026-07-09)

There is **no `make test`**. Tests are always `source .venv/bin/activate && python -m pytest`.

| Target | What it does | Verified sharp edges |
|---|---|---|
| `gen-env` | `chmod +x ./gen-env.sh && ./gen-env.sh` → writes root `.env` | **Chicken-and-egg on fresh clones**: makefile line 10 is a parse-time `include ./.env`, so if root `.env` doesn't exist yet, EVERY make target — including `make gen-env` itself — dies with `makefile:10: .env: No such file or directory / No rule to make target '.env'` (reproduced empirically). Bootstrap by running `./gen-env.sh` directly once. |
| `lint` | py3clean, isort, black, flake8 (ignores E501,E122,W503,E402,F401), pylint, mypy twice, bandit | **MUTATES the working tree** (isort + black reformat in place). AI agents: do not run casually on someone else's dirty tree. `bandit -c pyproject.toml` runs clean with defaults even though pyproject.toml has **no `[tool.bandit]` section** (verified with bandit 1.9.4 — earlier team notes calling this broken are doc-rot; the section is genuinely absent, bandit just tolerates it). |
| `audit` / `audit-json` | `poetry export --with dev,test,aws --all-extras` from poetry.lock, then pip-audit (mirrors Dependabot) | Requires `.venv` present and poetry on PATH. Read-only apart from a temp requirements file. |
| `publish-testpypi` / `publish` | `gen-env` then `poetry build` + `poetry publish`; tokens come from the generated root `.env` (`.EXPORT_ALL_VARIABLES` + the `include`) | Owner-only. Also subject to the never-push-without-Dhar rule. |

## 6. Known traps on a fresh machine

| Symptom | Cause | Fix |
|---|---|---|
| Suite "passes" but with a wall of skips; FastAPI/topology-B/probes/MCP-token tests all skip | Key-gated: 7 test files (`test_command_executor.py`, `test_fastapi_service.py`, `test_fastapi_topology_b.py`, `test_fastapi_turns_async.py`, `test_mcp_token_generation.py`, `test_probes.py`, `test_train_modern_stack.py`) require repo-local `env/.env` + `passwords/.env` to exist (and some need real keys + the trained hello_world). A fresh clone **cannot** reproduce the 478-passed baseline. | Section 3 (secrets) + section 4 (golden artifact) |
| 4 FastAPI tests fail with `FileNotFoundError ... threshold.json` | hello_world model missing/wiped | Section 4 train command; enforce fix-0hb rule |
| "CI was green but my clone fails/skips everything" | CI is a single workflow, `.github/workflows/train-modern-stack.yml`, running ONE file: `python -m pytest tests/test_train_modern_stack.py` on Python 3.11 — and even that skips without the `LITELLM_API_KEY_SYNDATA_GEN` repo secret. The other ~493 tests never run in CI. | Always run the suite locally before trusting a change (`fastworkflow-validation-and-qa`) |
| First `fastworkflow train` of a pip-installed user workflow is much slower than expected | Wheel excludes `**/___command_info`, so the internal CME workflow trains first. Worse (candidate bug, unverified by running): `is_fast_workflow_trained` checks `___command_info/ErrorCorrection/largemodel.pth`, a folder training has not produced since v2.7.0 (local CME has only `IntentDetection/` + `global/` — verified on disk), so the check likely never passes and CME may retrain on EVERY train of a path not containing the substring "fastworkflow". | Budget time for it; status open — confirm with Dhar before "fixing" |
| Shell `export SESSION_STATE_STORE=...` (or `INTENT_DETECTION_*`) ignored | `fastworkflow.get_env_var` returns a supplied code default WITHOUT consulting `os.environ` (fastworkflow/__init__.py) — env FILES are the only override channel for defaulted vars | Put overrides in the env file, not the shell |
| `fastworkflow refine` fails with a DSPy/LM ValueError | build/refine load no env files; `LLM_COMMAND_METADATA_GEN` + `LITELLM_API_KEY_COMMANDMETADATA_GEN` must be OS-exported and aren't in any template | `export` both, then rerun |
| `--help` for train/run says env default is ".env in current directory" | Help-text rot: actual default is `<workflow>/fastworkflow.env` + `<workflow>/fastworkflow.passwords.env` (`cli.py` `find_default_env_files`) | Pass paths explicitly; trust code |
| `make` anything fails on fresh clone | The `include ./.env` parse-time trap (section 5) | `./gen-env.sh` once |
| Training fails loading tokenizer/model_type | transformers 5.x vs old checkpoints; defaults are already 5.x-safe, but custom `INTENT_DETECTION_*` overrides may not be | See `fastworkflow-nlu-pipeline-reference` |

Platform note: the only Windows guidance of record is README's "on Windows use WSL"; the
reference box is WSL2 and everything in this skill was verified there. No other
platform-specific behavior is evidenced in the repo.

## 7. Related discipline rules (owner-confirmed, cite fastworkflow-change-control)

1. **Never `git commit`/`git push` without the developer'sexplicit request in that turn**
   (2026-07-08 incident: auto-pushed private doc → public-repo history rewrite).
2. **One bd (beads) write at a time; verify `.beads/issues.jsonl` after each.** (`bd close
   --reason` failed to persist on an older bd, 2026-06-11; it persists on 1.1.2, retested
   2026-08-13.)
3. **Never wipe `fastworkflow/examples/*/___command_info`** — fix-0hb, section 4.

## Provenance and maintenance

Facts verified 2026-07-09 against v2.22.2 (commit c33b9a5). Re-verification one-liners
for every volatile fact:

| Fact | Re-verify with |
|---|---|
| Python constraint `>=3.11,<3.14`; extras `training`/`server`/`fastapi`; wheel excludes `___command_info` | `grep -n -A2 'python =\|\[tool.poetry.extras\]\|^exclude' pyproject.toml` |
| Local venv is 3.12.2, in-project, editable install | `cat .venv/pyvenv.cfg; .venv/bin/pip show fastworkflow \| grep -i editable` |
| Poetry 1.8.x lock; in-project config | `head -1 poetry.lock; poetry config virtualenvs.in-project` |
| `env/.env` tracked; `passwords/.env`, root `.env` ignored | `git ls-files env/.env; git check-ignore -v .env passwords/.env` |
| gen-env.sh writes root `.env`, override/ wins, `--exclude` flag | `sed -n '85,120p' gen-env.sh` |
| makefile parse-time `include ./.env` trap | `cp makefile /tmp/mt/ && cd /tmp/mt && make -n lint` (expect the include error) |
| No `[tool.bandit]` section; bandit still runs | `grep -c tool.bandit pyproject.toml; .venv/bin/bandit -c pyproject.toml -r fastworkflow/utils/dspy_utils.py` |
| ~1803 tests collected (v3.1.0) | `source .venv/bin/activate && python -m pytest --collect-only -q \| tail -1` |
| 478-passed baseline + fix-0hb narrative | `bd show fix-0hb` (read-only) |
| 4 model-dependent FastAPI tests + skip guards | `sed -n '15,40p' tests/test_fastapi_service.py; grep -n 'def test_' tests/test_fastapi_service.py` |
| Golden-artifact contents & size | `ls fastworkflow/examples/hello_world/___command_info/global/; du -sh fastworkflow/examples/hello_world/___command_info` |
| CI = one workflow, one test file | `ls .github/workflows/; grep -n 'pytest' .github/workflows/train-modern-stack.yml` |
| Key-gated test files list | `grep -rln '"env", ".env"\|env/.env' tests --include='*.py'` |
| CME lacks `ErrorCorrection/` trained folder | `ls fastworkflow/_workflows/command_metadata_extraction/___command_info/` |
| refine/build ignore env files; COMMANDMETADATA spelling | `grep -n 'init(env_vars={})' fastworkflow/refine/__main__.py fastworkflow/build/__main__.py; grep -rn 'COMMANDMETADATA' fastworkflow/build/genai_postprocessor.py` |
| Default env-file resolution (help text is wrong) | `grep -n -A5 'def find_default_env_files' fastworkflow/cli.py` |

Open/unverified items are labeled inline: hello_world train duration (README claim only),
the CME perpetual-retrain candidate bug, full-suite wall time.
