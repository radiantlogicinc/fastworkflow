#!/usr/bin/env bash
# qa_env_report.sh — READ-ONLY. Reports which fastWorkflow test groups will RUN
# vs SKIP on this machine, by checking the same gates the tests check.
# Usage: bash .claude/skills/fastworkflow-validation-and-qa/scripts/qa_env_report.sh
# Exit code is always 0; this script never writes anything.

set -u
REPO="$(cd "$(dirname "$0")/../../../.." && pwd)"
cd "$REPO" || exit 0

ok()   { printf '  [RUN ] %s\n' "$1"; }
skip() { printf '  [SKIP] %s\n' "$1"; }
info() { printf '  [INFO] %s\n' "$1"; }

echo "fastWorkflow QA environment report — $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "Repo: $REPO"
echo

echo "-- Interpreter and pytest --"
if [ -x .venv/bin/python ]; then
    info "venv python: $(.venv/bin/python --version 2>&1)"
    info "pytest: $(.venv/bin/python -m pytest --version 2>/dev/null | head -1 || echo 'NOT INSTALLED')"
else
    skip "no .venv/bin/python — create the venv first (see fastworkflow-build-and-env)"
fi
echo

echo "-- Gate 1: repo-local env files (FastAPI / Topology-B / probes / MCP-token tests) --"
[ -f env/.env ]        && ok "env/.env present"        || skip "env/.env missing -> test_fastapi_service, test_fastapi_topology_b, test_fastapi_turns_async, test_probes, test_mcp_token_generation all SKIP"
[ -f passwords/.env ]  && ok "passwords/.env present"  || skip "passwords/.env missing -> same five files SKIP"
echo

echo "-- Gate 2: pre-trained hello_world example (4 tests in test_fastapi_service.py) --"
THRESH="fastworkflow/examples/hello_world/___command_info/global/threshold.json"
if [ -f "$THRESH" ]; then
    ok "$THRESH present"
else
    skip "$THRESH missing -> 4 FastAPI tests FAIL (not skip!). Restore with:"
    echo "         fastworkflow train ./fastworkflow/examples/hello_world ./env/.env ./passwords/.env"
fi
echo

echo "-- Gate 3: training regression test (tests/test_train_modern_stack.py) --"
if [ -x .venv/bin/python ] && .venv/bin/python -c "import datasets" 2>/dev/null; then
    ok "datasets package importable"
else
    skip "datasets package missing -> training test SKIPs"
fi
if [ -f passwords/.env ] && grep -qE '^LITELLM_API_KEY_SYNDATA_GEN=..' passwords/.env 2>/dev/null \
   && ! grep -E '^LITELLM_API_KEY_SYNDATA_GEN=' passwords/.env | grep -qE '<|your-'; then
    ok "LITELLM_API_KEY_SYNDATA_GEN looks real (non-placeholder) in passwords/.env"
else
    skip "no real LITELLM_API_KEY_SYNDATA_GEN in passwords/.env -> training test SKIPs"
fi
echo

echo "-- Gate 4: untracked-but-required golden dirs (gitignored; fresh clones lack them) --"
[ -d tests/example_workflow/_commands ] && ok "tests/example_workflow/_commands present" \
    || skip "tests/example_workflow/_commands MISSING (gitignored dir) -> ~6 runtime test files break; copy from a provisioned machine or regenerate"
[ -d tests/hello_world_workflow/___command_info ] && ok "tests/hello_world_workflow/___command_info present" \
    || info "tests/hello_world_workflow/___command_info missing (regenerated cheaply by tests that build it)"
echo

echo "-- Suite size (slow: collects all tests) --"
if [ "${1:-}" = "--collect" ] && [ -x .venv/bin/python ]; then
    .venv/bin/python -m pytest --collect-only -q 2>/dev/null | tail -1
else
    info "pass --collect to run 'pytest --collect-only -q' (expected: ~1803 tests collected at v3.1.0)"
fi
echo
echo "Done. This script wrote nothing."
exit 0
