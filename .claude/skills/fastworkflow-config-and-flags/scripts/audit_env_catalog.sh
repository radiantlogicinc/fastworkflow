#!/usr/bin/env bash
# audit_env_catalog.sh — READ-ONLY drift audit for the fastWorkflow env-var catalog.
#
# Re-derives (a) every env var consumed in fastworkflow/ core code and (b) every var
# declared in the packaged templates, then prints the two drift sets:
#   - consumed-but-untemplated (e.g. LLM_COMMAND_METADATA_GEN as of v2.22.2)
#   - templated-but-unconsumed (e.g. LLM_RESPONSE_GEN as of v2.22.2)
# Also prints direct os.environ reads (should stay ~3: __init__.py, turn.py, utils/env.py)
# and re-checks the headline sharp edges from SKILL.md.
#
# Usage: bash .claude/skills/fastworkflow-config-and-flags/scripts/audit_env_catalog.sh
# Run from the repo root. Writes nothing.

set -u
cd "$(git rev-parse --show-toplevel 2>/dev/null || echo .)" || exit 1

PKG=fastworkflow
ENV_TEMPLATE=$PKG/examples/fastworkflow.env
PW_TEMPLATE=$PKG/examples/fastworkflow.passwords.env

section() { printf '\n=== %s ===\n' "$1"; }

# ---------------------------------------------------------------------------
# (a) Vars consumed in core code
# ---------------------------------------------------------------------------
consumed=$(python3 - "$PKG" <<'EOF'
import pathlib, re, sys
pkg = pathlib.Path(sys.argv[1])
names = set()
patterns = [
    # get_env_var("X") — DOTALL so multi-line calls (e.g. model_pipeline_training.py:892) match
    re.compile(r"get_env_var\(\s*['\"]([A-Z][A-Z0-9_]+)['\"]", re.S),
    # get_lm("LLM_X", "LITELLM_API_KEY_X") — both args are env var names
    re.compile(r"get_lm\(\s*['\"]([A-Z][A-Z0-9_]+)['\"]\s*,\s*['\"]([A-Z][A-Z0-9_]+)['\"]", re.S),
    # direct os.environ.get("X") / os.getenv("X")
    re.compile(r"os\.(?:environ\.get|getenv)\(\s*['\"]([A-Z][A-Z0-9_]+)['\"]", re.S),
    # dotenv-dict probes at CLI entry: env_vars.get("X")
    re.compile(r"env_vars\.get\(\s*['\"]([A-Z][A-Z0-9_]+)['\"]", re.S),
]
for py in pkg.rglob("*.py"):
    text = py.read_text(errors="ignore")
    for pat in patterns:
        for m in pat.finditer(text):
            names.update(g for g in m.groups() if g)
print("\n".join(sorted(names)))
EOF
)

# ---------------------------------------------------------------------------
# (b) Vars declared in the packaged templates (active + commented examples)
# ---------------------------------------------------------------------------
templated=$(
  grep -hoE '^#? ?[A-Z][A-Z0-9_]+=' "$ENV_TEMPLATE" "$PW_TEMPLATE" 2>/dev/null \
    | sed -E 's/^#? ?//; s/=$//' | sort -u
)

section "Consumed in core code ($(echo "$consumed" | grep -c .))"
echo "$consumed"

section "Declared in packaged templates ($(echo "$templated" | grep -c .))"
echo "$templated"

section "DRIFT: consumed but NOT templated"
comm -23 <(echo "$consumed") <(echo "$templated") \
  | grep -vE '^(LOG_LEVEL|SESSION_STATE_STORE|SESSION_STATE_REDIS_URL|REDIS_URL|LITELLM_PROXY_API_BASE|LITELLM_PROXY_API_KEY|INTENT_DETECTION_TINY_MODEL|INTENT_DETECTION_LARGE_MODEL)$' \
  | sed 's/^/  /'
echo "  (vars with code defaults / deliberate non-template status are filtered; edit the"
echo "   grep -vE list above if you promote one to the template)"

section "DRIFT: templated but NOT consumed (dead config candidates)"
comm -13 <(echo "$consumed") <(echo "$templated") | sed 's/^/  /'

section "Direct os.environ/os.getenv reads in core (expect ~3 files)"
grep -rnE "os\.environ|os\.getenv" "$PKG" --include='*.py' | grep -v examples/ | sed 's/^/  /'

section "Sharp-edge spot checks"
printf '  COMMANDMETADATA spelling sites:\n'
grep -rn "COMMANDMETADATA" "$PKG" --include='*.py' | sed 's/^/    /'
printf '  LLM_RESPONSE_GEN consumers (expect none):\n'
hits=$(grep -rn "LLM_RESPONSE_GEN" "$PKG" tests/ --include='*.py')
if [ -n "$hits" ]; then echo "$hits" | sed 's/^/    /'; else echo "    (none — still dead)"; fi
printf '  expect_encrypted_jwt in cli.py (expect none — module-only flag):\n'
hits=$(grep -n "expect_encrypted_jwt" "$PKG/cli.py")
if [ -n "$hits" ]; then echo "$hits" | sed 's/^/    /'; else echo "    (none — still unreachable from wrapper)"; fi
printf '  build/refine empty init:\n'
grep -n "init(env_vars={})" "$PKG/build/__main__.py" "$PKG/refine/__main__.py" | sed 's/^/    /'
printf '  Import-time env reads (generate_synthetic.py:14-16, parameter_extraction.py:19-23):\n'
sed -n '14,16p' "$PKG/train/generate_synthetic.py" | sed 's/^/    /'

echo
echo "Done. Compare against Section 4 of SKILL.md; update the catalog on any new drift."
