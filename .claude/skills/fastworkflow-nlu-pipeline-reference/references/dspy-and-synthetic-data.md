# DSPy and synthetic data: full detail

Companion to `../SKILL.md` §3-4. Facts verified 2026-07-09 against v2.22.2 (commit
c33b9a5). Paths repo-relative.

## A. Synthetic utterances (train-time, litellm direct)

`fastworkflow/train/generate_synthetic.py::generate_diverse_utterances(seed_utterances,
command_name, ...)`. No command file needs to call it: a `Signature` without `generate_utterances` gets the framework
default (`command_directory.default_generate_utterances`), which is exactly the command name in
words plus `generate_diverse_utterances(plain_utterances)`. Define `generate_utterances` only to
replace that body.

Mechanics:

1. Loads `proj-persona/PersonaHub` `persona.jsonl` from HuggingFace (:40) — requires the
   `datasets` extra; otherwise returns `[command_name] + seeds` with a warning (:26-32).
2. Randomly samples `SYNTHETIC_UTTERANCE_GEN_NUMOF_PERSONAS` personas (template 4);
   batches of `SYNTHETIC_UTTERANCE_GEN_PERSONAS_PER_BATCH` (1), each asked for
   `SYNTHETIC_UTTERANCE_GEN_UTTERANCES_PER_PERSONA` (5) utterances (:14-16, env read at
   MODULE IMPORT with `int` coercion and no defaults).
3. One `litellm.completion` per batch on `LLM_SYNDATA_GEN` /
   `LITELLM_API_KEY_SYNDATA_GEN`: max_tokens=1000, temperature=1.0, top_p=0.9,
   stop=`["<|end_of_text|>"]` (:112-119). Note this path does NOT go through
   `dspy_utils.get_lm`, so `litellm_proxy/`-prefixed models here rely on litellm's own
   proxy handling — unverified whether proxy routing works for syndata gen; treat as
   open.
4. Reply parsed by splitting on `[PersonaName]` bracket sections (:128-143): drops lines
   < 4 chars or starting with `[`. Format-brittle by construction.
5. Returns `[command_name] + seed_utterances + generated` (:158).

Failure semantics table:

| Condition | Behavior | Risk |
|---|---|---|
| `litellm.exceptions.RateLimitError` | logs error, returns `[]` (:120-122) | Seeds are ALSO lost for that command; the classifier trains without it and training exits 0 |
| Any other exception | propagates (training fails loudly) | — |
| `datasets` missing | seeds-only + warning | Deliberate (slim images) |
| Malformed persona sections | silently skipped (`except IndexError: continue`, :142-143) | Quietly thin training data |

Post-train sanity check: count utterances per command implied by the trained label
distribution, or diff `___command_info` timestamps; a rate-limited command shows up as a
class with only ancestor 'wildcard' pressure against it.

## B. DSPy few-shot corpus for parameter extraction (train-time, litellm direct)

`fastworkflow/utils/generate_param_examples.py::generate_dspy_examples`, called per
parameterized command from `train/__main__.py:110-114` with `num_examples=15,
validation_threshold=0.3` (function defaults are 10 / 0.4 — the call site overrides).

1. Prompts `LLM_SYNDATA_GEN` (temperature hardcoded 0.9 at :335, max_tokens=4000 at :407)
   to emit literal `dspy.Example(command="...", param=..., ...).with_inputs("command")`
   blocks, given the command's Pydantic field annotations text.
2. Parses the reply by scanning for `dspy.Example(` ... `.with_inputs` line spans
   (:428-480) — format-brittle. Field values are then re-extracted per-type with regexes
   (:507-550). Field metadata parsing uses `eval()` on an LLM-derived examples list as a
   fallback (:179) — known-weak point.
3. Validates each example: every non-None string param must appear in the utterance
   exactly or within normalized-Levenshtein confidence ≥ 1 − threshold (i.e. ≥ 0.7 at the
   call site) against 1-5-word phrases of the utterance (:21-82, :553).
4. **VERIFIED DEFECT — validation does not filter.** The function returns
   `transform_examples_to_dict_format(examples)` — ALL parsed examples — at :608-610;
   `return validated_examples, rejected_examples` is commented out at :612. Rejections
   are only logged and dumped to `rejected_examples.json` in the CURRENT WORKING
   DIRECTORY (:605-607). So `valid_examples` in
   `___command_info/<cmd>_param_labeled.json` (written by `train/__main__.py:122-129`)
   can contain hallucinated values. If you fix this, you are changing extraction-quality
   behavior — treat as an experiment, not a cleanup.
5. Numbers-only caveat: examples with int/float/None params are auto-valid (:30-37).

## C. Run-time parameter extraction (real DSPy)

`fastworkflow/utils/signatures.py::InputForParamExtraction.extract_parameters` (:239-313):

```
Pydantic Input model
  └─ create_signature_from_pydantic_model (:156-237)
       - one dspy.OutputField per field; desc = description + enum values + examples
         + Required/Optional + default hint
       - docstring = numbered extraction steps + today's date
       - input field literally described as "Statement according to Dhar" (:169) —
         a quirk baked into every extraction prompt
  └─ ChainOfThought(signature)  inside a dspy.Module (:263-271)
  └─ LabeledFewShot(k=len(trainset)).compile(...)  (:279-283)
       - trainset = valid_examples from <cmd>_param_labeled.json (get_trainset :35-63);
         missing file → empty trainset → zero-shot extraction (silently weaker)
  └─ dspy.context(lm=dspy.LM(LLM_PARAM_EXTRACTION, api_key=...), adapter=dspy.JSONAdapter())
       (:255, :278 — direct dspy.LM; the one call site NOT using dspy_utils.get_lm)
  └─ BestOfN(module, N=3, reward_fn=basic_checks, threshold=1.0)  (:295-299)
       - basic_checks returns 0.0 if ANY extracted value equals one of that field's
         Field(examples=[...]) values — anti-parroting (:285-292)
  └─ model_construct(**param_dict)  — NO validation; gaps become NOT_FOUND sentinels
       (str → NOT_FOUND, int → -sys.maxsize, float → -sys.float_info.max; :185-193)
```

Validation is a separate pass — `validate_parameters` (:315-670): type coercion
(str/bool/int/float/Enum/list with JSON/Python/CSV parsing), regex `pattern` fullmatch,
the command's own `validate_extracted_parameters` hook, which may fuzzy-match
against a DB and refuse with `fastworkflow.Refused`. Missing/invalid fields produce the clarification error
message assembled from `MISSING_INFORMATION_ERRMSG` / `INVALID_INFORMATION_ERRMSG` —
downstream code STRING-MATCHES on these messages (`parameter_extraction.py:147-157`), so
never reword those env values casually.

Agent-mode fast path (`_workflows/command_metadata_extraction/parameter_extraction.py`):

- If agentic (`run_as_agent` set, not an `/`-forced assistant command), regex-extract
  `<field>value</field>` XML first (:65-80, :296-357). ALL fields must be found and none
  may equal a `Field(examples=)` value (:327-337), else fall back to the DSPy path.
- On a clarification re-entry (stored params present), the user's bare reply is slotted
  into the sentinel-valued fields by comma-splitting — no LLM at all (:360-411).

Agent-level `AdapterParseError` (DSPy couldn't parse the LLM's structured output) retry:
whole agent call retried up to 2 attempts (`workflow_execution_context.py:696-707`).

## D. Refine-time DSPy (genai postprocessor)

`fastworkflow/build/genai_postprocessor.py`, run by `fastworkflow refine -w <wf>` (and
importable at build). LM = `dspy_utils.get_lm("LLM_COMMAND_METADATA_GEN",
"LITELLM_API_KEY_COMMANDMETADATA_GEN", max_tokens=2000)` (:239) — these two vars are NOT
in the packaged env template and refine loads no env files, so they must be exported in
the OS environment.

Five signatures, each wrapped in `dspy.ChainOfThought`:

| Signature (line) | Generates | Module at |
|---|---|---|
| `FieldMetadataSignature` (:37) | field description, 2-3 examples, regex pattern | :123 |
| `UtteranceGeneratorSignature` (:52) | minimal plain_utterances covering param combos | :151 |
| `SignatureDocstringSignature` (:66) | command docstring incl. the agent-facing XML example format `<command_name><param>value</param></command_name>` | :180 |
| `ContextDocstringSignature` (:86) | context handler docstring | :181 |
| `WorkflowDescriptionSignature` (:98) | `workflow_description.txt` at workflow root | :217 |

Edits are applied via LibCST transformers and are strictly additive/idempotent:
docstrings only when missing, field metadata only when absent, utterances only appended,
files rewritten only on change (:394-427). Generation failures fall back to deterministic
defaults and never fail the run.

Known limitation: CLI `refine` passes `classes={}` (`refine/__main__.py:43`), so method
docstrings from the application source are NOT available as DSPy inputs — metadata is
generated from field names/types alone. Open question whether that's intentional.

## E. DSPy cache operations

DSPy caches LLM calls (memory + disk). Symptoms of a stale cache: refine output doesn't
change after a prompt/signature edit; agent behaves identically despite a model change;
`dspy.inspect_history()` shows nothing (cached calls don't appear).

```bash
# Nuke disk cache (safe, will re-pay API calls). Actual dir verified via the status
# command below: ~/.dspy_cache (there is NO ~/.cache/dspy).
rm -rf ~/.dspy_cache/ ./.dspy_cache/
# Or in-process
python -c "import dspy; dspy.configure_cache(enable_disk_cache=False, enable_memory_cache=False, enable_litellm_cache=False)"
# Repo utility (lives at fastworkflow/utils/, not repo root as the guide claims).
# MUST be run as a module: running the file by path (python fastworkflow/utils/dspy_cache_utils.py)
# puts fastworkflow/utils/ first on sys.path, so its logging.py shadows stdlib logging and
# the import chain crashes with a pydantic circular-import error.
python -m fastworkflow.utils.dspy_cache_utils status       # verified subcommands: clear | clear-disk | reset | status
python -m fastworkflow.utils.dspy_cache_utils clear-disk --cache-dir /custom/path
```

`docs/DSPY_CACHE_GUIDE.md` is the doc of record but references a removed
`fastworkflow.run_agent.agent_module` — ignore those snippets, use the two commands above.

## Re-verification

```bash
grep -n "temperature\|max_tokens" fastworkflow/train/generate_synthetic.py fastworkflow/utils/generate_param_examples.py
sed -n '604,613p' fastworkflow/utils/generate_param_examples.py          # non-filtering return still there?
grep -n "BestOfN\|LabeledFewShot\|JSONAdapter\|according to Dhar" fastworkflow/utils/signatures.py
grep -n "class .*Signature\|ChainOfThought\|get_lm" fastworkflow/build/genai_postprocessor.py
grep -n "classes={}" fastworkflow/refine/__main__.py
```
