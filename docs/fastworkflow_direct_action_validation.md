# Direct actions silently skip `validate_extracted_parameters`

**Fixed in fastworkflow 2.30.0** (`fix-1in`). First observed against 2.22.0;
re-verified still open on 2.29.2 after the Topology-B / TurnResult /
memory-bounds work (v2.25–v2.28); closed by running the same
`InputForParamExtraction.create()` factory on the direct-action path.

Cross-referenced from the memory-bounds write-up
(`docs/fastworkflow_memory_fixes.md`), which treated the skipped hook as a
separate bug that made eviction-time context loss harder to catch on the
direct-action path.

## Summary

A command's `Signature.validate_extracted_parameters()` hook used to run on the
NLU/agent path but **never for a direct action**. Type and format validation
still happened when parameters were present; only the command's own hook — the
one that expresses cross-field rules and context preconditions — was skipped.
Nothing was logged, so the command executed as if its preconditions were
satisfied.

## Mechanism (root cause, retained for provenance)

`CommandExecutor.perform_action` used to build the extractor with the bare
constructor:

```python
# fastworkflow/command_executor.py (before 2.30.0)
            input_for_param_extraction = InputForParamExtraction(command=action.command)
            is_valid, error_msg, _, _ = input_for_param_extraction.validate_parameters(
                workflow, action.command_name, input_obj
            )
```

`InputForParamExtraction.input_for_param_extraction_class` defaults to `None`,
and only the `create()` factory resolves it from the routing definition:

```python
# fastworkflow/utils/signatures.py -- InputForParamExtraction.create()
        input_for_param_extraction_class = subject_command_routing_definition.get_command_class(
            subject_command_name, ModuleType.INPUT_FOR_PARAM_EXTRACTION_CLASS)
```

So on the direct-action path that attribute stayed `None`, and
`validate_parameters` took its early return:

```python
# fastworkflow/utils/signatures.py -- InputForParamExtraction.validate_parameters()
        if is_valid:
            if not (
                self.input_for_param_extraction_class and \
                            hasattr(self.input_for_param_extraction_class, 'validate_extracted_parameters')
            ):
                return (True, "All required parameters are valid.", {}, [])
```

The NLU path was fine because it constructs the extractor through the factory
in `_workflows/command_metadata_extraction/parameter_extraction.py`.

A related consequence of the same root cause: if a field declared `db_lookup`
and the direct action supplied a value for it, `validate_parameters` raised
`ValueError("input_for_param_extraction_class is not set.")` before the hook
would have run. So db-lookup commands failed closed on this path, while
precondition-only hooks failed open.

A second, narrower gap sat alongside it: the validation call was inside `if
action.parameters:`, so an action invoked with no parameters (including the
default empty `{}`) skipped even the type/format checks. A command whose fields
are all optional but which has real context preconditions got nothing at all.

## Scope (callers that were affected)

Everything that eventually calls `CommandExecutor.perform_action` with an
already-built `Action`:

| Entry point | Path | Hook (before / after) |
|---|---|---|
| `/invoke_agent`, `/invoke_assistant`, `/invoke_agent_stream` | `process_turn` → CME → `InputForParamExtraction.create()` | yes / yes |
| `/perform_action` | `process_action_turn` → `_process_action` → `perform_action` | **no** / **yes** |
| `/initialize` with a `startup_action` | same (`process_action_turn`) | **no** / **yes** |
| MCP tool calls | `CommandExecutor.perform_mcp_tool_call` → `perform_action` | **no** / **yes** |
| `ChatSession` Action / CLI `--startup_action` | `process_action` → `perform_action` | **no** / **yes** |

The `/perform_action` docstring still says it "bypasses parameter extraction"
(the NLU/DSPy step). That remains intentional; the *post*-extraction
precondition hook now runs.

### Interaction with session eviction (post-2.24.1)

Releases A–C (v2.25–v2.27, epic `fix-g03`) made live-session eviction routine and
added durable workflow-context restore. That closed the *loss* of context on the
eviction path for hooked workflows. This fix closes the complementary gap: a
missing precondition is now rejected on `/perform_action` instead of producing a
successful-looking `TurnOutput` grounded in absent state.

## Fix that landed (2.30.0)

In `CommandExecutor.perform_action`:

```python
        if action.parameters:
            input_obj = command_parameters_class(**action.parameters)
        else:
            input_obj = command_parameters_class()

        input_for_param_extraction = InputForParamExtraction.create(
            workflow, action.command_name, action.command
        )
        is_valid, error_msg, _, _ = input_for_param_extraction.validate_parameters(
            workflow, action.command_name, input_obj
        )
        if not is_valid:
            raise ValueError(
                f"Invalid action parameters for command '{action.command_name}'\n{error_msg}"
            )
```

Behaviour notes:

1. **Behaviour change.** Direct actions that previously sailed past a command's
   hook are now rejected. Callers that relied on the gap (knowingly or not)
   will start seeing `ValueError` / HTTP 500 from the turn layer.
2. **`create()` is used in full.** Routing lookup cost per action is accepted;
   resolving only the class was considered and deferred as unnecessary.
3. **Failure shape unchanged in this release.** A failed validation still raises
   `ValueError`, which `run_fastapi_mcp` surfaces as HTTP 500 via
   `TurnExecution.error`. Returning a structured `CommandOutput` /
   `TurnOutput` (matching the NLU path) remains a follow-up.

## Regression coverage

`tests/test_direct_action_validation.py` against
`tests/direct_action_validation_workflow/`:

- `create()` path and `perform_action` both reject with the hook's own message
  when a required context key is missing.
- `perform_action` succeeds when the key is present.
- Empty `{}` and optional-only parameters still run the hook (the old
  `if action.parameters:` guard).
