# Harness reference

A working skeleton for the two-tier benchmark described in `SKILL.md`. Adapt names to the
workflow; the structure is what matters. Nothing here is workflow-specific.

## Placeholders and turns

A turn's parameters are resolved at run time, because most of them are not known when the task is
written — they come from seeding, or from an earlier turn's output.

```python
class Ref:
    """A value produced by an earlier turn of the same conversation."""
    def __init__(self, key): self.key = key
    def resolve(self, state): return state[self.key]


class Seed:
    """A value discovered during seeding, for chains that need a specific shape
    of object (a container that has children, a record that has a parent)."""
    def __init__(self, key): self.key = key
    def resolve(self, state): return state["_seeds"][self.key]


def resolve(value, state):
    if isinstance(value, (Ref, Seed)):
        return value.resolve(state)
    return value(state) if callable(value) else value


class Turn:
    """One thing the user says, and the command that should handle it.

    write          reversible write; needs --writes
    unsafe         real side effect; validated, never sent
    restore        undo turn; still runs when the chain is BLOCKED
    expect_context the context this turn must land in; also drives static planning
    capture        state key -> handle index into the rendered listing, or a callable
    """
    def __init__(self, say, command, params=None, capture=None, write=False,
                 unsafe=False, restore=False, expect_context=None):
        self.say, self.command = say, command
        self.params, self.capture = params or {}, capture or {}
        self.write, self.unsafe, self.restore = write, unsafe, restore
        self.expect_context = expect_context


class Conversation:
    def __init__(self, name, purpose, turns):
        self.name, self.purpose, self.turns = name, purpose, turns


class Task(Conversation):
    """`combines` names the tier-1 conversations this task re-walks. It is what
    makes a failure attributable: each block is green on its own, so a step that
    only fails at length is a length problem."""
    def __init__(self, name, purpose, combines, turns):
        super().__init__(name, purpose, turns)
        self.combines = combines
```

## Executing one turn

A command path is `Context/command_name`, or a bare `command_name` for the global context.

```python
def module_for(command_path):
    context_name, command_name = (command_path.split("/", 1) if "/" in command_path
                                  else ("*", command_path))
    module_name = (f"{WORKFLOW_PACKAGE}._commands.{command_name}" if context_name == "*"
                   else f"{WORKFLOW_PACKAGE}._commands.{context_name}.{command_name}")
    return importlib.import_module(module_name), command_name


def context_name_of(context):
    return "*" if context is None else type(context).__name__


def execute(workflow, command_path, context, params):
    """Invoke one command the way the runtime does, and report where it left us."""
    module, command_name = module_for(command_path)
    generator = module.ResponseGenerator()
    workflow.command_context_for_response_generation = context
    workflow.current_command_context = context
    if not hasattr(module.Signature, "Input"):
        return generator(workflow, command_name), workflow.current_command_context

    parameters = module.Signature.Input(**params)
    if hasattr(module.Signature, "validate_extracted_parameters"):
        ok, message = module.Signature.validate_extracted_parameters(
            workflow, command_name, parameters)
        if not ok:
            # A soft outcome, not an exception: the runtime hands the message back
            # to the caller to correct, so a bad call stays a conversation.
            return (fastworkflow.CommandOutput(
                command_response=fastworkflow.CommandResponse(
                    response=f"rejected its parameters: {message}", success=False)),
                context)
    return generator(workflow, command_name, parameters), workflow.current_command_context
```

## Reading handles the way an agent does

Listings normally render a summary line, a header naming the columns, then one row per line. Parse
the handle out of the text, never out of internal state.

```python
def handles_from(response):
    lines = response.splitlines()
    if len(lines) < 3 or "  " not in lines[1]:
        return []
    return [line.split("  ")[0] for line in lines[2:] if "  " in line]
```

## Verdicts

```python
def verdict_for(output, response):
    if not output.command_response.success:
        return "SOFT"
    if "is not available yet" in response:
        return "STUB"
    if response.startswith("No ") and "found." in response:
        return "EMPTY"
    if handles_from(response) or response.startswith("Entered "):
        return "OK"
    return "OK" if response.strip() else "EMPTY"
```

`SKIP` and `BLOCKED` are assigned by the runner, not by this function: `SKIP` when the run mode
excludes a write or unsafe turn, `BLOCKED` when an earlier turn already failed.

## Framework navigation commands

These live in the framework's metadata-extraction workflow, not the workflow's own command folder,
and they mutate `workflow.context["app_workflow"]`. Run the real bodies through a shim so the
parent chain being walked is the framework's, not a reimplementation. Do not overwrite the app
workflow's own context dict — it already holds live session state.

```python
CME_COMMANDS_DIR = os.path.join(os.path.dirname(fastworkflow.__file__),
                                "_workflows", "command_metadata_extraction", "_commands")
RESET = "IntentDetection/reset_context"
GO_UP = "IntentDetection/go_up"
WHAT_CAN_I_DO = "IntentDetection/what_can_i_do"
WHERE_AM_I = "IntentDetection/what_is_current_context"


class _CmeShim:
    def __init__(self, app_workflow):
        self.context = {"app_workflow": app_workflow}
        self.folderpath = os.path.dirname(CME_COMMANDS_DIR)


def core_module(command_path):
    context_name, command_name = command_path.split("/", 1)
    path = os.path.join(CME_COMMANDS_DIR, context_name, f"{command_name}.py")
    spec = importlib.util.spec_from_file_location(f"cme_{command_name}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def execute_any(workflow, command_path, context, params):
    """`execute` for workflow commands, the real core command for the rest."""
    if not command_path.startswith("IntentDetection/"):
        return execute(workflow, command_path, context, params)
    workflow.current_command_context = context
    workflow.command_context_for_response_generation = context
    output = core_module(command_path).ResponseGenerator()(
        _CmeShim(workflow), command_path)
    return output, workflow.current_command_context
```

## Static context planning

Navigation is declared, so the context each step runs in is knowable without a backend. Needed for
offline routing; doubles as an assertion in live mode.

```python
def plan(task):
    current, stack, planned = "*", [], []
    for turn in task.turns:
        runs_in = current
        if turn.command == RESET:
            current, stack = "*", []
        elif turn.command == GO_UP:
            current = stack.pop() if stack else "*"
        elif turn.expect_context:
            stack.append(current)
            current = turn.expect_context
        planned.append({"turn": turn, "runs_in": runs_in,
                        "ancestors": list(reversed(stack))})
    return planned
```

Order matters: test the two navigation verbs **before** `expect_context`, so a go-up turn that also
asserts its destination is planned as a pop rather than a push.

## Routing scorer

Loads one classifier per context and reuses it. Never touches the backend.

```python
class Router:
    def __init__(self, artifacts_dir):
        self.artifacts_dir, self._cache = artifacts_dir, {}

    def labels_for(self, context, utterance):
        from fastworkflow.model_pipeline_training import CommandRouter
        folder = "global" if context == "*" else context
        if folder not in self._cache:
            path = os.path.join(self.artifacts_dir, folder)
            self._cache[folder] = CommandRouter(path) if os.path.isdir(path) else None
        router = self._cache[folder]
        return None if router is None else [str(x) for x in router.predict(utterance)]


def route_turn(router, entry):
    turn, context = entry["turn"], entry["runs_in"]
    labels = router.labels_for(context, turn.say)
    if labels is None:
        return {"outcome": "no-model", "labels": []}
    want = turn.command

    if labels == [want]:
        return {"outcome": "direct", "labels": labels}
    if want in labels:
        return {"outcome": "ambiguous", "labels": labels}

    # Only a lone escalation label escalates. Beside real candidates it takes the
    # ambiguity branch at runtime and never walks up.
    if labels == ["wildcard"]:
        for ancestor in entry["ancestors"] + ["*"]:
            up = router.labels_for(ancestor, turn.say)
            if up is None:
                continue
            if up == [want]:
                return {"outcome": "escalated", "resolved_at": ancestor, "labels": labels}
            if want in up:
                return {"outcome": "ambiguous", "resolved_at": ancestor, "labels": up}
            if up != ["wildcard"]:
                break
        return {"outcome": "lost", "labels": labels}
    return {"outcome": "misrouted", "labels": labels}
```

## Held-out phrasing check

Run this before anything else and return non-zero on a leak. A phrasing that is also a training
seed measures memory, not routing.

```python
def check_heldout(phrasing_pools, seed_utterances):
    seeds = {s.lower() for s in seed_utterances}
    leaked = sorted({t.lower() for pool in phrasing_pools.values()
                     for t in pool if t.lower() in seeds})
    for text in leaked:
        print(f"  LEAKED {text!r}")
    return not leaked
```

## Phrasing rotation

Keep a pool per command and select deterministically, so variety costs nothing in reproducibility.
Fields let one pool serve every variant of a parameterised question.

```python
PHRASINGS = {
    "Container/list_members": [
        "who sits inside this {kind}",
        "enumerate the membership here",
        "name everyone in this {kind}",
    ],
}

def say(command, variant=0, **fields):
    text = PHRASINGS[command][variant % len(PHRASINGS[command])]
    return text.format(**fields) if fields else text


def step(command, variant=0, phrase_fields=None, **kwargs):
    """A Turn whose wording comes from the rotation rather than the call site."""
    return Turn(say(command, variant, **(phrase_fields or {})), command, **kwargs)
```

## Block functions

Tier-1 errands become the tier-2 vocabulary. Keep them small and parameterise the wording variant
so the same block reads differently in different tasks.

```python
def entity_overview(v=0):
    return [
        step("Resource/show_properties", v),
        step("Resource/show_risk_score", v),
        step("Resource/list_controls_hit", v),
    ]


def tag_cycle(tag, v=0):
    """Apply a working label, prove it landed, take it off again."""
    return [
        step("Resource/show_metadata", v),
        step("Resource/add_tag", v, {"tag": tag}, params={"tag": tag}, write=True),
        step("Resource/show_metadata", v + 1, write=True),
        step("Resource/remove_tag", v, {"tag": tag}, params={"tag": tag},
             write=True, restore=True),
    ]


task = Task("periodic-review", "...", ["entity-360", "stewardship"],
            [step("open_directory", 0, expect_context="Explorer"),
             *entity_overview(0),
             *tag_cycle(WORKING_TAG, 0)])
```

## CLI shape

```
--execute    run every step against the live backend
--route      score intent routing per step (no backend, no LLM)
--writes     run the reversible write turns
--unsafe     dispatch real side effects (not net-neutral)
--report     write the markdown task report
--only NAME  run one task
```

Require at least one of `--execute` / `--route`. Emit a machine-readable results file alongside the
markdown so a report or dashboard can be generated without re-running.

## Reported output

The markdown report should carry, per task: purpose, step count, contexts visited, which tier-1
conversations it combines, and a table of every step with its wording, command, the context it ran
in, and one column per scored axis. Include the ids the run actually used so a reader can replay
the task by hand.

The summary should carry, per task: steps, execution verdict counts, per-step clean routing rate,
`p^n`, and the index of the first derailing step.
