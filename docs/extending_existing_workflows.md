# Extending Existing Workflows in fastWorkflow

This document describes how an application developer can build **custom workflows** that reuse and incrementally override features from one or more **base (template) workflows**.

---

## 1. High-level Flow

1.  Install or otherwise make available the template workflow(s).
2.  Create a new workflow folder (e.g. `./my_workflow/`).
3.  Add an **optional** `workflow_inheritance_model.json` that declares which workflow(s) it extends (the *base* workflows).
4.  Drop new or "wrapper" command modules under your workflow’s `_commands/` tree – only the bits you change need to be re-implemented; everything else is inherited from the base workflow.
5.  Run the standard CLI commands (`fastworkflow build/train/run …`). The loader merges the commands behind the scenes.

---

## 2. `workflow_inheritance_model.json`

Location: root of the workflow folder (sibling to `_commands/`). Optional – omit if you do **not** extend anything.

```json
{
    "base": [
        "fastworkflow.examples.simple_workflow_template",   // import-path of built-in template
        "acme_team.reporting_workflow",              // or any other importable package
        "./shared_workflows/local_template"          // or relative/absolute filesystem path
    ]
}
```

Rules

*  `base` order **matters** – first template in the list has lower precedence than later ones, and **all templates have lower precedence than the local workflow**.
*  Each entry must resolve either to
   * a Python *package* that contains an `_commands/` folder, or
   * a directory path whose basename is treated as the workflow folder.

---

## 3. Command-discovery search order

When `CommandDirectory.load(<workflow_root>)` runs, it walks these locations **in order**:

1.  Each path in `workflow_inheritance_model.json["base"]` (in given order) ⇒ their own `_commands` folders.
2.  `<workflow_root>/_commands` *(your custom workflow – highest precedence)*
3.  Built-in core workflow `fastworkflow/_workflows/command_metadata_extraction`.

For every command name (e.g. `get_task_summary` or `User/send_message`) the **last occurrence wins**. The
information in the last occurrence of the command name overwrites all previous ones.

---

## 4. Overriding behaviour with fastWorkflow command wrapper modules

A fastWorkflow command *wrapper module* is a Python file whose sole job is to subclass or re-export pieces from the template while modifying only what you need.

### 4.1  Example – override `plain_utterances` only

```python
# my_workflow/_commands/get_task_summary.py
from fastworkflow.templates.template_workflow._commands.get_task_summary import (
    Signature as BaseSig,
    ResponseGenerator as BaseRG,
)

class Signature(BaseSig):
    # add new user phrasings – everything else stays the same
    plain_utterances = BaseSig.plain_utterances + [
        "describe this task",
        "what is this about?",
    ]

# keep original runtime logic unchanged
ResponseGenerator = BaseRG  # simple alias
```

### 4.2  Example – override only the input schema

```python
from fastworkflow.templates.template_workflow._commands.load_workflow_definition import (
    Signature as BaseSig,
    ResponseGenerator,
)

class Signature(BaseSig):
    class Input(BaseSig.Input):
        # make path optional
        file_path: str | None = None
```

### 4.3  Example – override a validation helper

```python
from fastworkflow.templates.template_workflow._commands.get_task_summary import (
    Signature as BaseSig,
    ResponseGenerator,
)

class Signature(BaseSig):
    @staticmethod
    def validate_extracted_parameters(workflow: fastworkflow.Workflow, command: str, cmd_parameters: "Signature.Input") -> tuple[bool, str]:
        # call default checks first
        success, error_msg = super(Signature, Signature).validate_extracted_parameters(input)
        if not success:
            return (False, error_msg)

        if len(input.task_id) > 10:
            return (False, "Task id too long for this org policy")
```

No other attributes or methods need to be copied.

---

## 5. Precedence & conflict rules

* **Your workflow** overrides *all* templates.
* When two templates both define the same command, the one that appears **later** in `base` wins.
* If a command exists in neither your workflow nor any template, only the core commands are available.

---

## 6. Packaging & distribution of templates

Template workflows are ordinary Python packages that ship an `_commands/` folder plus any helper files (`context_inheritance_model.json`, `workflow_inheritance_model.json`, `context_hierarchy_model.json`, etc.). They can be:

* **Built-in** – published inside `fastworkflow.examples.*`.
* **Third-party** – installed via `pip install mycompany-wf-templates`.
* **Local** – referenced via relative path in `base`.

No code cloning or copying is required.

---

## 7. CLI helpers (road-map)

These helpers are planned but not yet implemented:

* `fastworkflow create-workflow <name> --extends fastworkflow.examples.simple_workflow_template` – scaffold a new workflow folder with a ready-made `workflow_inheritance_model.json` and `_commands/` stub.
* `fastworkflow build --check-overrides` – show which file provides each command after merge.

---

## 8. Limitations & future extensions

* The current merge strategy is **module-level** – you cannot mix `Signature` from one module with `ResponseGenerator` from another unless you write a wrapper that re-exports the combination.
* Declarative field-level overlays (`command_aliases.yaml`) are possible but *not* part of this minimal spec.
* Precedence is static; dynamic enable/disable of templates at runtime is out of scope.

---

### 9. Glossary

| Term | Meaning |
|------|---------|
| **Workflow folder** | Directory that contains `_commands/`, optional `workflow_inheritance_model.json`, and other fastWorkflow artefacts. |
| **Template / Base workflow** | A workflow package designed to be extended by others. |
| **Wrapper module** | A Python file in the extending workflow that subclasses or re-exports pieces from a template to override selective behaviour. |