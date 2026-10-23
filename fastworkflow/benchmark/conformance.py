"""Did an experiment run its benchmark version as written? (`fix-mgoi`)

Two independent answers, derived at read time from the benchmark version, the
experiment's attempt rows and the turns it recorded:

- **completeness** — ``complete`` when every task of the benchmark version has
  as many runs as the experiment declared per task, ``incomplete`` when any
  task is short of a run.
- **validity** — ``invalid`` when any run's first message differs from its
  task's saved prompt, ``valid`` otherwise. A task saved without a prompt
  cannot be checked and does not make a run invalid. An experiment created
  with ``user_simulator`` reads ``simulated_user`` instead and is not checked:
  its simulator turns each prompt into its own utterances (`fix-va9v`).

Read-only, and separate from the store's own experiment ``status``: that is the
evidence-integrity verdict a score rests on, this is whether the run was a run
of the benchmark.
"""

from __future__ import annotations

from typing import Any, Mapping

# How many mismatching runs a response names. The count is always exact.
_MISMATCH_LIMIT = 20
_PREVIEW = 200


def _preview(text: Any) -> str:
    text = "" if text is None else str(text)
    return text if len(text) <= _PREVIEW else text[:_PREVIEW] + "…"


def experiment_conformance(
    store: Any, manifest: Mapping[str, Any], experiment: Mapping[str, Any]
) -> dict[str, Any]:
    """Completeness and validity of one recorded experiment against its version."""
    experiment_id = str(experiment["experiment_id"])
    declared_attempts = int(experiment.get("declared_attempts") or 1)
    tasks = list(manifest.get("tasks") or [])
    runs: dict[str, int] = {}
    for row in store.experiment_attempt_rows(experiment_id):
        if row.get("execution_finished_at") is not None:
            runs[str(row["task_id"])] = runs.get(str(row["task_id"]), 0) + 1
    short = [
        {"task_id": task["task_id"], "runs": runs.get(task["task_id"], 0),
         "declared": declared_attempts}
        for task in tasks
        if runs.get(task["task_id"], 0) < declared_attempts
    ]

    prompts = {task["task_id"]: (task.get("prompt") or "").strip() for task in tasks}
    simulated = bool(experiment.get("user_simulator"))
    mismatches = []
    first_messages = {} if simulated else store.experiment_first_messages(experiment_id)
    for (task_id, attempt), sent in sorted(first_messages.items()):
        expected = prompts.get(task_id)
        if expected and (sent or "").strip() != expected:
            mismatches.append({"task_id": task_id, "attempt": attempt,
                               "expected": _preview(expected), "sent": _preview(sent)})
    if simulated:
        validity = "simulated_user"
    else:
        validity = "invalid" if mismatches else "valid"

    return {
        "benchmark_tasks": len(tasks),
        "tasks_fully_run": len(tasks) - len(short),
        "completeness": "incomplete" if short else "complete",
        "short_tasks": short,
        "validity": validity,
        "user_simulator": simulated,
        "prompt_mismatch_count": len(mismatches),
        "prompt_mismatches": mismatches[:_MISMATCH_LIMIT],
        "tasks_without_prompt": sum(1 for prompt in prompts.values() if not prompt),
    }


__all__ = ["experiment_conformance"]
