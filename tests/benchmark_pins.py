"""Benchmark pins for test fixtures (`fix-ujat`).

Every experiment pins a benchmark version. A test about something else still
has to name one, so it takes it from here:

- ``STORE_PIN`` for a test that writes the observability store directly; the
  store records the pin and does not look the version up.
- ``pin_for(workflow_path, task_ids)`` for a test that goes through
  ``ExperimentController``, which checks the pin against a saved version and
  the declared task ids against its tasks. It saves that version once per
  task set and returns its pin.
"""

import hashlib

from fastworkflow.benchmark import catalog

STORE_PIN = {
    "benchmark_id": "bench-test",
    "benchmark_version": "v1",
    "benchmark_digest_sha256": "0" * 64,
}


def pin_for(workflow_path, task_ids, *, version="v1"):
    """The pin of a saved benchmark version whose tasks are exactly ``task_ids``."""
    task_ids = list(dict.fromkeys(task_ids))
    benchmark_id = "bench-" + hashlib.sha256(
        "\x1f".join(sorted(task_ids)).encode("utf-8")
    ).hexdigest()[:12]
    try:
        manifest = catalog.load_version(workflow_path, benchmark_id, version)
    except (KeyError, FileNotFoundError, catalog.BenchmarkCatalogError):
        manifest = catalog.write_version(workflow_path, {
            "benchmark_id": benchmark_id,
            "version": version,
            "description": "",
            "tasks": [{"task_id": t, "description": "", "payload": {}} for t in task_ids],
        })
    return {
        "benchmark_id": benchmark_id,
        "benchmark_version": version,
        "benchmark_digest_sha256": manifest["digest_sha256"],
    }


def create_pinned(controller, experiment_id, description, *, declarations, **kwargs):
    """``controller.create_experiment`` pinned to a version of exactly its declared tasks."""
    declarations = list(declarations)
    pin = pin_for(controller.workflow_folderpath, [item[0] for item in declarations])
    return controller.create_experiment(
        experiment_id, description, declarations=declarations, **pin, **kwargs
    )
