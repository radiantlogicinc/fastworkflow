import json
from typing import List, Dict
from .evaluator import TaskResult


def print_console(results: List[TaskResult], metrics: Dict[str, float]) -> None:
    print("Tau Bench Retail Evaluation Results")
    print(f"Tasks: {len(results)}")
    for k, v in metrics.items():
        print(f"{k}: {v}")


def write_json(path: str, results: List[TaskResult], metrics: Dict[str, float]) -> None:
    payload = {
        "results": [r.__dict__ for r in results],
        "metrics": metrics,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)