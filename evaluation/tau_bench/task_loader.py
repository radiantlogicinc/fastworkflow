import json
import os
from typing import Iterable, List, Dict, Optional


def load_tasks(tasks_path: str, task_ids: Optional[Iterable[int]] = None) -> List[Dict]:
    """
    Load Tau Bench retail tasks from a file or directory.

    - If tasks_path is a file, it's treated as a JSON file containing a list of tasks.
    - If it's a directory, looks for a default 'retail.json' inside.
    - Optionally filter by 0-based indices specified in task_ids.
    """
    path = tasks_path
    if os.path.isdir(tasks_path):
        candidate = os.path.join(tasks_path, "retail.json")
        if not os.path.exists(candidate):
            raise FileNotFoundError(f"No retail.json found in {tasks_path}")
        path = candidate

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("Tasks file must contain a JSON list")

    if task_ids is None:
        return data

    selected = []
    idx_set = set(int(i) for i in task_ids)
    for idx, task in enumerate(data):
        if idx in idx_set:
            selected.append(task)
    return selected