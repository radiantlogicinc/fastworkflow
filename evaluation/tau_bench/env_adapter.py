from typing import Any, Dict, Tuple


class RetailEnvAdapter:
    """
    Minimal adapter around a Tau Bench-like environment.
    Expects an object with reset(task_index) and step(action) -> (obs, reward, done, info).
    """

    def __init__(self, env: Any):
        self._env = env

    def reset(self, task_index: int) -> Dict:
        return self._env.reset(task_index)

    def step(self, action: Dict) -> Tuple[Dict, float, bool, Dict]:
        return self._env.step(action)