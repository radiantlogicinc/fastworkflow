from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Optional


@dataclass
class TaskResult:
    task_id: int
    reward: float
    steps: int
    success_turn: Optional[int]
    cost: float
    history: List[Dict]


def evaluate(tasks: List[Dict], env, agent_runner, max_steps: int = 30) -> Tuple[List[TaskResult], Dict[str, float]]:
    results: List[TaskResult] = []

    for idx, _task in enumerate(tasks):
        obs = env.reset(idx)
        history: List[Dict] = [
            {"role": "user", "content": obs.get("instruction") or obs}
        ]
        total_cost = 0.0
        success_turn: Optional[int] = None

        for step in range(1, max_steps + 1):
            action, step_cost = agent_runner.next_action(obs, history)
            total_cost += step_cost
            obs, reward, done, info = env.step(action)
            history.append({"role": "tool", "action": action, "obs": obs, "reward": reward})
            if reward == 1.0 and success_turn is None:
                success_turn = step
            if done:
                break

        results.append(
            TaskResult(
                task_id=idx,
                reward=float(obs.get("reward", 0.0)) if isinstance(obs, dict) else (1.0 if success_turn else 0.0),
                steps=len(history) - 1,
                success_turn=success_turn,
                cost=total_cost,
                history=history,
            )
        )

    # Aggregate metrics
    n = max(1, len(results))
    pass_at = {k: 0 for k in (1, 2, 3, 4)}
    avg_reward = 0.0
    total_cost = 0.0

    for r in results:
        avg_reward += 1.0 if r.reward == 1.0 else 0.0
        total_cost += r.cost
        for k in (1, 2, 3, 4):
            if r.success_turn == k:
                pass_at[k] += 1

    metrics = {
        "Pass^1": pass_at[1] * 100.0 / n,
        "Pass^2": pass_at[2] * 100.0 / n,
        "Pass^3": pass_at[3] * 100.0 / n,
        "Pass^4": pass_at[4] * 100.0 / n,
        "Average Reward": avg_reward / n,
        "Total Cost": total_cost,
        "Overall Score": sum(pass_at[k] * 100.0 / n for k in (1, 2, 3, 4)) / 4.0,
    }

    return results, metrics