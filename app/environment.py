"""
AP Clerk Environment — Core Environment Class
Implements the canonical OpenEnv interface: reset / step / state
"""

from __future__ import annotations
import copy
from typing import Optional, Tuple, Dict, Any

from .models import APObservation, APAction, APReward
from .tasks import TASKS, grade_action


class APClerkEnvironment:
    """
    AI Accounts Payable Clerk — Three-Way Invoice Matching Environment.

    Episode flow:
        obs = env.reset(task_id, seed=None)
        obs, reward, done, info = env.step(action)

    seed=None produces a fresh random episode each call.
    A fixed integer seed produces a fully reproducible episode.
    Episodes are single-step: done is True after the first step.
    """

    MAX_STEPS = 1

    def __init__(self) -> None:
        self._task_id: Optional[str] = None
        self._observation: Optional[APObservation] = None
        self._step_count: int = 0
        self._done: bool = False
        self._episode_score: float = 0.0

    def reset(self, task_id: str, seed: Optional[int] = None) -> APObservation:
        if task_id not in TASKS:
            raise ValueError(
                f"Unknown task_id {task_id!r}. "
                f"Valid options: {list(TASKS.keys())}"
            )
        spec = TASKS[task_id]
        self._task_id = task_id
        self._step_count = 0
        self._done = False
        self._episode_score = 0.0

        obs = spec.generator(seed=seed)
        obs.step_count = 0
        obs.max_steps = self.MAX_STEPS
        self._observation = obs
        return obs

    def step(self, action: APAction) -> Tuple[APObservation, APReward, bool, Dict[str, Any]]:
        if self._observation is None:
            raise RuntimeError("Call reset(task_id) before step().")
        if self._done:
            raise RuntimeError("Episode already finished. Call reset() to start a new one.")

        self._step_count += 1
        reward = grade_action(self._task_id, self._observation, action)

        self._done = True
        self._episode_score = reward.score
        self._observation.step_count = self._step_count

        info: Dict[str, Any] = {
            "task_id": self._task_id,
            "step_count": self._step_count,
            "episode_score": self._episode_score,
        }
        return self._observation, reward, self._done, info

    def state(self) -> Dict[str, Any]:
        return {
            "task_id": self._task_id,
            "step_count": self._step_count,
            "done": self._done,
            "episode_score": self._episode_score,
            "current_observation": self._observation,
        }

    @staticmethod
    def list_tasks() -> Dict[str, Dict[str, str]]:
        return {
            tid: {
                "name": spec.name,
                "difficulty": spec.difficulty,
                "description": spec.description,
            }
            for tid, spec in TASKS.items()
        }
