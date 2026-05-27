"""
Frozen UVA + inference-time CoT orchestration (Recommendation 1).

Each replan cycle: planner (CoT text) -> subgoal -> language_goal -> inner UVA.
UVA weights are not updated.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import torch

from unified_video_action.cot.planner import CoTPlanner, pick_candidate_index
from unified_video_action.cot.planner import _format_language_goal
from unified_video_action.policy.base_image_policy import BaseImagePolicy
from unified_video_action.model.common.normalizer import LinearNormalizer


class CoTOrchestratedPolicy(BaseImagePolicy):
    def __init__(
        self,
        inner_policy: BaseImagePolicy,
        planner: CoTPlanner,
        replan_every: int = 8,
        num_candidates: int = 1,
        candidate_strategy: str = "first",
        verbose: bool = False,
    ):
        super().__init__()
        self.inner = inner_policy
        self.planner = planner
        self.replan_every = max(1, int(replan_every))
        self.num_candidates = max(1, int(num_candidates))
        self.candidate_strategy = candidate_strategy
        self.verbose = verbose

        self._step_index = 0
        self._replan_index = 0
        self._current_plan = None
        self._base_goal: str = ""
        self.last_cot_trace: str = ""

    def reset(self):
        self.inner.reset()
        self._step_index = 0
        self._replan_index = 0
        self._current_plan = None
        self._base_goal = ""
        self.last_cot_trace = ""

    def set_normalizer(self, normalizer: LinearNormalizer):
        self.inner.set_normalizer(normalizer)

    def _resolve_base_goal(self, language_goal: Optional[List[str]]) -> str:
        if language_goal is None or len(language_goal) == 0:
            return self._base_goal or "complete the manipulation task"
        return str(language_goal[0])

    def _needs_replan(self) -> bool:
        if self._current_plan is None:
            return True
        return self._step_index > 0 and self._step_index % self.replan_every == 0

    def predict_action(
        self,
        obs_dict: Dict[str, torch.Tensor],
        language_goal=None,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        base = self._resolve_base_goal(language_goal)
        if self._base_goal == "":
            self._base_goal = base

        if self._needs_replan():
            plan = self.planner.plan(
                base_goal=self._base_goal,
                step_index=self._step_index,
                replan_index=self._replan_index,
                obs_dict=obs_dict,
                num_candidates=self.num_candidates,
            )
            self._current_plan = plan
            self.last_cot_trace = plan.cot_trace
            self._replan_index += 1
            if self.verbose:
                print("[CoT] replan ->", plan.language_goal)
                print(plan.cot_trace)

        assert self._current_plan is not None
        batch_size = _batch_size(obs_dict)

        if self.num_candidates == 1:
            goal_batch = [self._current_plan.language_goal] * batch_size
            result = self.inner.predict_action(
                obs_dict, language_goal=goal_batch, **kwargs
            )
            self._step_index += int(result["action"].shape[1])
            return result

        candidates = self._current_plan.candidate_subgoals[: self.num_candidates]
        action_tensors = []
        for subgoal in candidates:
            goal_str = _format_language_goal(self._base_goal, subgoal)
            out = self.inner.predict_action(
                obs_dict,
                language_goal=[goal_str] * batch_size,
                **kwargs,
            )
            action_tensors.append(out["action"])

        pick = pick_candidate_index(action_tensors, self.candidate_strategy)
        action = action_tensors[pick]
        self._step_index += int(action.shape[1])
        return {"action": action, "action_pred": action}

    @property
    def device(self):
        return self.inner.device

    @property
    def dtype(self):
        return self.inner.dtype


def _batch_size(obs_dict: Dict[str, torch.Tensor]) -> int:
    for v in obs_dict.values():
        if isinstance(v, torch.Tensor) and v.ndim >= 1:
            return int(v.shape[0])
    return 1
