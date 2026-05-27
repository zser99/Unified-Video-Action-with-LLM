"""Inference-time CoT planners (no UVA weight updates)."""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class CoTPlan:
    """One replan cycle: subgoal string fed to CLIP as language_goal."""

    language_goal: str
    subgoal: str
    phase_index: int
    cot_trace: str = ""
    candidate_subgoals: List[str] = field(default_factory=list)


# Generic manipulation phases for long-horizon LIBERO-style tasks.
DEFAULT_PHASES = [
    "approach the relevant object with the gripper aligned",
    "grasp or secure the object",
    "move toward the target placement region",
    "release or place the object at the goal",
]


class CoTPlanner(ABC):
    @abstractmethod
    def plan(
        self,
        *,
        base_goal: str,
        step_index: int,
        replan_index: int,
        obs_dict: Optional[Dict[str, Any]] = None,
        num_candidates: int = 1,
    ) -> CoTPlan:
        raise NotImplementedError


class RuleBasedCoTPlanner(CoTPlanner):
    """
    Offline-friendly planner: cycles subgoals from base task text.
    No external LLM/API required (good for laptops without API keys).
    """

    def __init__(self, phases: Optional[List[str]] = None):
        self.phases = phases or list(DEFAULT_PHASES)

    def plan(
        self,
        *,
        base_goal: str,
        step_index: int,
        replan_index: int,
        obs_dict: Optional[Dict[str, Any]] = None,
        num_candidates: int = 1,
    ) -> CoTPlan:
        del obs_dict, step_index
        phase_index = replan_index % len(self.phases)
        subgoal = self.phases[phase_index]

        candidates = [subgoal]
        if num_candidates > 1:
            candidates.extend(
                [
                    f"{subgoal}, move slowly",
                    f"{subgoal}, keep gripper steady",
                ][: num_candidates - 1]
            )

        perceive = f"Task: {base_goal}."
        constraint = f"Current phase {phase_index + 1}/{len(self.phases)}: {subgoal}."
        cot_trace = (
            f"1) Perceive: {perceive}\n"
            f"2) Constraint: {constraint}\n"
            f"3) Subgoal: {subgoal}\n"
            f"4) UVA will predict actions for this subgoal only."
        )

        language_goal = _format_language_goal(base_goal, subgoal)
        return CoTPlan(
            language_goal=language_goal,
            subgoal=subgoal,
            phase_index=phase_index,
            cot_trace=cot_trace,
            candidate_subgoals=candidates,
        )


def _format_language_goal(base_goal: str, subgoal: str) -> str:
    base = re.sub(r"\s+", " ", base_goal.strip())
    sub = re.sub(r"\s+", " ", subgoal.strip())
    # Keep within CLIP max_length=30 token budget (rough char guard).
    text = f"{base} | {sub}"
    if len(text) > 200:
        text = text[:200]
    return text


def pick_candidate_index(
    candidate_actions: List[Any],
    strategy: str = "first",
) -> int:
    """Cheap reranker when num_candidates>1 (no extra VLM call)."""
    if not candidate_actions:
        return 0
    if strategy == "first" or len(candidate_actions) == 1:
        return 0
    if strategy == "smallest_delta":
        import torch

        norms = []
        for act in candidate_actions:
            t = act if isinstance(act, torch.Tensor) else torch.as_tensor(act)
            norms.append(float(t[..., :9].norm().item()))
        return int(min(range(len(norms)), key=lambda i: norms[i]))
    return 0
