"""OpenAI vision/text CoT planner (inference-time only)."""

from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, List, Optional

from unified_video_action.cot.obs_encoding import (
    encode_rgb_jpeg_data_url,
    find_image_array,
    obs_summary,
)
from unified_video_action.cot.planner import (
    CoTPlan,
    CoTPlanner,
    RuleBasedCoTPlanner,
    _format_language_goal,
)

_SYSTEM_PROMPT = """You are a robot manipulation planner. Given a task goal and optionally a camera image, output ONE short subgoal for the current moment and a brief chain-of-thought.

Respond with JSON only (no markdown fences):
{
  "subgoal": "imperative phrase under 25 words",
  "cot_trace": "numbered steps: Perceive, Reason, Subgoal",
  "candidate_subgoals": ["optional alternates if multiple candidates requested"]
}

Subgoals must be concrete, physically actionable, and suitable as CLIP text after the base goal.
"""


class LLMCoTPlanner(CoTPlanner):
    """
    Calls OpenAI Chat Completions (vision when obs image is present).

    Requires: pip install openai
    Auth: OPENAI_API_KEY environment variable (or api_key=...).
    On API/parse errors, optionally falls back to RuleBasedCoTPlanner.
    """

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        api_key: Optional[str] = None,
        fallback_rule_based: bool = True,
        temperature: float = 0.2,
    ):
        self.model = model
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.fallback_rule_based = fallback_rule_based
        self.temperature = temperature
        self._fallback = RuleBasedCoTPlanner()

    def plan(
        self,
        *,
        base_goal: str,
        step_index: int,
        replan_index: int,
        obs_dict: Optional[Dict[str, Any]] = None,
        num_candidates: int = 1,
    ) -> CoTPlan:
        if not self.api_key:
            if self.fallback_rule_based:
                plan = self._fallback.plan(
                    base_goal=base_goal,
                    step_index=step_index,
                    replan_index=replan_index,
                    obs_dict=obs_dict,
                    num_candidates=num_candidates,
                )
                plan.cot_trace = (
                    "[LLM skipped: OPENAI_API_KEY not set]\n" + plan.cot_trace
                )
                return plan
            raise RuntimeError(
                "OPENAI_API_KEY is not set. Export it or pass api_key= to LLMCoTPlanner."
            )

        try:
            return self._plan_llm(
                base_goal=base_goal,
                step_index=step_index,
                replan_index=replan_index,
                obs_dict=obs_dict,
                num_candidates=num_candidates,
            )
        except Exception as exc:
            if not self.fallback_rule_based:
                raise
            plan = self._fallback.plan(
                base_goal=base_goal,
                step_index=step_index,
                replan_index=replan_index,
                obs_dict=obs_dict,
                num_candidates=num_candidates,
            )
            plan.cot_trace = f"[LLM error: {exc}]\n" + plan.cot_trace
            return plan

    def _plan_llm(
        self,
        *,
        base_goal: str,
        step_index: int,
        replan_index: int,
        obs_dict: Optional[Dict[str, Any]],
        num_candidates: int,
    ) -> CoTPlan:
        from openai import OpenAI

        client = OpenAI(api_key=self.api_key)
        rgb = find_image_array(obs_dict)
        user_parts: List[Dict[str, Any]] = []

        prompt_text = (
            f"Base task goal: {base_goal}\n"
            f"Environment step index: {step_index}\n"
            f"Replan cycle: {replan_index}\n"
            f"Number of candidate subgoals requested: {num_candidates}\n"
        )
        if rgb is not None:
            prompt_text += "Use the attached camera image to judge progress.\n"
        else:
            prompt_text += f"Proprio summary: {obs_summary(obs_dict)}\n"

        user_parts.append({"type": "text", "text": prompt_text})
        if rgb is not None:
            user_parts.append(
                {
                    "type": "image_url",
                    "image_url": {"url": encode_rgb_jpeg_data_url(rgb)},
                }
            )

        response = client.chat.completions.create(
            model=self.model,
            temperature=self.temperature,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": user_parts},
            ],
            response_format={"type": "json_object"},
        )
        raw = response.choices[0].message.content or "{}"
        data = _parse_json_object(raw)

        subgoal = str(data.get("subgoal", "")).strip()
        if not subgoal:
            raise ValueError(f"LLM returned empty subgoal: {raw!r}")

        cot_trace = str(data.get("cot_trace", "")).strip()
        if not cot_trace:
            cot_trace = f"LLM subgoal: {subgoal}"

        candidates = data.get("candidate_subgoals") or []
        candidates = [str(c).strip() for c in candidates if str(c).strip()]
        if subgoal not in candidates:
            candidates = [subgoal] + candidates
        candidates = candidates[: max(1, num_candidates)]
        while len(candidates) < num_candidates:
            candidates.append(subgoal)

        language_goal = _format_language_goal(base_goal, subgoal)
        return CoTPlan(
            language_goal=language_goal,
            subgoal=subgoal,
            phase_index=replan_index,
            cot_trace=cot_trace,
            candidate_subgoals=candidates,
        )


def _parse_json_object(text: str) -> Dict[str, Any]:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    return json.loads(text)
