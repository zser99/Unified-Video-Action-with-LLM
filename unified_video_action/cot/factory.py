"""Construct CoT planners by name."""

from __future__ import annotations

from typing import Any

from unified_video_action.cot.planner import CoTPlanner, RuleBasedCoTPlanner


def create_planner(name: str, **kwargs: Any) -> CoTPlanner:
    name = name.lower().strip()
    if name in ("rule", "rule_based", "rules"):
        kwargs.pop("model", None)
        return RuleBasedCoTPlanner(**kwargs)
    if name in ("llm", "openai", "gpt"):
        from unified_video_action.cot.llm_planner import LLMCoTPlanner

        return LLMCoTPlanner(**kwargs)
    raise ValueError(f"Unknown planner {name!r}. Use 'rule' or 'llm'.")
