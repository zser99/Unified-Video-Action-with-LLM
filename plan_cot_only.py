"""
Run CoT planning only (no UVA, no simulator).

Examples:
  # Rule-based phases (no API)
  python plan_cot_only.py --base_goal "put the bowl on the stove" --num_replans 4

  # OpenAI vision + text (needs OPENAI_API_KEY, pip install openai)
  python plan_cot_only.py --planner llm --base_goal "pick up the mug" \\
      --image path/to/frame.jpg --num_replans 3 -o plans.json

  export OPENAI_API_KEY=sk-...
"""

from __future__ import annotations

import json
import os
import pathlib
from typing import Any, Dict, List, Optional

import click

from unified_video_action.cot.factory import create_planner
from unified_video_action.cot.obs_encoding import image_file_to_obs_dict


def _plan_to_dict(plan) -> Dict[str, Any]:
    return {
        "language_goal": plan.language_goal,
        "subgoal": plan.subgoal,
        "phase_index": plan.phase_index,
        "cot_trace": plan.cot_trace,
        "candidate_subgoals": list(plan.candidate_subgoals),
    }


@click.command()
@click.option(
    "--base_goal",
    required=True,
    help="Full task instruction (e.g. LIBERO language goal).",
)
@click.option("--num_replans", default=4, show_default=True)
@click.option("--replan_every", default=8, show_default=True, help="Simulated step_index stride.")
@click.option(
    "--planner",
    default="rule",
    type=click.Choice(["rule", "llm"]),
    show_default=True,
)
@click.option("--image", type=click.Path(exists=True), default=None, help="Optional RGB frame.")
@click.option("--llm_model", default="gpt-4o-mini", show_default=True)
@click.option("--num_candidates", default=1, show_default=True)
@click.option(
    "--no_llm_fallback",
    is_flag=True,
    help="If set, LLM errors propagate instead of rule fallback.",
)
@click.option("-o", "--output", type=click.Path(), default=None, help="Write JSON plans here.")
def main(
    base_goal: str,
    num_replans: int,
    replan_every: int,
    planner: str,
    image: Optional[str],
    llm_model: str,
    num_candidates: int,
    no_llm_fallback: bool,
    output: Optional[str],
):
    obs_dict = image_file_to_obs_dict(image) if image else None

    planner_kwargs: Dict[str, Any] = {}
    if planner == "llm":
        planner_kwargs = {
            "model": llm_model,
            "fallback_rule_based": not no_llm_fallback,
        }
        if not os.environ.get("OPENAI_API_KEY"):
            click.echo(
                "WARNING: OPENAI_API_KEY is not set. "
                "LLM planner will use rule fallback if enabled."
            )

    cot_planner = create_planner(planner, **planner_kwargs)

    records: List[Dict[str, Any]] = []
    for replan_index in range(num_replans):
        step_index = replan_index * replan_every
        plan = cot_planner.plan(
            base_goal=base_goal,
            step_index=step_index,
            replan_index=replan_index,
            obs_dict=obs_dict,
            num_candidates=num_candidates,
        )
        row = {
            "replan_index": replan_index,
            "step_index": step_index,
            **_plan_to_dict(plan),
        }
        records.append(row)
        click.echo(f"\n=== replan {replan_index} (step {step_index}) ===")
        click.echo(plan.cot_trace)
        click.echo(f"subgoal: {plan.subgoal}")
        click.echo(f"language_goal: {plan.language_goal}")

    payload = {
        "base_goal": base_goal,
        "planner": planner,
        "num_replans": num_replans,
        "replan_every": replan_every,
        "image": image,
        "plans": records,
    }

    if output:
        out_path = pathlib.Path(output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        click.echo(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
