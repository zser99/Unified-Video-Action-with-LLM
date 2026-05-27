"""
Simulation eval with inference-time CoT orchestration (frozen UVA).

Examples:
  # LIBERO + CoT (needs NVIDIA GPU, checkpoint, dataset)
  python eval_sim_cot.py -c checkpoints/libero10.ckpt -o outputs/libero_cot

  # Laptop smoke test: one LIBERO task, fewer steps
  python eval_sim_cot.py -c checkpoints/libero10.ckpt -o outputs/smoke --quick_test

  # Baseline without CoT wrapper (same as eval_sim.py)
  python eval_sim_cot.py -c checkpoints/libero10.ckpt -o outputs/base --no_cot

  # CPU only (very slow; may OOM on large models)
  python eval_sim_cot.py -c checkpoints/pusht.ckpt -o outputs/pusht --device cpu --no_cot

  # OpenAI vision CoT (pip install openai; export OPENAI_API_KEY=...)
  python eval_sim_cot.py -c checkpoints/libero10.ckpt -o outputs/llm_cot \\
      --planner llm --llm_model gpt-4o-mini --quick_test --verbose_cot
"""

import json
import os
import pathlib
import random
import sys

import click
import dill
import hydra
import numpy as np
import torch
import wandb
from omegaconf import open_dict

from unified_video_action.workspace.base_workspace import BaseWorkspace
from unified_video_action.utils.load_env import load_env_runner
from unified_video_action.cot.factory import create_planner
from unified_video_action.policy.cot_orchestrated_policy import CoTOrchestratedPolicy


sys.stdout = open(sys.stdout.fileno(), mode="w", buffering=1)
sys.stderr = open(sys.stderr.fileno(), mode="w", buffering=1)


def _apply_quick_test(env_runners, cfg):
    runners = env_runners if isinstance(env_runners, list) else [env_runners]
    for runner in runners:
        if hasattr(runner, "n_test"):
            runner.n_test = min(getattr(runner, "n_test", 3), 1)
        if hasattr(runner, "n_train"):
            runner.n_train = min(getattr(runner, "n_train", 1), 0)
        if hasattr(runner, "max_steps"):
            runner.max_steps = min(getattr(runner, "max_steps", 500), 120)
    if isinstance(env_runners, list) and len(env_runners) > 1:
        return env_runners[:1]
    return env_runners


@click.command()
@click.option("-c", "--checkpoint", required=True)
@click.option("-o", "--output_dir", required=True)
@click.option("-d", "--device", default="cuda:0")
@click.option("--no_cot", is_flag=True, help="Run inner UVA only (baseline).")
@click.option("--replan_every", default=8, show_default=True)
@click.option("--num_candidates", default=1, show_default=True)
@click.option(
    "--candidate_strategy",
    default="first",
    type=click.Choice(["first", "smallest_delta"]),
)
@click.option("--verbose_cot", is_flag=True)
@click.option(
    "--quick_test",
    is_flag=True,
    help="1 LIBERO task, 1 test episode, shorter horizon (for laptops).",
)
@click.option(
    "--planner",
    default="rule",
    type=click.Choice(["rule", "llm"]),
    show_default=True,
    help="CoT planner: rule-based or OpenAI LLM (uses env obs images when available).",
)
@click.option("--llm_model", default="gpt-4o-mini", show_default=True)
@click.option(
    "--no_llm_fallback",
    is_flag=True,
    help="If set, LLM planner errors fail instead of rule fallback.",
)
def main(
    checkpoint,
    output_dir,
    device,
    no_cot,
    replan_every,
    num_candidates,
    candidate_strategy,
    verbose_cot,
    quick_test,
    planner,
    llm_model,
    no_llm_fallback,
):
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)

    if device.startswith("cuda") and not torch.cuda.is_available():
        print(
            "WARNING: CUDA not available. Use --device cpu (slow) or install NVIDIA drivers."
        )
        device = "cpu"

    payload = torch.load(open(checkpoint, "rb"), pickle_module=dill)
    cfg = payload["cfg"]

    seed = cfg.training.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    with open_dict(cfg):
        cfg.output_dir = output_dir

    cls = hydra.utils.get_class(cfg.model._target_)
    workspace = cls(cfg, output_dir=output_dir)
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)

    inner = workspace.ema_model
    inner.to(device)
    inner.eval()

    policy = inner
    cot_meta = {}
    if not no_cot:
        if getattr(cfg.task.dataset, "language_emb_model", None) is None:
            print(
                "WARNING: CoT orchestration uses language_goal strings; "
                "this checkpoint task has language_emb_model=null. "
                "CoT helps most on LIBERO / UMI with CLIP. Continuing anyway."
            )
        cot_planner = create_planner(
            planner,
            model=llm_model,
            fallback_rule_based=not no_llm_fallback,
        )
        policy = CoTOrchestratedPolicy(
            inner_policy=inner,
            planner=cot_planner,
            replan_every=replan_every,
            num_candidates=num_candidates,
            candidate_strategy=candidate_strategy,
            verbose=verbose_cot,
        )
        cot_meta = {
            "cot_enabled": True,
            "planner": planner,
            "replan_every": replan_every,
            "num_candidates": num_candidates,
            "candidate_strategy": candidate_strategy,
        }
        if planner == "llm":
            cot_meta["llm_model"] = llm_model
    else:
        cot_meta = {"cot_enabled": False}

    env_runners = load_env_runner(cfg, output_dir)
    if quick_test:
        env_runners = _apply_quick_test(env_runners, cfg)
        cot_meta["quick_test"] = True

    if "libero" in cfg.task.name:
        step_log = {}
        for env_runner in env_runners:
            runner_log = env_runner.run(policy)
            step_log.update(runner_log)
            print(step_log)

        all_test_mean_score = {
            k: v for k, v in step_log.items() if "test/" in k and "_mean_score" in k
        }
        step_log["test_mean_score"] = float(np.mean(list(all_test_mean_score.values())))
        runner_log = step_log
    else:
        env_runner = env_runners
        runner_log = env_runner.run(policy)
        if "test/mean_score" in runner_log:
            runner_log["test_mean_score"] = runner_log["test/mean_score"]

    json_log = {}
    for key, value in runner_log.items():
        if isinstance(value, wandb.sdk.data_types.video.Video):
            json_log[key] = value._path
        else:
            json_log[key] = value
    json_log.update(cot_meta)
    json_log["device"] = device

    out_path = os.path.join(
        output_dir, f"eval_cot_log_{os.path.basename(checkpoint)}.json"
    )
    print("Saving log to", out_path)
    with open(out_path, "w") as f:
        json.dump(json_log, f, indent=2, sort_keys=True)

    for k, v in json_log.items():
        print(k, v)


if __name__ == "__main__":
    main()
