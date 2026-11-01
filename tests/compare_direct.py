# Copyright (c) 2024-2026, Ryan Donald
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Checks that a direct task reproduces its manager-based twin bit for bit.

Steps each env from the same seed with the same action stream and staggered starts, and
compares observations, rewards, dones and per-term rewards.

    python tests/compare_direct.py --task Ryan-Reach-B601-DM
"""

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

import typing_extensions  # noqa: F401
from isaaclab.app import add_launcher_args, launch_simulation

parser = argparse.ArgumentParser()
parser.add_argument("--task", required=True)
parser.add_argument("--num_envs", type=int, default=256)
parser.add_argument("--steps", type=int, default=1500)
parser.add_argument(
    "--out", default=None, help="Run one task and save its rollout here."
)
add_launcher_args(parser)
args, _ = parser.parse_known_args()

if args.out is None:
    with tempfile.TemporaryDirectory() as tmp:
        paths = [Path(tmp) / "manager.pt", Path(tmp) / "direct.pt"]
        for task, out in zip((args.task, f"{args.task}-Direct"), paths):
            run = [sys.executable, __file__, "--task", task, "--out", str(out)]
            run += ["--num_envs", str(args.num_envs), "--steps", str(args.steps)]
            subprocess.run(run, check=True, stdout=subprocess.DEVNULL)
        import torch

        manager, direct = (torch.load(p) for p in paths)
    for name in manager:
        same = (manager[name] == direct[name]).reshape(args.steps, -1).all(dim=1)
        first = "none" if same.all() else int((~same).nonzero()[0])
        error = (manager[name].float() - direct[name].float()).abs().max()
        print(f"{name}: first differing step {first}, largest difference {error:.3g}")
    sys.exit(0)

args.require_kit = True
with launch_simulation(None, args):
    import gymnasium as gym
    import isaaclab_tasks  # noqa: F401
    import torch
    from isaaclab.cli import load_external_tasks
    from isaaclab_tasks.utils import parse_env_cfg

    load_external_tasks()
    cfg = parse_env_cfg(args.task, device="cuda:0", num_envs=args.num_envs)
    cfg.seed = 7
    env = gym.make(args.task, cfg=cfg)
    u = env.unwrapped
    obs, _ = env.reset()
    u.episode_length_buf.copy_(
        torch.randint_like(u.episode_length_buf, high=int(u.max_episode_length) // 24)
        * 24
    )
    generator = torch.Generator(device=u.device).manual_seed(3)
    shape = (u.num_envs, env.action_space.shape[-1])
    action = torch.zeros(shape, device=u.device)
    log = {"obs": [], "reward": [], "term": [], "done": [], "terms": []}
    for _ in range(args.steps):
        noise = torch.randn(shape, device=u.device, generator=generator) * 0.1
        action = 0.95 * action + noise
        obs, reward, term, trunc, _ = env.step(action)
        log["obs"].append(obs["policy"].clone())
        log["reward"].append(reward.clone())
        log["term"].append(term.clone())
        log["done"].append(trunc.clone())
        log["terms"].append(u.reward_manager._step_reward.clone())
    torch.save({k: torch.stack(v).cpu() for k, v in log.items()}, args.out)
    env.close()
sys.exit(0)
