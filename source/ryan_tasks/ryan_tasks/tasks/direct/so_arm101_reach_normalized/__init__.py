# Copyright (c) 2024-2026, Ryan Donald
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

ENV_CFG = f"{__name__}.reach_env_cfg"
AGENTS = "ryan_tasks.tasks.manager_based.so_arm101_reach_normalized.agents"
KWARGS = {
    "rsl_rl_cfg_entry_point": f"{AGENTS}.rsl_rl_ppo_cfg:ReachPPORunnerCfg",
    "rl_games_cfg_entry_point": f"{AGENTS}:rl_games_ppo_cfg.yaml",
    "skrl_cfg_entry_point": f"{AGENTS}:skrl_ppo_cfg.yaml",
    "sb3_cfg_entry_point": f"{AGENTS}:sb3_ppo_cfg.yaml",
}

gym.register(
    id="Ryan-Reach-SO-ARM101-Normalized-Direct-v0",
    entry_point=f"{__name__}.reach_env:SoArm101ReachDirectEnv",
    disable_env_checker=True,
    kwargs={"env_cfg_entry_point": f"{ENV_CFG}:SoArm101ReachDirectEnvCfg", **KWARGS},
)

gym.register(
    id="Ryan-Reach-SO-ARM101-Normalized-Direct-Newton-v0",
    entry_point=f"{__name__}.reach_env:SoArm101ReachDirectEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{ENV_CFG}:SoArm101ReachDirectNewtonEnvCfg",
        **KWARGS,
    },
)
