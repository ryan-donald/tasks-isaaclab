# Copyright (c) 2024-2026, Ryan Donald
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

ENV_CFG = f"{__name__}.joint_pos_env_cfg"
SO_ARM101 = "ryan_tasks.tasks.reach.config.so_arm101"
RSL_RL_CFG = f"{SO_ARM101}.agents.rsl_rl_ppo_cfg:ReachPPORunnerCfg"

gym.register(
    id="Ryan-Reach-B601-DM",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{ENV_CFG}:B601ReachNormalizedEnvCfg",
        "rsl_rl_cfg_entry_point": RSL_RL_CFG,
    },
)

gym.register(
    id="Ryan-Reach-B601-DM-Play",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{ENV_CFG}:B601ReachNormalizedEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": RSL_RL_CFG,
    },
)

# direct-workflow port of the manager-based task.
gym.register(
    id="Ryan-Reach-B601-DM-Direct",
    entry_point=f"{SO_ARM101}.direct_env:SoArm101ReachDirectEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.direct_env_cfg:B601ReachDirectEnvCfg",
        "rsl_rl_cfg_entry_point": RSL_RL_CFG,
    },
)
