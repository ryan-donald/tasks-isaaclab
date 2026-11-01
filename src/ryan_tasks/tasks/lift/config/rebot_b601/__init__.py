# Copyright (c) 2024-2026, Ryan Donald
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Pick and place: move the cube to a table target, release it and go home."""

import gymnasium as gym

RSL_RL = f"{__name__}.agents.rsl_rl_ppo_cfg:PlaceRunnerCfg"

ENV_CFG = f"{__name__}.env_cfg"

gym.register(
    id="Ryan-Place-Cube-B601-DM",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{ENV_CFG}:PlaceEnvCfg",
        "rsl_rl_cfg_entry_point": RSL_RL,
    },
)

gym.register(
    id="Ryan-Place-Cube-B601-DM-Play",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{ENV_CFG}:PlaceEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": RSL_RL,
    },
)

gym.register(
    id="Ryan-Place-Cube-B601-DM-Direct",
    entry_point=f"{__name__}.direct_env:PlaceDirectEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.direct_env:PlaceDirectEnvCfg",
        "rsl_rl_cfg_entry_point": RSL_RL,
    },
)
