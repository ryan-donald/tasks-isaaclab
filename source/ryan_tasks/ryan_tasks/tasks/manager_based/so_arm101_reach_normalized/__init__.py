import gymnasium as gym

from . import agents, joint_pos_normalized_env_cfg  # noqa: F401

ENV_CFG = f"{__name__}.joint_pos_normalized_env_cfg"
AGENTS = f"{__name__}.agents"
RSL_RL_CFG = f"{AGENTS}.rsl_rl_ppo_cfg:ReachPPORunnerCfg"

# Register Normalized Gym environments ([-100, +100] observation/action space).
##

gym.register(
    id="Ryan-Reach-SO-ARM101-Normalized-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{ENV_CFG}:SoArm101ReachNormalizedEnvCfg",
        "rsl_rl_cfg_entry_point": RSL_RL_CFG,
        "rl_games_cfg_entry_point": f"{AGENTS}:rl_games_ppo_cfg.yaml",
        "skrl_cfg_entry_point": f"{AGENTS}:skrl_ppo_cfg.yaml",
        "sb3_cfg_entry_point": f"{AGENTS}:sb3_ppo_cfg.yaml",
    },
)

gym.register(
    id="Ryan-Reach-SO-ARM101-Normalized-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{ENV_CFG}:SoArm101ReachNormalizedEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": RSL_RL_CFG,
    },
)

gym.register(
    id="Ryan-Reach-SO-ARM101-Normalized-Newton-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{ENV_CFG}:SoArm101ReachNormalizedNewtonEnvCfg",
        "rsl_rl_cfg_entry_point": RSL_RL_CFG,
    },
)

gym.register(
    id="Ryan-Reach-SO-ARM101-Normalized-Newton-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{ENV_CFG}:SoArm101ReachNormalizedNewtonEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": RSL_RL_CFG,
    },
)
