import gymnasium as gym

# the variants share a runner cfg and differ only in their env cfg class.
ENV_CFG = f"{__name__}.joint_pos_env_cfg"
AGENTS = f"{__name__}.agents"
PPO_CFG = f"{AGENTS}.rsl_rl_ppo_cfg:LiftCubePPORunnerCfg"

# arm in normalized [-100, +100], gripper in [0, 100]. PhysX by default; pass
# physics=newton_mjwarp for Newton (MJWarp).

gym.register(
    id="Ryan-Lift-Cube-SO-ARM101-Normalized-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{ENV_CFG}:SoArm101LiftNormalizedEnvCfg",
        "rsl_rl_cfg_entry_point": PPO_CFG,
    },
)

gym.register(
    id="Ryan-Lift-Cube-SO-ARM101-Normalized-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{ENV_CFG}:SoArm101LiftNormalizedEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": PPO_CFG,
    },
)

# lift with hold-start resets and a bonus for holding a still arm command at the
# goal.
gym.register(
    id="Ryan-Lift-Cube-SO-ARM101-Normalized-Settle-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{ENV_CFG}:SoArm101LiftNormalizedSettleEnvCfg",
        "rsl_rl_cfg_entry_point": PPO_CFG,
    },
)

gym.register(
    id="Ryan-Lift-Cube-SO-ARM101-Normalized-Settle-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{ENV_CFG}:SoArm101LiftNormalizedSettleEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": PPO_CFG,
    },
)
