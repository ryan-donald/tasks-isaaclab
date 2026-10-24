import gymnasium as gym

# the variants share a runner cfg and differ only in their env cfg class.
ENV_CFG = f"{__name__}.joint_pos_env_cfg"
AGENTS = f"{__name__}.agents"
RSL_RL_CFG = f"{AGENTS}.rsl_rl_ppo_cfg:PushCubePPORunnerCfg"

# normalized [-100, +100] observation and action space. PhysX by default; pass
# physics=newton_mjwarp for Newton (MJWarp).

gym.register(
    id="Ryan-Push-Cube-SO-ARM101-Normalized",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{ENV_CFG}:SoArm101PushNormalizedEnvCfg",
        "rsl_rl_cfg_entry_point": RSL_RL_CFG,
    },
)

gym.register(
    id="Ryan-Push-Cube-SO-ARM101-Normalized-Play",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{ENV_CFG}:SoArm101PushNormalizedEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": RSL_RL_CFG,
    },
)

# Push-T: the cube push with a T block and a yaw goal. Same runner cfg, since the
# arm, action space and network shape are the cube push's.
PUSH_T_CFG = f"{__name__}.push_t_env_cfg"
for suffix, cfg in (
    ("", "SoArm101PushTNormalizedEnvCfg"),
    ("-Play", "SoArm101PushTNormalizedEnvCfg_PLAY"),
):
    gym.register(
        id=f"Ryan-Push-T-SO-ARM101-Normalized{suffix}",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{PUSH_T_CFG}:{cfg}",
            "rsl_rl_cfg_entry_point": RSL_RL_CFG,
        },
    )
