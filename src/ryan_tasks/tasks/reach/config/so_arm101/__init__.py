import gymnasium as gym

ENV_CFG = f"{__name__}.joint_pos_env_cfg"
AGENTS = f"{__name__}.agents"
RSL_RL_CFG = f"{AGENTS}.rsl_rl_ppo_cfg:ReachPPORunnerCfg"

# normalized [-100, +100] observation and action space. PhysX by default; pass
# physics=newton_mjwarp for Newton (MJWarp).

gym.register(
    id="Ryan-Reach-SO-ARM101-Normalized",
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

# same env, library-neutral agent configs (4 minibatches, entropy 0.01) for the
# cross-framework benchmark.
gym.register(
    id="Ryan-Reach-SO-ARM101-Normalized-Matched",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{ENV_CFG}:SoArm101ReachNormalizedEnvCfg",
        "rsl_rl_cfg_entry_point": f"{AGENTS}.rsl_rl_ppo_cfg:ReachMatchedPPORunnerCfg",
        "rl_games_cfg_entry_point": f"{AGENTS}:rl_games_ppo_matched_cfg.yaml",
        "skrl_cfg_entry_point": f"{AGENTS}:skrl_ppo_matched_cfg.yaml",
        "sb3_cfg_entry_point": f"{AGENTS}:sb3_ppo_matched_cfg.yaml",
    },
)

gym.register(
    id="Ryan-Reach-SO-ARM101-Normalized-Play",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{ENV_CFG}:SoArm101ReachNormalizedEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": RSL_RL_CFG,
    },
)

# direct-workflow port of the manager-based task (same MDP, bit-for-bit).
DIRECT_KWARGS = {
    "rsl_rl_cfg_entry_point": RSL_RL_CFG,
    "rl_games_cfg_entry_point": f"{AGENTS}:rl_games_ppo_cfg.yaml",
    "skrl_cfg_entry_point": f"{AGENTS}:skrl_ppo_cfg.yaml",
    "sb3_cfg_entry_point": f"{AGENTS}:sb3_ppo_cfg.yaml",
}
gym.register(
    id="Ryan-Reach-SO-ARM101-Normalized-Direct",
    entry_point=f"{__name__}.direct_env:SoArm101ReachDirectEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.direct_env_cfg:SoArm101ReachDirectEnvCfg",
        **DIRECT_KWARGS,
    },
)
