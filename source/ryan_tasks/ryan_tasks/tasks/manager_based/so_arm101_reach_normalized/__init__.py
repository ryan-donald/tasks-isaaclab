import gymnasium as gym

from . import joint_pos_normalized_env_cfg, agents

##
# Register Normalized Gym environments ([-100, +100] observation/action space).
##

gym.register(
    id="Ryan-Reach-SO-ARM101-Normalized-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.joint_pos_normalized_env_cfg:SoArm101ReachNormalizedEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.agents.rsl_rl_ppo_cfg:ReachPPORunnerCfg",
    },
)

gym.register(
    id="Ryan-Reach-SO-ARM101-Normalized-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.joint_pos_normalized_env_cfg:SoArm101ReachNormalizedEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{__name__}.agents.rsl_rl_ppo_cfg:ReachPPORunnerCfg",
    },
)
