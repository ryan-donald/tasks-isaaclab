# Copyright (c) 2024-2026, Ryan Donald
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""rsl_rl runner configs for the SO-ARM101 reach tasks.

The tuned runner follows ryan_ppo's cfg/Ryan-Reach-SO-ARM101-Normalized-v0.ini:
separate 64x64 ELU actor and critic, both with observation normalization, 16
minibatches, adaptive learning rate on a 0.008 KL target.
"""

from isaaclab.utils.configclass import configclass
from isaaclab_rl.rsl_rl import (
    RslRlMLPModelCfg,
    RslRlOnPolicyRunnerCfg,
    RslRlPpoAlgorithmCfg,
)

HIDDEN_DIMS = [64, 64]


@configclass
class ReachPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    experiment_name = "so_arm101_reach"
    run_name = ""
    resume = False
    max_iterations = 1000
    save_interval = 50
    num_steps_per_env = 24

    actor = RslRlMLPModelCfg(
        hidden_dims=HIDDEN_DIMS,
        activation="elu",
        obs_normalization=True,
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(
            init_std=0.5, std_type="log"
        ),
    )
    critic = RslRlMLPModelCfg(
        hidden_dims=HIDDEN_DIMS, activation="elu", obs_normalization=True
    )
    algorithm = RslRlPpoAlgorithmCfg(
        num_learning_epochs=5,
        num_mini_batches=16,
        learning_rate=1.0e-3,
        schedule="adaptive",
        desired_kl=0.008,
        gamma=0.99,
        lam=0.95,
        clip_param=0.2,
        entropy_coef=1e-4,
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        max_grad_norm=1.0,
    )


@configclass
class ReachMatchedPPORunnerCfg(ReachPPORunnerCfg):
    """Library-neutral benchmark config: 4 minibatches and entropy 0.01.

    skrl and rl_games' adaptive-LR rules collapse the lr at the tuned 16
    minibatches, so the cross-framework comparison runs at this setting.
    Matches ppo/cfg/Ryan-Reach-SO-ARM101-Normalized-Matched-v0.ini.
    """

    experiment_name = "so_arm101_reach_matched"

    def __post_init__(self):
        self.algorithm.num_mini_batches = 4
        self.algorithm.entropy_coef = 0.01
