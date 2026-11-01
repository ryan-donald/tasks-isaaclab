# Copyright (c) 2024-2026, Ryan Donald
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""rsl_rl runner configs, matching the ryan_ppo configs in cfg/."""

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import (
    RslRlMLPModelCfg,
    RslRlOnPolicyRunnerCfg,
    RslRlPpoAlgorithmCfg,
)


def _model(hidden_dims: list[int], **kwargs) -> RslRlMLPModelCfg:
    return RslRlMLPModelCfg(
        hidden_dims=hidden_dims, activation="elu", obs_normalization=True, **kwargs
    )


def _algorithm(mini_batches: int, entropy: float) -> RslRlPpoAlgorithmCfg:
    return RslRlPpoAlgorithmCfg(
        num_learning_epochs=5,
        num_mini_batches=mini_batches,
        learning_rate=1.0e-3,
        schedule="adaptive",
        desired_kl=0.008,
        gamma=0.99,
        lam=0.95,
        clip_param=0.2,
        entropy_coef=entropy,
        value_loss_coef=1.0,
        use_clipped_value_loss=False,
        max_grad_norm=1.0,
    )


GAUSSIAN = RslRlMLPModelCfg.GaussianDistributionCfg(init_std=0.5, std_type="log")


@configclass
class PlaceRunnerCfg(RslRlOnPolicyRunnerCfg):
    experiment_name = "b601_place"
    max_iterations = 5000
    save_interval = 500
    num_steps_per_env = 24
    actor = _model([128, 128], distribution_cfg=GAUSSIAN)
    critic = _model([128, 128])
    algorithm = _algorithm(32, 1e-3)
