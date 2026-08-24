# Copyright (c) 2025-2026, Ryan Donald
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Custom reward terms for the normalized SO-ARM101 reach environment."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import combine_frame_transforms

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def position_settle_reward(
    env: ManagerBasedRLEnv,
    std: float,
    vel_std: float,
    command_name: str,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Reward that is high only when the EE is close to the goal AND the arm is slow.

    Multiplicative "close AND slow": the position tanh kernel ``(1 - tanh(d/std))``
    times a joint-speed tanh kernel ``(1 - tanh(speed/vel_std))``. Because the
    product collapses whenever the arm is moving, oscillating through the target
    earns almost nothing -- so this rewards stopping at the goal rather than hunting
    around it. That hunting is the failure mode of the tight position-tracking term
    under the action delay: best error improves (the policy reaches the target) but
    it cannot hold, which inflates the settled error.

    ``asset_cfg`` must resolve both the EE body (``body_names``) and the arm joints
    (``joint_names``) whose speed defines "slow".
    """
    asset: Articulation = env.scene[asset_cfg.name]

    # EE distance to the commanded position, in the world frame (same as
    # position_command_error).
    command = env.command_manager.get_command(command_name)
    des_pos_b = command[:, :3]
    des_pos_w, _ = combine_frame_transforms(
        asset.data.root_pos_w.torch, asset.data.root_quat_w.torch, des_pos_b
    )
    curr_pos_w = asset.data.body_pos_w.torch[:, asset_cfg.body_ids[0]]
    distance = torch.linalg.norm(curr_pos_w - des_pos_w, dim=1)
    near = 1.0 - torch.tanh(distance / std)

    # arm joint speed; the kernel is ~1 only when the arm is nearly still.
    speed = torch.linalg.norm(asset.data.joint_vel.torch[:, asset_cfg.joint_ids], dim=1)
    slow = 1.0 - torch.tanh(speed / vel_std)

    return near * slow
