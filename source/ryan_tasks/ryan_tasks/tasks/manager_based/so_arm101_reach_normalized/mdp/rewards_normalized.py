# Copyright (c) 2024-2026, Ryan Donald
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


def action_rate_l2_near_goal(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    radius: float,
) -> torch.Tensor:
    """Squared action change, applied only while the end effector is within radius of the goal.

    stepped smoothness penalty: leaves the approach fast and only enforces settling at the goal.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    des_pos_w, _ = combine_frame_transforms(
        asset.data.root_pos_w.torch, asset.data.root_quat_w.torch, command[:, :3]
    )
    curr_pos_w = asset.data.body_pos_w.torch[:, asset_cfg.body_ids[0]]
    distance = torch.linalg.norm(curr_pos_w - des_pos_w, dim=1)
    rate = torch.sum(
        torch.square(env.action_manager.action - env.action_manager.prev_action), dim=1
    )
    return rate * (distance < radius).float()
