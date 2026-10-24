# Copyright (c) 2024-2026, Ryan Donald
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Reset events for the lift and push tasks."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def hold_joint_position_target(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor | slice,
    asset_cfg: SceneEntityCfg,
    position: float,
) -> None:
    """Set a joint's position target to ``position`` on reset.

    A joint that is in no action term never has its target written, and an unwritten
    target is 0.0, so the actuator would drive it to zero. Since no action term touches
    the joint, the target persists for the episode.
    """
    asset = env.scene[asset_cfg.name]
    joint_ids = asset_cfg.joint_ids
    rows = env_ids
    if not isinstance(env_ids, slice) and not isinstance(joint_ids, slice):
        rows = env_ids[:, None]
    target = torch.full_like(asset.data.joint_pos.torch[rows, joint_ids], position)
    asset.actuators.target_command.set_position_index(
        value=target, joint_ids=joint_ids, env_ids=env_ids
    )
