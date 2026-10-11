# Copyright (c) 2024-2026, Ryan Donald
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Push-T observation terms. Yaw is reported as ``(sin, cos)``, which is continuous
and bounded."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg

from .push_t_rewards import goal_yaw_in_root_frame, object_yaw_in_root_frame

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def object_yaw_sin_cos(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """The block's yaw in the robot root frame, as ``(sin, cos)``. Shape (N, 2)."""
    robot: RigidObject = env.scene[robot_cfg.name]
    obj: RigidObject = env.scene[object_cfg.name]
    yaw = object_yaw_in_root_frame(robot, obj)
    return torch.stack((torch.sin(yaw), torch.cos(yaw)), dim=-1)


def goal_pose_in_robot_root_frame(
    env: ManagerBasedRLEnv,
    command_name: str = "object_pose",
) -> torch.Tensor:
    """The goal as ``(x, y, sin(yaw), cos(yaw))`` in the robot root frame. Shape (N, 4).

    z is left out: it is constant, and a constant column breaks the running
    observation normalizer.
    """
    command = env.command_manager.get_command(command_name)
    yaw = goal_yaw_in_root_frame(command)
    return torch.cat(
        (command[:, :2], torch.sin(yaw).unsqueeze(-1), torch.cos(yaw).unsqueeze(-1)),
        dim=-1,
    )
