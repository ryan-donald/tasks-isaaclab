# Copyright (c) 2024-2026, Ryan Donald
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Termination terms for the lift task."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def physics_blowup(
    env: ManagerBasedRLEnv,
    max_vel_ratio: float = 5.0,
    max_object_distance: float = 1.0,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Terminate an env whose physics has diverged, before its state is observed.

    A diverged env's values (~1e9) would poison the running observation normalizer.
    Trips on a non-finite state, a joint faster than ``max_vel_ratio`` times its
    velocity limit (contacts can briefly exceed 1x), or the cube further than
    ``max_object_distance`` from the robot root.
    """
    robot: Articulation = env.scene[robot_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]
    joint_pos = robot.data.joint_pos.torch
    joint_vel = robot.data.joint_vel.torch
    object_pos = object.data.root_pos_w.torch
    nonfinite = ~(
        torch.isfinite(joint_pos).all(dim=1)
        & torch.isfinite(joint_vel).all(dim=1)
        & torch.isfinite(object_pos).all(dim=1)
    )
    too_fast = (
        joint_vel.abs() > max_vel_ratio * robot.data.joint_vel_limits.torch
    ).any(dim=1)
    too_far = (
        torch.norm(object_pos - robot.data.root_pos_w.torch, dim=1)
        > max_object_distance
    )
    return nonfinite | too_fast | too_far
