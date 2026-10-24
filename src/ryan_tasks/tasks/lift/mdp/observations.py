# Copyright (c) 2024-2026, Ryan Donald
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Observation terms for the lift task."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import subtract_frame_transforms

# the shared encoder-quantized joint observations.
from ryan_tasks.mdp.observations import (  # noqa: F401
    joint_pos_normalized_100_rel,
    joint_vel_finite_diff,
)

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def joint_pos_arm_normalized_100_rel(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    joint_names: list[str] | None = None,
) -> torch.Tensor:
    """Arm joint positions relative to default, normalized to [-100, +100].
    Shape (num_envs, num_arm_joints)."""
    asset: Articulation = env.scene[asset_cfg.name]
    if joint_names is None:
        # default to arm joints (exclude gripper).
        joint_ids, _ = asset.find_joints(["shoulder_.*", "elbow_flex", "wrist_.*"])
    else:
        joint_ids, _ = asset.find_joints(joint_names)

    joint_pos = asset.data.joint_pos.torch[:, joint_ids]
    default_pos = asset.data.default_joint_pos.torch[:, joint_ids]
    joint_limits = asset.data.soft_joint_pos_limits.torch[:, joint_ids, :]
    lower = joint_limits[:, :, 0]
    upper = joint_limits[:, :, 1]

    current_norm = 200.0 * (joint_pos - lower) / (upper - lower) - 100.0
    default_norm = 200.0 * (default_pos - lower) / (upper - lower) - 100.0
    return current_norm - default_norm


def joint_pos_gripper_normalized_0_100(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    joint_names: list[str] | None = None,
) -> torch.Tensor:
    """Gripper position in [0, 100]: 0 = closed (lower limit), 100 = open (upper
    limit). Shape (num_envs, num_gripper_joints)."""
    asset: Articulation = env.scene[asset_cfg.name]
    if joint_names is None:
        joint_ids, _ = asset.find_joints(["gripper"])
    else:
        joint_ids, _ = asset.find_joints(joint_names)

    joint_pos = asset.data.joint_pos.torch[:, joint_ids]
    joint_limits = asset.data.soft_joint_pos_limits.torch[:, joint_ids, :]
    lower = joint_limits[:, :, 0]
    upper = joint_limits[:, :, 1]

    normalized = 100.0 * (joint_pos - lower) / (upper - lower)
    return normalized


def effective_last_action(
    env: ManagerBasedRLEnv,
    arm_action_name: str = "arm_action",
    gripper_action_name: str = "gripper_action",
) -> torch.Tensor:
    """The last action as the robot executes it, in action units. Shape (num_envs, 5).

    Arm: the normalized target clamped to the joint limits, mapped back through the
    action scale and offset. Gripper: the sign of the command (+1 open, -1 close).
    Unlike the raw action, this is bounded, so it cannot feed back into ever larger
    commands.
    """
    arm = env.action_manager.get_term(arm_action_name)
    target = (arm.cfg.scale * arm.raw_actions + arm._offset).clamp(-100.0, 100.0)
    arm_eff = (target - arm._offset) / arm.cfg.scale
    grip = env.action_manager.get_term(gripper_action_name).raw_actions
    grip_eff = torch.where(grip < 0, -1.0, 1.0)
    return torch.cat([arm_eff, grip_eff], dim=1)


def command_position(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """Position part of a pose command, in the robot root frame. Shape (num_envs, 3).
    The lift goal is position-only."""
    return env.command_manager.get_command(command_name)[:, :3]


def object_position_in_robot_root_frame(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """The position of the object in the robot's root frame."""
    robot: RigidObject = env.scene[robot_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]
    object_pos_w = object.data.root_pos_w.torch[:, :3]
    object_pos_b, _ = subtract_frame_transforms(
        robot.data.root_link_pose_w.torch[:, :3],
        robot.data.root_link_pose_w.torch[:, 3:7],
        object_pos_w,
    )
    return object_pos_b
