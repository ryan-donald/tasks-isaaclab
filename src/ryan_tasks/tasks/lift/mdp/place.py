# Copyright (c) 2024-2026, Ryan Donald
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Command and reward terms for placing the cube on a table target.

The task ends in a resting state: the cube released on its target and the arm back at
its default pose. Every placement reward requires the cube to have been carried (lifted
while pinched) this episode, so pushing it along the table earns nothing.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import MISSING
from typing import TYPE_CHECKING

import torch
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.configclass import configclass
from isaaclab.utils.math import combine_frame_transforms, subtract_frame_transforms
from isaaclab_tasks.core.lift.mdp.commands.pose_commands import ObjectUniformPoseCommand
from isaaclab_tasks.core.lift.mdp.commands.pose_commands_cfg import (
    ObjectUniformPoseCommandCfg,
)

from .rewards import (
    _object_clearance,
    object_ee_distance,
    object_grasped,
    object_lift_shaped,
    object_lifted_clearance,
    object_pinched,
)

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

class PlaceTargetCommand(ObjectUniformPoseCommand):
    """Table-level cube target, ``min_distance`` or more (xy) from the cube's spawn.

    Also latches ``carried`` once the cube has been lifted ``carry_clearance`` off the
    table while pinched, until the next resample (one per episode).
    """

    cfg: PlaceTargetCommandCfg

    def __init__(self, cfg: PlaceTargetCommandCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.carried = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.metrics["carried"] = torch.zeros(self.num_envs, device=self.device)

    def _resample_command(self, env_ids: Sequence[int]):
        super()._resample_command(env_ids)
        if isinstance(env_ids, slice):
            env_ids = torch.arange(self.num_envs, device=self.device)[env_ids]
        env_ids = torch.as_tensor(env_ids, device=self.device)
        root = self.robot.data.root_link_pose_w.torch[env_ids]
        cube_b, _ = subtract_frame_transforms(
            root[:, :3], root[:, 3:7], self.object.data.root_pos_w.torch[env_ids]
        )
        # rejection sampling: each try redraws the targets still too close to the cube.
        for _ in range(16):
            target = self.pose_command_b[env_ids, :2]
            close = torch.norm(target - cube_b[:, :2], dim=1) < self.cfg.min_distance
            ids = env_ids[close]
            if len(ids) == 0:
                break
            r = torch.empty(len(ids), device=self.device)
            self.pose_command_b[ids, 0] = r.uniform_(*self.cfg.ranges.pos_x)
            self.pose_command_b[ids, 1] = r.uniform_(*self.cfg.ranges.pos_y)
        self.carried[env_ids] = False

    def _update_command(self):
        clearance = _object_clearance(self.object, self.cfg.half_extent, 0.0)
        lifted = clearance > self.cfg.carry_clearance
        self.carried |= lifted & (object_pinched(self._env) > 0.0)

    def _update_metrics(self):
        super()._update_metrics()
        self.metrics["carried"] = self.carried.float()


@configclass
class PlaceTargetCommandCfg(ObjectUniformPoseCommandCfg):
    """Configuration for the place target command."""

    class_type: type = PlaceTargetCommand

    min_distance: float = MISSING
    """Least xy distance [m] from the cube's spawn position to its target."""

    carry_clearance: float = MISSING
    """Height [m] of the cube's lowest vertex above the table that counts as carried."""

    half_extent: float = MISSING
    """Half the cube's edge length [m]."""


def _carried(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    return env.command_manager.get_term(command_name).carried.float()


def object_carried(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """1.0 once the cube has been carried this episode."""
    return _carried(env, command_name)


def carried_observation(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """:func:`object_carried` as an observation, shape (num_envs, 1)."""
    return _carried(env, command_name).unsqueeze(1)


def object_lift_until_carried(
    env: ManagerBasedRLEnv,
    command_name: str,
    std: float,
    half_extent: float,
    table_z: float,
    reach_std: float,
    ee_cfg: SceneEntityCfg,
    ee_offset: tuple[float, float, float],
) -> torch.Tensor:
    """:func:`object_lift_shaped` until the cube has been carried, then 0."""
    lift = object_lift_shaped(
        env, std, half_extent, table_z, reach_std, ee_cfg=ee_cfg, ee_offset=ee_offset
    )
    return (1.0 - _carried(env, command_name)) * lift


def object_lifted_until_carried(
    env: ManagerBasedRLEnv,
    command_name: str,
    clearance: float,
    half_extent: float,
    table_z: float,
) -> torch.Tensor:
    """:func:`object_lifted_clearance` until the cube has been carried, then 0."""
    lifted = object_lifted_clearance(env, clearance, half_extent, table_z)
    return (1.0 - _carried(env, command_name)) * lifted


def object_ee_distance_until_carried(
    env: ManagerBasedRLEnv,
    command_name: str,
    std: float,
    ee_cfg: SceneEntityCfg,
    ee_offset: tuple[float, float, float],
) -> torch.Tensor:
    """:func:`object_ee_distance` until the cube has been carried, then 0."""
    reach = object_ee_distance(env, std, ee_cfg=ee_cfg, ee_offset=ee_offset)
    return (1.0 - _carried(env, command_name)) * reach


def object_grasped_until_carried(
    env: ManagerBasedRLEnv,
    command_name: str,
    grasp_std: float,
    gripper_open: float,
    gripper_cfg: SceneEntityCfg,
    ee_cfg: SceneEntityCfg,
    ee_offset: tuple[float, float, float],
) -> torch.Tensor:
    """:func:`object_grasped` until the cube has been carried, then 0."""
    grasped = object_grasped(
        env, grasp_std, gripper_open, gripper_cfg, ee_cfg=ee_cfg, ee_offset=ee_offset
    )
    return (1.0 - _carried(env, command_name)) * grasped


def object_pinched_until_carried(
    env: ManagerBasedRLEnv, command_name: str
) -> torch.Tensor:
    """:func:`object_pinched` until the cube has been carried, then 0."""
    return (1.0 - _carried(env, command_name)) * object_pinched(env)


def _target_offset_w(
    env: ManagerBasedRLEnv,
    command_name: str,
    robot_cfg: SceneEntityCfg,
    object_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Target minus cube position in the world frame, shape (num_envs, 3)."""
    robot: Articulation = env.scene[robot_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]
    command = env.command_manager.get_command(command_name)
    root = robot.data.root_link_pose_w.torch
    target_w, _ = combine_frame_transforms(root[:, :3], root[:, 3:7], command[:, :3])
    return target_w - object.data.root_pos_w.torch


def object_target_distance(
    env: ManagerBasedRLEnv,
    std: float,
    command_name: str,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """``1 - tanh(distance to the target / std)`` once the cube has been carried."""
    offset = _target_offset_w(env, command_name, robot_cfg, object_cfg)
    kernel = 1.0 - torch.tanh(torch.norm(offset, dim=1) / std)
    return _carried(env, command_name) * kernel


def object_placed(
    env: ManagerBasedRLEnv,
    std: float,
    command_name: str,
    rest_clearance: float,
    half_extent: float,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """``1 - tanh(xy distance to the target / std)`` while a carried cube is down.

    On the table means its lowest vertex is within ``rest_clearance`` [m] of it.
    """
    offset = _target_offset_w(env, command_name, robot_cfg, object_cfg)
    kernel = 1.0 - torch.tanh(torch.norm(offset[:, :2], dim=1) / std)
    object: RigidObject = env.scene[object_cfg.name]
    on_table = _object_clearance(object, half_extent, 0.0) < rest_clearance
    return _carried(env, command_name) * on_table.float() * kernel


def _released(
    env: ManagerBasedRLEnv,
    open_position: float,
    gripper_cfg: SceneEntityCfg,
    sensor_names: tuple[str, ...],
    max_force: float,
) -> torch.Tensor:
    """1.0 while the gripper is open past ``open_position`` [rad], off the cube."""
    robot: Articulation = env.scene[gripper_cfg.name]
    is_open = robot.data.joint_pos.torch[:, gripper_cfg.joint_ids][:, 0] > open_position
    touching = torch.zeros_like(is_open)
    for name in sensor_names:
        forces = env.scene[name].data.force_matrix_w.torch
        touching |= forces.flatten(1, 2).norm(dim=-1).amax(dim=1) > max_force
    return (is_open & ~touching).float()


def object_released(
    env: ManagerBasedRLEnv,
    std: float,
    command_name: str,
    rest_clearance: float,
    half_extent: float,
    open_position: float,
    gripper_cfg: SceneEntityCfg,
    sensor_names: tuple[str, ...] = ("contact_gripper", "contact_jaw"),
    max_force: float = 0.1,
) -> torch.Tensor:
    """:func:`object_placed` while the gripper is open and off the cube."""
    placed = object_placed(env, std, command_name, rest_clearance, half_extent)
    return placed * _released(env, open_position, gripper_cfg, sensor_names, max_force)


def gripper_open_command(
    env: ManagerBasedRLEnv,
    std: float,
    command_name: str,
    rest_clearance: float,
    half_extent: float,
    gripper_action_name: str = "gripper_action",
) -> torch.Tensor:
    """:func:`object_placed` times the gripper action clipped to [-1, 1], plus 1.

    Pulls the gripper command toward open (+1) while the cube rests on its target.
    """
    placed = object_placed(env, std, command_name, rest_clearance, half_extent)
    action = env.action_manager.get_term(gripper_action_name).raw_actions[:, 0]
    return placed * (action.clamp(-1.0, 1.0) + 1.0)


def arm_home(
    env: ManagerBasedRLEnv,
    std: float,
    home_std: float,
    command_name: str,
    rest_clearance: float,
    half_extent: float,
    open_position: float,
    gripper_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg,
    sensor_names: tuple[str, ...] = ("contact_gripper", "contact_jaw"),
    max_force: float = 0.1,
) -> torch.Tensor:
    """:func:`object_released` times ``1 - tanh(|q - q_default| / home_std)``.

    ``|q - q_default|`` is the norm [rad] over the ``asset_cfg`` joints.
    """
    robot: Articulation = env.scene[asset_cfg.name]
    q = robot.data.joint_pos.torch[:, asset_cfg.joint_ids]
    q_home = robot.data.default_joint_pos.torch[:, asset_cfg.joint_ids]
    home = 1.0 - torch.tanh(torch.norm(q - q_home, dim=1) / home_std)
    released = object_released(
        env,
        std,
        command_name,
        rest_clearance,
        half_extent,
        open_position,
        gripper_cfg,
        sensor_names,
        max_force,
    )
    return released * home
