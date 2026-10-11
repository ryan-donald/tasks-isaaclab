# Copyright (c) 2024-2026, Ryan Donald
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Reward terms for the push tasks."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from isaaclab.assets import RigidObject
from isaaclab.managers import ManagerTermBase, SceneEntityCfg
from isaaclab.utils.math import combine_frame_transforms, quat_apply_inverse

from ryan_tasks.tasks.lift.mdp.rewards import (
    _gripper_hull,
)

if TYPE_CHECKING:
    from isaaclab.assets import Articulation
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.sensors import ContactSensor


def _object_goal_distance_xy(
    env: ManagerBasedRLEnv,
    command_name: str,
    robot_cfg: SceneEntityCfg,
    object_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Planar (xy) distance from the object to the commanded goal, in metres."""
    robot: RigidObject = env.scene[robot_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]
    command = env.command_manager.get_command(command_name)
    # the command is in the robot root frame.
    des_pos_w, _ = combine_frame_transforms(
        robot.data.root_state_w.torch[:, :3],
        robot.data.root_state_w.torch[:, 3:7],
        command[:, :3],
    )
    return torch.norm(des_pos_w[:, :2] - object.data.root_pos_w.torch[:, :2], dim=1)


def object_at_goal(
    env: ManagerBasedRLEnv,
    threshold: float,
    command_name: str,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """1.0 while the object is within ``threshold`` metres of the goal (planar)."""
    distance = _object_goal_distance_xy(env, command_name, robot_cfg, object_cfg)
    return (distance < threshold).float()


def _quadratic_cost(value: torch.Tensor, ref: float, max_cost: float) -> torch.Tensor:
    """``(value / ref) ** 2``, clamped at ``max_cost``.

    Quadratic, so a faster or harder motion always costs more: over a fixed distance
    ``d`` at speed ``v`` the cost integrates to ``v * d``. ``max_cost`` is set far
    outside the normal range, only to keep outliers from destabilizing the value
    function.
    """
    return torch.clamp((value / ref) ** 2, max=max_cost)


def gripper_downforce_penalty(
    env: ManagerBasedRLEnv,
    sensor_names: tuple[str, ...] = ("contact_gripper", "contact_jaw"),
    ref_force: float = 5.0,
    max_cost: float = 400.0,
) -> torch.Tensor:
    """Quadratic penalty on the gripper's downward contact force, on the cube or the
    table.

    Only the vertical component counts; the horizontal component is the push.
    ``net_forces_w`` is the force on the gripper, so +z is the gripper pressing down.
    """
    total = None
    for name in sensor_names:
        sensor: ContactSensor = env.scene[name]
        f = sensor.data.net_forces_w.torch[..., 2].clamp(min=0.0).amax(dim=1)
        total = f if total is None else torch.maximum(total, f)
    return _quadratic_cost(total, ref_force, max_cost)


def object_lin_speed_penalty(
    env: ManagerBasedRLEnv,
    ref_speed: float = 0.1,
    max_cost: float = 100.0,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Quadratic penalty on the cube's planar speed, at all times (including at the
    goal)."""
    obj: RigidObject = env.scene[object_cfg.name]
    speed = obj.data.root_lin_vel_w.torch[:, :2].norm(dim=-1)
    return _quadratic_cost(speed, ref_speed, max_cost)


def object_ang_speed_penalty(
    env: ManagerBasedRLEnv,
    ref_speed: float = 5.0,
    max_cost: float = 100.0,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Quadratic penalty on the cube's angular speed."""
    obj: RigidObject = env.scene[object_cfg.name]
    speed = obj.data.root_ang_vel_w.torch.norm(dim=-1)
    return _quadratic_cost(speed, ref_speed, max_cost)


def gripper_surface_proximity_penalty(
    env: ManagerBasedRLEnv,
    body_cfg: SceneEntityCfg,
    clearance: float = 0.015,
    surface_z: float = 0.0,
    max_cost: float = 25.0,
) -> torch.Tensor:
    """Quadratic penalty on the gripper approaching the surface.

    0 at or above ``clearance``, 1 at the surface, and growing below it, so deeper is
    always worse. The lowest point of each body is found exactly from its collision
    hull vertices (``so101_gripper_hulls.npz``). ``body_cfg`` must be passed through
    ``params`` so the manager resolves its ``body_ids``.
    """
    robot: Articulation = env.scene[body_cfg.name]
    lowest = None
    for name, bid in zip(body_cfg.body_names, body_cfg.body_ids):
        pos = robot.data.body_pos_w.torch[:, bid, :]
        # quaternion (x, y, z, w).
        quat = robot.data.body_quat_w.torch[:, bid, :]
        verts = _gripper_hull(name, pos.device)
        up_b = quat_apply_inverse(quat, pos.new_tensor((0.0, 0.0, 1.0)).expand_as(pos))
        z = pos[:, 2] + (up_b @ verts.t()).min(dim=1).values
        lowest = z if lowest is None else torch.minimum(lowest, z)

    # 0 at or above the clearance band, 1 at the surface, >1 below it.
    intrusion = ((surface_z + clearance) - lowest) / clearance
    return torch.clamp(intrusion.clamp(min=0.0) ** 2, max=max_cost)


class object_goal_progress(ManagerTermBase):
    """Rate at which the cube closes on the goal, in m/s.

    Summed over an episode this is ``d_0 - d_final``, independent of how long the
    push took. Dividing by ``dt`` cancels the reward manager's ``dt`` scaling, so the
    weight is reward per metre. The first step after a reset scores zero, and the
    closing speed is clamped to ``max_closing_speed`` so a launch cannot dump an
    episode's reward into one step.
    """

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self._dt = env.step_dt
        self._prev = torch.zeros(env.num_envs, device=env.device)
        self._has_prev = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    def reset(self, env_ids=None) -> None:
        if env_ids is None:
            self._has_prev[:] = False
        else:
            self._has_prev[env_ids] = False

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        command_name: str = "object_pose",
        max_closing_speed: float = 1.0,
        robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
        object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ) -> torch.Tensor:
        d = _object_goal_distance_xy(env, command_name, robot_cfg, object_cfg)
        closing = (self._prev - d) / self._dt
        closing = torch.where(self._has_prev, closing, torch.zeros_like(closing))
        self._prev = d.clone()
        self._has_prev[:] = True
        return torch.clamp(closing, min=-max_closing_speed, max=max_closing_speed)
