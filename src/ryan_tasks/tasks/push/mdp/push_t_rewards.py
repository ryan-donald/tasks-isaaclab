# Copyright (c) 2024-2026, Ryan Donald
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Push-T reward terms: yaw error, yaw progress, pose success, and tip speed near
the block. The position terms come from the cube push."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from isaaclab.assets import RigidObject
from isaaclab.managers import ManagerTermBase, SceneEntityCfg
from isaaclab.utils.math import (
    quat_apply,
    quat_apply_inverse,
    quat_conjugate,
    quat_mul,
)

from ryan_tasks.tasks.push.mdp.rewards import (
    _object_goal_distance_xy,
    _quadratic_cost,
)

# the T's two collision boxes in its body frame, (centre, half-extent), from
# assets/t_block_geom.usda.
T_BOXES = (
    ((0.0, 0.0123529, 0.015), (0.03, 0.009, 0.015)),
    ((0.0, -0.0176471, 0.015), (0.009, 0.021, 0.015)),
)

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _yaw_from_quat(quat: torch.Tensor) -> torch.Tensor:
    """Yaw about z from an ``(x, y, z, w)`` quaternion. Shape (N,), in (-pi, pi]."""
    x, y, z, w = quat.unbind(-1)
    return torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def object_yaw_in_root_frame(robot: RigidObject, obj: RigidObject) -> torch.Tensor:
    """The object's yaw expressed in the robot root frame. Shape (N,)."""
    rel = quat_mul(
        quat_conjugate(robot.data.root_quat_w.torch), obj.data.root_quat_w.torch
    )
    return _yaw_from_quat(rel)


def goal_yaw_in_root_frame(command: torch.Tensor) -> torch.Tensor:
    """The commanded goal's yaw. Shape (N,).

    The command is ``(x, y, z, qx, qy, qz, qw)`` in the robot root frame.
    """
    return _yaw_from_quat(command[:, 3:7])


def object_yaw_error(
    env: ManagerBasedRLEnv,
    command_name: str = "object_pose",
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Absolute yaw error between the block and the goal, in rad, wrapped to [0, pi].

    The T has no rotational symmetry, so the error needs no folding.
    """
    robot: RigidObject = env.scene[robot_cfg.name]
    obj: RigidObject = env.scene[object_cfg.name]
    command = env.command_manager.get_command(command_name)
    error = goal_yaw_in_root_frame(command) - object_yaw_in_root_frame(robot, obj)
    return torch.abs(torch.atan2(torch.sin(error), torch.cos(error)))


def object_pose_at_goal(
    env: ManagerBasedRLEnv,
    dist_threshold: float,
    yaw_threshold: float,
    command_name: str = "object_pose",
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """1.0 while the block is within both the distance and the yaw threshold of the
    goal: the task's success condition."""
    distance = _object_goal_distance_xy(env, command_name, robot_cfg, object_cfg)
    yaw = object_yaw_error(env, command_name, robot_cfg, object_cfg)
    return ((distance < dist_threshold) & (yaw < yaw_threshold)).float()


class object_yaw_progress(ManagerTermBase):
    """Rate at which the yaw error closes, in rad/s.

    The yaw counterpart of ``object_goal_progress``: summed over an episode it is
    ``yaw_0 - yaw_final``, independent of time, and the weight is reward per radian.
    The first step after a reset scores zero and the rate is clamped to
    ``max_closing_rate``.
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
        max_closing_rate: float = 8.0,
        robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
        object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ) -> torch.Tensor:
        e = object_yaw_error(env, command_name, robot_cfg, object_cfg)
        closing = (self._prev - e) / self._dt
        closing = torch.where(self._has_prev, closing, torch.zeros_like(closing))
        self._prev = e.clone()
        self._has_prev[:] = True
        return torch.clamp(closing, min=-max_closing_rate, max=max_closing_rate)


def tip_speed_near_object_penalty(
    env: ManagerBasedRLEnv,
    ee_cfg: SceneEntityCfg,
    ee_offset: tuple[float, float, float] = (0.015, 0.0, -0.098),
    radius: float = 0.03,
    ref_speed: float = 0.1,
    max_cost: float = 100.0,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Quadratic penalty on the gripper tip's speed near the T.

    Gated linearly from 1 at the T's surface to 0 at ``radius``, using the tip's
    signed distance to the T's two collision boxes. ``ee_cfg`` must be passed through
    ``params`` so the manager resolves its ``body_ids``.
    """
    robot = env.scene[ee_cfg.name]
    obj: RigidObject = env.scene[object_cfg.name]
    bid = ee_cfg.body_ids
    pos = robot.data.body_pos_w.torch[:, bid][:, 0]
    # quaternion (x, y, z, w).
    quat = robot.data.body_quat_w.torch[:, bid][:, 0]
    r = quat_apply(quat, pos.new_tensor(ee_offset).expand_as(pos))
    tip = pos + r
    tip_vel = robot.data.body_lin_vel_w.torch[:, bid][:, 0] + torch.cross(
        robot.data.body_ang_vel_w.torch[:, bid][:, 0], r, dim=-1
    )

    # tip distance to the T, in the T's frame, via the box signed-distance function.
    p = quat_apply_inverse(obj.data.root_quat_w.torch, tip - obj.data.root_pos_w.torch)
    dist = None
    for c, h in T_BOXES:
        q = (p - p.new_tensor(c)).abs() - p.new_tensor(h)
        d = q.clamp(min=0.0).norm(dim=-1) + q.max(dim=-1).values.clamp(max=0.0)
        dist = d if dist is None else torch.minimum(dist, d)

    gate = (1.0 - dist / radius).clamp(0.0, 1.0)
    return gate * _quadratic_cost(tip_vel.norm(dim=-1), ref_speed, max_cost)
