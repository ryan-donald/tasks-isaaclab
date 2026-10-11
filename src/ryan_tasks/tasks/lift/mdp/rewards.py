# Copyright (c) 2024-2026, Ryan Donald
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Reward terms for the lift task."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor
from isaaclab.utils.math import (
    combine_frame_transforms,
    quat_apply,
)

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def object_is_lifted(
    env: ManagerBasedRLEnv,
    minimal_height: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """1.0 when the object centre is above ``minimal_height``."""
    object: RigidObject = env.scene[object_cfg.name]
    return torch.where(object.data.root_pos_w.torch[:, 2] > minimal_height, 1.0, 0.0)


def _object_clearance(
    object: RigidObject,
    half_extent: float | tuple[float, float, float],
    table_z: float,
) -> torch.Tensor:
    """Height of the box's lowest vertex above the table.

    The lowest vertex is the support point along world -z, ``center_z - (hx|R20| +
    hy|R21| + hz|R22|)``, using the third row of the rotation matrix. A box tipped
    onto an edge still reads ~0. ``half_extent`` is a scalar for a cube or
    ``(hx, hy, hz)`` for a cuboid.
    """
    # quaternion (x, y, z, w).
    quat = object.data.root_state_w.torch[:, 3:7]
    x, y, z, w = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
    # third row of the rotation matrix: world-z components of the box's local axes.
    r20 = 2.0 * (x * z - w * y)
    r21 = 2.0 * (y * z + w * x)
    r22 = 1.0 - 2.0 * (x * x + y * y)
    if isinstance(half_extent, (tuple, list)):
        hx, hy, hz = half_extent
    else:
        hx = hy = hz = half_extent
    span = hx * r20.abs() + hy * r21.abs() + hz * r22.abs()
    lowest_z = object.data.root_pos_w.torch[:, 2] - span
    return lowest_z - table_z


def object_lift_shaped(
    env: ManagerBasedRLEnv,
    std: float,
    half_extent: float | tuple[float, float, float] = 0.015,
    table_z: float = 0.0,
    reach_std: float = 0.1,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=["gripper_link"]),
    ee_offset: tuple[float, float, float] = (0.015, 0.0, -0.09),
) -> torch.Tensor:
    """Dense lift reward, gated on the gripper being at the cube.

    ``tanh(clearance / std)`` of the cube's lowest vertex above the table, times the
    reach kernel ``1 - tanh(d_ee / reach_std)``. Tipping the cube onto an edge earns
    nothing. ``reach_std`` should match ``reaching_object``'s ``std``.
    """
    object: RigidObject = env.scene[object_cfg.name]
    clearance = _object_clearance(object, half_extent, table_z).clamp(min=0.0)
    lift = torch.tanh(clearance / std)
    grasp_gate = object_ee_distance(env, reach_std, object_cfg, ee_cfg, ee_offset)
    return lift * grasp_gate


def object_lifted_clearance(
    env: ManagerBasedRLEnv,
    clearance: float,
    half_extent: float | tuple[float, float, float] = 0.015,
    table_z: float = 0.0,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """1.0 when the cube's lowest vertex is more than ``clearance`` metres above
    the table."""
    object: RigidObject = env.scene[object_cfg.name]
    return (_object_clearance(object, half_extent, table_z) > clearance).float()


def object_grasped(
    env: ManagerBasedRLEnv,
    grasp_std: float = 0.03,
    gripper_open: float = 0.75,
    gripper_cfg: SceneEntityCfg = SceneEntityCfg("robot", joint_names=["gripper"]),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=["gripper_link"]),
    ee_offset: tuple[float, float, float] = (0.015, 0.0, -0.09),
) -> torch.Tensor:
    """Reward closing the jaw with the cube at the grasp point.

    (cube at the grasp point) * (jaw closing): ``1 - tanh(d_ee / grasp_std)`` with a
    tight ``grasp_std``, times ``(gripper_open - gripper_pos) / gripper_open``
    clamped to [0, 1]. ``gripper_cfg`` must be passed through ``params`` so the
    manager resolves its ``joint_ids``.
    """
    robot: Articulation = env.scene[gripper_cfg.name]
    gripper_pos = robot.data.joint_pos.torch[:, gripper_cfg.joint_ids[0]]
    closing = ((gripper_open - gripper_pos) / gripper_open).clamp(0.0, 1.0)
    in_jaw = object_ee_distance(env, grasp_std, object_cfg, ee_cfg, ee_offset)
    return in_jaw * closing


def object_ee_distance(
    env: ManagerBasedRLEnv,
    std: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=["gripper_link"]),
    ee_offset: tuple[float, float, float] = (0.015, 0.0, -0.09),
) -> torch.Tensor:
    """Reward reaching the object, ``1 - tanh(distance / std)``.

    The end-effector point is ``ee_offset`` in the ``ee_cfg`` body's frame.
    ``ee_cfg`` must be passed through ``params`` so the manager resolves its
    ``body_ids``.
    """
    object: RigidObject = env.scene[object_cfg.name]
    robot: Articulation = env.scene[ee_cfg.name]
    body_id = ee_cfg.body_ids[0]
    # end-effector position: gripper_link body pose + body-frame offset -> world.
    ee_pos_w = robot.data.body_pos_w.torch[:, body_id, :]
    # quaternion (x, y, z, w).
    ee_quat_w = robot.data.body_quat_w.torch[:, body_id, :]
    offset = ee_pos_w.new_tensor(ee_offset).expand_as(ee_pos_w)
    ee_w = ee_pos_w + quat_apply(ee_quat_w, offset)
    # distance of the end-effector to the object: (num_envs,)
    cube_pos_w = object.data.root_pos_w.torch
    object_ee_distance = torch.norm(cube_pos_w - ee_w, dim=1)

    return 1 - torch.tanh(object_ee_distance / std)


def object_goal_distance(
    env: ManagerBasedRLEnv,
    std: float,
    minimal_height: float,
    command_name: str,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """``1 - tanh(distance to goal / std)`` once the object is above
    ``minimal_height``."""
    robot: RigidObject = env.scene[robot_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]
    command = env.command_manager.get_command(command_name)
    # compute the desired position in the world frame
    des_pos_b = command[:, :3]
    des_pos_w, _ = combine_frame_transforms(
        robot.data.root_state_w.torch[:, :3],
        robot.data.root_state_w.torch[:, 3:7],
        des_pos_b,
    )
    # distance of the object to the goal: (num_envs,)
    distance = torch.norm(des_pos_w - object.data.root_pos_w.torch[:, :3], dim=1)
    # rewarded only if the object is lifted above the threshold.
    return (object.data.root_pos_w.torch[:, 2] > minimal_height) * (
        1 - torch.tanh(distance / std)
    )


def action_l2_term(
    env: ManagerBasedRLEnv, action_name: str = "arm_action"
) -> torch.Tensor:
    """Squared L2 norm of one action term's raw actions.

    Unlike ``action_l2``, this leaves out the binary gripper, whose raw action sign
    is its open/close decision.
    """
    return torch.sum(
        torch.square(env.action_manager.get_term(action_name).raw_actions), dim=1
    )


def action_past_limits(
    env: ManagerBasedRLEnv,
    arm_action_name: str = "arm_action",
) -> torch.Tensor:
    """How far the arm command goes past the joint limits, in action units.

    The normalized target beyond +/-100, divided by the action scale, summed over
    joints. Zero for any executable command. The gripper is left out: its
    magnitude keeps its sampled sign stable under exploration noise.
    """
    arm = env.action_manager.get_term(arm_action_name)
    target = arm.cfg.scale * arm.raw_actions + arm._offset
    return ((target.abs() - 100.0).clamp(min=0.0) / arm.cfg.scale).sum(dim=1)


def arm_command_rate(
    env: ManagerBasedRLEnv,
    arm_action_name: str = "arm_action",
) -> torch.Tensor:
    """Squared step-to-step change of the executed arm command, in action units.

    Uses the normalized arm target after the joint-limit clamp, so the gripper and
    commands past the limits do not count.
    """
    am = env.action_manager
    arm = am.get_term(arm_action_name)
    start = 0
    for name, dim in zip(am.active_terms, am.action_term_dim):
        if name == arm_action_name:
            break
        start += dim
    prev_raw = am.prev_action[:, start : start + arm.action_dim]
    cur = (arm.cfg.scale * arm.raw_actions + arm._offset).clamp(-100.0, 100.0)
    prev = (arm.cfg.scale * prev_raw + arm._offset).clamp(-100.0, 100.0)
    return ((cur - prev) / arm.cfg.scale).square().sum(dim=1)


def settled_at_goal(
    env: ManagerBasedRLEnv,
    goal_std: float,
    rate_scale: float,
    minimal_height: float,
    command_name: str,
    arm_action_name: str = "arm_action",
) -> torch.Tensor:
    """Bonus for holding a still arm command at the goal while the cube is lifted.

    ``(1 - tanh(goal distance / goal_std)) * (1 - tanh(|arm command change| /
    rate_scale))``. ``rate_scale`` is in action units and must sit above the
    command change caused by exploration noise (about 2.8 * std for four joints).
    """
    rate = arm_command_rate(env, arm_action_name).sqrt()
    return object_goal_distance(env, goal_std, minimal_height, command_name) * (
        1.0 - torch.tanh(rate / rate_scale)
    )


def object_pinched(
    env: ManagerBasedRLEnv,
    min_force: float = 0.5,
    jaw_sensor: str = "contact_jaw",
    gripper_sensor: str = "contact_gripper",
) -> torch.Tensor:
    """1.0 when both jaws press on the cube from opposite sides.

    Both sensors filter on the cube, so ``force_matrix_w`` is the cube's force on
    each jaw. Both must exceed ``min_force`` and point against each other.
    """
    n = env.num_envs
    fj = env.scene[jaw_sensor].data.force_matrix_w.torch.reshape(n, -1, 3).sum(1)
    fg = env.scene[gripper_sensor].data.force_matrix_w.torch.reshape(n, -1, 3).sum(1)
    pinch = (
        (fj.norm(dim=1) > min_force)
        & (fg.norm(dim=1) > min_force)
        & ((fj * fg).sum(dim=1) < 0.0)
    )
    return pinch.float()


def gripper_level(
    env: ManagerBasedRLEnv,
    minimal_height: float,
    ee_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=["gripper_link"]),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Reward carrying with the fingers level, once the cube is lifted.

    Scores ``1 - |z|`` of the finger direction (``gripper_link``'s local -z): 1
    level, 0 pointing straight up or down. Zero until the cube centre is above
    ``minimal_height``. ``ee_cfg`` must be passed through ``params`` so the manager
    resolves its ``body_ids``.
    """
    robot: Articulation = env.scene[ee_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]
    q = robot.data.body_quat_w.torch[:, ee_cfg.body_ids[0]]
    down = q.new_tensor((0.0, 0.0, -1.0)).expand(env.num_envs, 3)
    fingers_z = quat_apply(q, down)[:, 2]
    lifted = object.data.root_pos_w.torch[:, 2] > minimal_height
    return lifted * (1.0 - fingers_z.abs())


def joint_near_limits(
    env: ManagerBasedRLEnv,
    band: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalty ramping linearly from 0 at ``band`` radians from a joint limit to 1
    at the limit, summed over the selected joints.

    ``asset_cfg`` must be passed through ``params`` so the manager resolves its
    ``joint_ids``.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    pos = asset.data.joint_pos.torch[:, asset_cfg.joint_ids]
    lim = asset.data.soft_joint_pos_limits.torch[:, asset_cfg.joint_ids]
    margin = torch.minimum(pos - lim[..., 0], lim[..., 1] - pos)
    return (1.0 - margin / band).clamp(0.0, 1.0).sum(dim=1)


_GRIPPER_HULLS: dict[str, "torch.Tensor"] = {}


def _gripper_hull(body_name: str, device) -> "torch.Tensor":
    """Load and cache the gripper collision hull vertices for ``body_name``."""
    key = f"{body_name}:{device}"
    if key not in _GRIPPER_HULLS:
        import numpy as np

        from ryan_tasks.robots.so_arm101 import URDF_ASSETS_DATA_DIR

        data = np.load(f"{URDF_ASSETS_DATA_DIR}/so101_gripper_hulls.npz")
        _GRIPPER_HULLS[key] = torch.as_tensor(
            data[body_name], dtype=torch.float32, device=device
        )
    return _GRIPPER_HULLS[key]


def gripper_ground_force_penalty(
    env: ManagerBasedRLEnv,
    sensor_names: tuple[str, ...] = ("contact_gripper", "contact_jaw"),
    deadband: float = 1.0,
    max_force: float = 50.0,
) -> torch.Tensor:
    """Penalty in [0, 1] for pressing the gripper into the surface.

    The ground force is the net contact force minus the cube's (the sensors filter
    on the cube; the ground plane cannot be filtered against), as vectors. It is
    scaled to [0, 1] between ``deadband`` and ``max_force``.
    """
    total = None
    for name in sensor_names:
        sensor: ContactSensor = env.scene[name]
        net = sensor.data.net_forces_w.torch  # (N, B, 3)
        matrix = sensor.data.force_matrix_w
        if matrix is not None:
            # sum over the filtered partners, then remove them from the net force.
            net = net - matrix.torch.sum(dim=2)
        # strongest ground contact across the sensor's bodies: (N,)
        f = net.norm(dim=-1).amax(dim=1)
        total = f if total is None else torch.maximum(total, f)

    scaled = (total - deadband) / (max_force - deadband)
    return torch.clamp(scaled, min=0.0, max=1.0)
