# Copyright (c) 2024-2026, Ryan Donald
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Manager terms of the place task."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import warp as wp
from isaaclab.managers import ActionTerm, ActionTermCfg, CommandTerm, CommandTermCfg
from isaaclab.utils import configclass

from ryan_tasks.mdp.rebot_b601 import ArmCommand, GripperCommand, randomize_delay
from ryan_tasks.robots.rebot_b601 import (
    ARM_JOINTS,
    GRIPPER_BODY,
    JAW_JOINT,
    JAW_OPEN,
)
from ryan_tasks.robots.so_arm101 import MAX_SLEW_RAD_S

from . import params as p
from .logic import TERM_NAMES, PlaceLogic

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


class PlaceAction(ActionTerm):
    """Four normalized arm joints and a binary jaw, on one delay."""

    cfg: PlaceActionCfg

    def __init__(self, cfg: PlaceActionCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._asset = env.scene[cfg.asset_name]
        arm_ids = self._asset.find_joints(ARM_JOINTS)[0]
        jaw_ids = self._asset.find_joints(JAW_JOINT)[0]
        self._arm_ids = torch.tensor(arm_ids, device=self.device)
        self._arm_ids_wp = wp.array(arm_ids, dtype=wp.int32, device=self.device)
        self._jaw_ids_wp = wp.array(jaw_ids, dtype=wp.int32, device=self.device)
        data = self._asset.data
        self.arm = ArmCommand(
            data.soft_joint_pos_limits.torch[:, self._arm_ids].clone(),
            data.default_joint_pos.torch[:, self._arm_ids].clone(),
            data.joint_pos.torch[:, self._arm_ids],
            p.ACTION_SCALE,
            p.MAX_DELAY,
            MAX_SLEW_RAD_S * env.physics_dt,
        )
        self.gripper = GripperCommand(self.arm, JAW_OPEN)
        self._raw_actions = torch.zeros(env.num_envs, 5, device=self.device)

    @property
    def action_dim(self) -> int:
        return 5

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self.arm.desired

    def process_actions(self, actions: torch.Tensor):
        self._raw_actions[:] = actions
        self.arm.push(actions[:, :4])
        self.gripper.push(actions[:, 4:])

    def apply_actions(self):
        self.arm.ramp()
        targets = self._asset.actuators.target_command
        targets.set_position_index(value=self.arm.target, joint_ids=self._arm_ids_wp)
        targets.set_position_index(
            value=self.gripper.target, joint_ids=self._jaw_ids_wp
        )

    def reset(self, env_ids=None):
        ids = slice(None) if env_ids is None else env_ids
        self.arm.reset(ids, self._asset.data.joint_pos.torch[:, self._arm_ids])
        self.gripper.reset(ids)
        self._raw_actions[ids] = 0.0


@configclass
class PlaceActionCfg(ActionTermCfg):
    class_type: type = PlaceAction


def randomize_action_delay(env: ManagerBasedRLEnv, env_ids, low: int, high: int):
    randomize_delay(env.action_manager.get_term("arm").arm, env_ids, low, high)


def reset_cube(env: ManagerBasedRLEnv, env_ids):
    """Cube at rest, square to the world, anywhere in the spawn box."""
    cube = env.scene["object"]
    pose = torch.zeros(env.num_envs, 7, device=env.device)[env_ids]
    low = pose.new_tensor(p.CUBE_LOW)
    high = pose.new_tensor(p.CUBE_HIGH)
    pose[:, :2] = low + (high - low) * torch.rand_like(pose[:, :2])
    pose[:, 2] = p.CUBE_HALF
    pose[:, :3] += env.scene.env_origins[env_ids]
    pose[:, 6] = 1.0
    cube.write_root_pose_to_sim_index(root_pose=pose, env_ids=env_ids)
    velocity = torch.zeros_like(pose[:, :6])
    cube.write_root_velocity_to_sim_index(root_velocity=velocity, env_ids=env_ids)


class PlaceTarget(CommandTerm):
    """The table target and the carried latch, see :class:`PlaceLogic`."""

    cfg: PlaceTargetCfg

    def __init__(self, cfg: PlaceTargetCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.logic = PlaceLogic(env.num_envs, env.scene.env_origins)
        self._robot = env.scene["robot"]
        self._cube = env.scene["object"]
        self._gripper_id = self._robot.find_bodies(GRIPPER_BODY)[0][0]
        self._arm_ids = torch.tensor(
            self._robot.find_joints(ARM_JOINTS)[0], device=self.device
        )
        self._jaw_id = self._robot.find_joints(JAW_JOINT)[0][0]
        self._vel_ids = torch.cat(
            (self._arm_ids, torch.tensor([self._jaw_id], device=self.device))
        )
        self._step = -1

    @property
    def command(self) -> torch.Tensor:
        return self.logic.target_b

    def terms(self) -> torch.Tensor:
        """The reward terms; shared, computed once per step."""
        if self._step != self._env.common_step_counter:
            self._step = self._env.common_step_counter
            data = self._robot.data
            manager = self._env.action_manager
            joint_pos = data.joint_pos.torch
            limits = data.soft_joint_pos_limits.torch[:, self._arm_ids]
            arm = joint_pos[:, self._arm_ids]
            self.logic.write_terms(
                data.body_link_pose_w.torch[:, self._gripper_id],
                self._cube.data.root_link_pose_w.torch,
                joint_pos[:, self._jaw_id],
                arm,
                data.default_joint_pos.torch[:, self._arm_ids],
                data.joint_vel.torch[:, self._vel_ids],
                torch.minimum(arm - limits[..., 0], limits[..., 1] - arm),
                manager.action,
                manager.prev_action,
            )
        return self.logic.terms

    def reset(self, env_ids=None) -> dict:
        ids = slice(None) if env_ids is None else env_ids
        self.logic.reset(ids, self._cube.data.root_pos_w.torch)
        return {}

    def compute(self, dt: float):
        pass

    def _update_metrics(self):
        pass

    def _resample_command(self, env_ids):
        pass

    def _update_command(self):
        pass


@configclass
class PlaceTargetCfg(CommandTermCfg):
    class_type: type = PlaceTarget
    resampling_time_range: tuple[float, float] = (1e9, 1e9)


def _target(env: ManagerBasedRLEnv) -> PlaceTarget:
    return env.command_manager.get_term("place_target")


def reward_term(env: ManagerBasedRLEnv, name: str) -> torch.Tensor:
    """One reward term, from the shared computation."""
    return _target(env).terms()[:, TERM_NAMES.index(name)]


def joint_pos_normalized(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Arm joint positions from the default pose, in normalized units."""
    command = env.action_manager.get_term("arm").arm
    data = env.scene["robot"].data
    joint_pos = data.joint_pos.torch[:, _target(env)._arm_ids]
    return 200.0 * (joint_pos - command.default_pos) / (command.upper - command.lower)


def jaw_pos_normalized(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Jaw position, 0 closed and 100 open."""
    jaw = env.scene["robot"].data.joint_pos.torch[:, _target(env)._jaw_id]
    return 100.0 * jaw.unsqueeze(1) / JAW_OPEN


def joint_vel(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Arm and jaw velocities."""
    return env.scene["robot"].data.joint_vel.torch[:, _target(env)._vel_ids]


def cube_position(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Cube position in the root frame."""
    return env.scene["object"].data.root_pos_w.torch - env.scene.env_origins


def carried(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Whether the cube has been carried this episode."""
    return _target(env).logic.carried.float().unsqueeze(1)


def last_action(env: ManagerBasedRLEnv) -> torch.Tensor:
    """The last action as executed: clamped arm targets and the jaw's sign."""
    term = env.action_manager.get_term("arm")
    return torch.cat((term.arm.executed_action(), term.gripper.sign()), dim=1)


def cube_flew_off(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Ends an env whose physics diverged, before its values reach the normalizer."""
    cube = cube_position(env)
    vel = env.scene["robot"].data.joint_vel.torch
    finite = torch.isfinite(cube).all(dim=1) & torch.isfinite(vel).all(dim=1)
    return ~finite | (torch.norm(cube, dim=1) > p.MAX_CUBE_RANGE)
