# Copyright (c) 2024-2026, Ryan Donald
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Hold-start resets: episodes that begin with the cube pinched at its goal.

They let the policy visit, and learn to hold, the "arm and cube still at the goal"
state. The bank of states was recorded from a trained lift policy.

The reset event and the command class share per-env state on the env object: the
command manager resamples after the reset events run, so the event parks each hold
start's goal in ``env._hold_goal_b`` and the command applies it.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING

import torch
from isaaclab.utils.math import quat_apply, quat_mul
from isaaclab_tasks.core.lift.mdp.commands.pose_commands import ObjectUniformPoseCommand

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

# 4086 banked hold states.
HOLD_BANK_PATH = str(
    Path(__file__).resolve().parent.parent / "data" / "hold_bank_long14000.pt"
)


class HoldAwarePoseCommand(ObjectUniformPoseCommand):
    """ObjectUniformPoseCommand, except envs just put into a banked hold state get that
    state's goal."""

    def _resample_command(self, env_ids: Sequence[int]):
        super()._resample_command(env_ids)
        pending = getattr(self._env, "_hold_goal_b", None)
        if pending is None:
            return
        if isinstance(env_ids, slice):
            env_ids = torch.arange(self.num_envs, device=self.device)[env_ids]
        env_ids = torch.as_tensor(env_ids, device=self.device)
        goal = pending[env_ids]
        m = ~torch.isnan(goal[:, 0])
        if m.any():
            ids = env_ids[m]
            self.pose_command_b[ids, :3] = goal[m]
            pending[ids] = float("nan")
            # the action manager zeroes raw actions after the events, and the first
            # observation's action history reads them. The command resets after the
            # actions and before that observation, so restore "hold this pose" here.
            arm = self._env.action_manager.get_term("arm_action")
            grip = self._env.action_manager.get_term("gripper_action")
            arm._raw_actions[ids] = self._env._hold_raw[ids, :4]
            grip._raw_actions[ids] = self._env._hold_raw[ids, 4:]


def hold_start_reset(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor | slice,
    fraction: float,
    bank_path: str = HOLD_BANK_PATH,
):
    """Put ``fraction`` of the resetting envs into a banked state: cube lifted and
    pinched in the jaws within 2 cm of its goal, all velocities zero. Also primes the
    arm and gripper command buffers with that pose so the first (delayed) commands
    hold it instead of the previous episode's last command.

    Needs ``HoldAwarePoseCommand`` as the ``object_pose`` command class.
    """
    if not hasattr(env, "_hold_bank"):
        bank = torch.load(bank_path, map_location=env.device)
        robot = env.scene["robot"]
        assert bank["joint_names"] == list(robot.joint_names), bank["joint_names"]
        env._hold_bank = bank
        env._hold_goal_b = torch.full(
            (env.num_envs, 3), float("nan"), device=env.device
        )
        env._hold_raw = torch.zeros(env.num_envs, 5, device=env.device)
    bank = env._hold_bank
    if isinstance(env_ids, slice):
        env_ids = torch.arange(env.num_envs, device=env.device)[env_ids]
    env_ids = torch.as_tensor(env_ids, device=env.device)
    sel = env_ids[torch.rand(len(env_ids), device=env.device) < fraction]
    if len(sel) == 0:
        return
    idx = torch.randint(0, bank["joint_pos"].shape[0], (len(sel),), device=env.device)
    robot, obj = env.scene["robot"], env.scene["object"]

    q = bank["joint_pos"][idx]
    robot.write_joint_position_to_sim_index(position=q, env_ids=sel)
    robot.write_joint_velocity_to_sim_index(velocity=torch.zeros_like(q), env_ids=sel)
    robot.actuators.target_command.set_position_index(value=q, env_ids=sel)

    root_p, root_q = robot.data.root_pos_w.torch[sel], robot.data.root_quat_w.torch[sel]
    pos_w = root_p + quat_apply(root_q, bank["cube_pos_b"][idx])
    quat_w = quat_mul(root_q, bank["cube_quat_b"][idx])
    obj.write_root_pose_to_sim_index(
        root_pose=torch.cat([pos_w, quat_w], dim=-1), env_ids=sel
    )
    obj.write_root_velocity_to_sim_index(
        root_velocity=torch.zeros(len(sel), 6, device=env.device), env_ids=sel
    )

    # command buffers: action_manager.reset (after the events) primes each term's
    # delay buffer from its _processed_actions, so set those to "hold this pose".
    arm = env.action_manager.get_term("arm_action")
    lim = robot.data.soft_joint_pos_limits.torch[sel][:, arm._joint_ids]
    q_arm = q[:, arm._joint_ids]
    target = 200.0 * (q_arm - lim[..., 0]) / (lim[..., 1] - lim[..., 0]) - 100.0
    arm._processed_actions[sel] = target
    grip = env.action_manager.get_term("gripper_action")
    grip._processed_actions[sel] = grip.cfg.close_command_expr["gripper"]
    offset = torch.as_tensor(arm._offset, device=env.device)
    offset = offset[sel] if offset.dim() == 2 else offset
    env._hold_raw[sel, :4] = (target - offset) / arm.cfg.scale
    env._hold_raw[sel, 4] = -1.0

    env._hold_goal_b[sel] = bank["goal_b"][idx]
