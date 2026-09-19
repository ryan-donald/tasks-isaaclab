# Copyright (c) 2024-2026, Ryan Donald
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Custom action terms for normalized space [-100, +100]."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import MISSING
from typing import TYPE_CHECKING

import torch
import warp as wp
from isaaclab.assets.articulation import Articulation
from isaaclab.managers.action_manager import ActionTerm, ActionTermCfg
from isaaclab.utils.configclass import configclass

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


@configclass
class NormalizedJointPositionActionCfg(ActionTermCfg):
    # configuration for position controlled joint action term matching lerobot.

    class_type: type[ActionTerm] = MISSING
    joint_names: list[str] = MISSING
    scale: float = 1.0

    # offset for default position, i.e. 0 in normalized space
    offset: float = 0.0
    use_default_offset: bool = False
    preserve_order: bool = False
    delay_steps: int = 0

    # servo stiction band, normalized units. a goal this close to the current
    # position is replaced by it. 0 disables.
    deadband: float = 0.0

    # servo speed limit, rad/s (the STS3215's Goal_Velocity). 0 disables.
    max_slew_rad_s: float = 0.0

    # servo setpoint acceleration limit, rad/s^2. 0 disables.
    max_accel_rad_s2: float = 0.0


class NormalizedJointPositionAction(ActionTerm):
    # action term that receives actions in normalized space [-100, 100] and
    # translates them to radians.

    cfg: NormalizedJointPositionActionCfg

    def __init__(self, cfg: NormalizedJointPositionActionCfg, env: ManagerBasedEnv):
        # initialize the action term
        super().__init__(cfg, env)

        # resolve the joints over which the action term is applied
        self._asset: Articulation = env.scene[cfg.asset_name]
        self._joint_ids, self._joint_names = self._asset.find_joints(cfg.joint_names)
        self._num_joints = len(self._joint_ids)
        # device copy of the joint indices, as a python list syncs on every write.
        self._joint_ids_wp = wp.array(
            self._joint_ids, dtype=wp.int32, device=self.device
        )

        # log info for debugging
        print(
            f"[NormalizedJointPositionAction] Resolved joint names for"
            f" {self.cfg.asset_name}: {self._joint_names} [{self._joint_ids}]"
        )

        # create tensors for raw and processed actions
        self._raw_actions = torch.zeros(
            env.num_envs, self.action_dim, device=self.device
        )
        self._processed_actions = torch.zeros_like(self._raw_actions)

        # get joint limits for conversion
        self._joint_limits = self._asset.data.soft_joint_pos_limits.torch[
            :, self._joint_ids, :
        ].clone()

        # calculate offset in normalized space
        if cfg.use_default_offset:
            # convert default joint positions to normalized space [-100, 100]
            default_pos = self._asset.data.default_joint_pos.torch[:, self._joint_ids]
            lower = self._joint_limits[:, :, 0]
            upper = self._joint_limits[:, :, 1]
            self._offset = 200.0 * (default_pos - lower) / (upper - lower) - 100.0
        else:
            self._offset = cfg.offset

        print(
            f"[NormalizedJointPositionAction] Using offset"
            f" (normalized [-100,+100] space): {self._offset}"
        )

        # ramped command target. apply_actions runs every physics step.
        self._prev_cmd = self._asset.data.joint_pos.torch[:, self._joint_ids].clone()
        self._prev_vel = torch.zeros_like(self._prev_cmd)
        self._slew_dt = env.physics_dt

        # action delay buffer, per env action delay.
        self._max_delay = cfg.delay_steps
        if self._max_delay > 0:
            self._action_delay_buf = torch.zeros(
                self._max_delay + 1, env.num_envs, self.action_dim, device=self.device
            )
            self._delay_per_env = torch.full(
                (env.num_envs,), self._max_delay, dtype=torch.long, device=self.device
            )
            self._env_idx = torch.arange(env.num_envs, device=self.device)
            print(
                f"[NormalizedJointPositionAction] Max action delay:"
                f" {self._max_delay} step(s)"
            )

    @property
    def action_dim(self) -> int:
        return self._num_joints

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions

    def process_actions(self, actions: torch.Tensor):
        # store the raw actions
        self._raw_actions[:] = actions
        # apply scaling and offset in normalized space
        self._processed_actions = self.cfg.scale * self._raw_actions + self._offset

        if self._max_delay > 0:
            # newest action -> slot [-1]; buffer rolls so older actions shift down
            self._action_delay_buf = torch.roll(
                self._action_delay_buf, shifts=-1, dims=0
            )
            self._action_delay_buf[-1] = self._processed_actions
            # per-env read index: delay d reads slot (max_delay - d)
            read_idx = self._max_delay - self._delay_per_env  # (num_envs,)
            self._processed_actions = self._action_delay_buf[
                read_idx, self._env_idx
            ].clone()

    def apply_actions(self):
        # clamp to [-100, 100] range
        normalized_clamped = torch.clamp(self._processed_actions, -100.0, 100.0)

        lower = self._joint_limits[:, :, 0]
        upper = self._joint_limits[:, :, 1]

        if self.cfg.deadband > 0.0:
            # compare in normalized space, where the deadband is defined
            current = self._asset.data.joint_pos.torch[:, self._joint_ids]
            current_norm = 200.0 * (current - lower) / (upper - lower) - 100.0
            inside = (normalized_clamped - current_norm).abs() < self.cfg.deadband
            normalized_clamped = torch.where(inside, current_norm, normalized_clamped)

        # convert from normalized [-100, 100] to radians
        radians = (normalized_clamped + 100.0) / 200.0 * (upper - lower) + lower

        if self.cfg.max_accel_rad_s2 > 0.0:
            dt = self._slew_dt
            amax = self.cfg.max_accel_rad_s2
            err = radians - self._prev_cmd
            v_target = torch.sqrt(2.0 * amax * err.abs())
            if self.cfg.max_slew_rad_s > 0.0:
                v_target = v_target.clamp(max=self.cfg.max_slew_rad_s)
            v_target = torch.sign(err) * v_target
            self._prev_vel += (v_target - self._prev_vel).clamp(-amax * dt, amax * dt)
            step = self._prev_vel * dt
            snap = step.abs() > err.abs()
            radians = self._prev_cmd + torch.where(snap, err, step)
            self._prev_vel = torch.where(snap, err / dt, self._prev_vel)
            self._prev_cmd = radians.clone()
        elif self.cfg.max_slew_rad_s > 0.0:
            # ramp the command, like the servo's own setpoint. clamping to the joint
            # instead caps the PD error and the soft arm crawls.
            step = self.cfg.max_slew_rad_s * self._slew_dt
            radians = self._prev_cmd + (radians - self._prev_cmd).clamp(-step, step)
            self._prev_cmd = radians.clone()

        # apply position commands
        self._asset.set_joint_position_target_index(
            target=radians, joint_ids=self._joint_ids_wp
        )

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        if self.cfg.max_slew_rad_s > 0.0 or self.cfg.max_accel_rad_s2 > 0.0:
            # restart the ramp from the current joint position
            ids = slice(None) if env_ids is None else env_ids
            self._prev_cmd[ids] = self._asset.data.joint_pos.torch[:, self._joint_ids][ids]
            self._prev_vel[ids] = 0.0
        if self._max_delay > 0 and env_ids is not None:
            for slot in range(self._max_delay + 1):
                self._action_delay_buf[slot, env_ids] = self._processed_actions[env_ids]
        self._raw_actions[env_ids] = 0.0


def randomize_action_delay(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    action_name: str,
    min_delay: int,
    max_delay: int,
) -> None:
    """Sample a fresh per-env action delay in [min_delay, max_delay] on reset.

    ``max_delay`` must not exceed the action term's ``delay_steps`` (the buffer depth).
    The action term must expose a ``_delay_per_env`` buffer (``delay_steps > 0``); if it
    does not, the event is a no-op so a misconfigured term can't crash the reset.
    """
    term: NormalizedJointPositionAction = env.action_manager.get_term(action_name)
    if getattr(term, "_delay_per_env", None) is None:
        return
    delays = torch.randint(
        min_delay, max_delay + 1, (env_ids.shape[0],), device=term.device
    )
    term._delay_per_env[env_ids] = delays
