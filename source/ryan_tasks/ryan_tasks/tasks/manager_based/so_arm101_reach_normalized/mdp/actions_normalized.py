# Copyright (c) 2024-2025, Ryan Donald
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Custom action terms for normalized space [-100, +100]."""

from __future__ import annotations

import torch
from collections.abc import Sequence
from dataclasses import MISSING
from typing import TYPE_CHECKING

from isaaclab.assets.articulation import Articulation
from isaaclab.managers.action_manager import ActionTerm, ActionTermCfg
from isaaclab.utils.configclass import configclass

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


@configclass
class NormalizedJointPositionActionCfg(ActionTermCfg):
    #Configuration for position controlled joint action term matching lerobot


    class_type: type[ActionTerm] = MISSING
    joint_names: list[str] = MISSING
    scale: float = 1.0

    # offset for default position, i.e. 0 in normalized space
    offset: float = 0.0
    use_default_offset: bool = False
    preserve_order: bool = False
    delay_steps: int = 0


class NormalizedJointPositionAction(ActionTerm):
   # action term that receives actions in normalized space [-100, 100] and translates to radians

    cfg: NormalizedJointPositionActionCfg

    def __init__(self, cfg: NormalizedJointPositionActionCfg, env: ManagerBasedEnv):
        # initialize the action term
        super().__init__(cfg, env)

        # resolve the joints over which the action term is applied
        self._asset: Articulation = env.scene[cfg.asset_name]
        self._joint_ids, self._joint_names = self._asset.find_joints(cfg.joint_names)
        self._num_joints = len(self._joint_ids)

        # log info for debugging
        print(
            f"[NormalizedJointPositionAction] Resolved joint names for {self.cfg.asset_name}:"
            f" {self._joint_names} [{self._joint_ids}]"
        )

        # create tensors for raw and processed actions
        self._raw_actions = torch.zeros(env.num_envs, self.action_dim, device=self.device)
        self._processed_actions = torch.zeros_like(self._raw_actions)

        # get joint limits for conversion
        self._joint_limits = self._asset.data.soft_joint_pos_limits.torch[:, self._joint_ids, :].clone()

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
            f"[NormalizedJointPositionAction] Using offset (normalized [-100,+100] space): {self._offset}"
        )

        # action delay buffer, per env action delay.
        self._max_delay = cfg.delay_steps
        if self._max_delay > 0:
            self._action_delay_buf = torch.zeros(
                self._max_delay + 1, env.num_envs, self.action_dim, device=self.device
            )
            self._delay_per_env = torch.full(
                (env.num_envs,), self._max_delay, dtype=torch.long, device=self.device
            )
            print(f"[NormalizedJointPositionAction] Max action delay: {self._max_delay} step(s)")

    """
    Properties.
    """

    @property
    def action_dim(self) -> int:
        return self._num_joints

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions

    """
    Operations.
    """

    def process_actions(self, actions: torch.Tensor):
        # store the raw actions
        self._raw_actions[:] = actions
        # apply scaling and offset in normalized space
        self._processed_actions = self.cfg.scale * self._raw_actions + self._offset

        if self._max_delay > 0:
            # newest action -> slot [-1]; buffer rolls so older actions shift down
            self._action_delay_buf = torch.roll(self._action_delay_buf, shifts=-1, dims=0)
            self._action_delay_buf[-1] = self._processed_actions
            # per-env read index: delay d reads slot (max_delay - d)
            read_idx = self._max_delay - self._delay_per_env  # (num_envs,)
            env_idx = torch.arange(self.num_envs, device=self.device)
            self._processed_actions = self._action_delay_buf[read_idx, env_idx].clone()

    def apply_actions(self):
        # clamp to [-100, 100] range
        normalized_clamped = torch.clamp(self._processed_actions, -100.0, 100.0)
        
        # convert from normalized [-100, 100] to radians
        lower = self._joint_limits[:, :, 0]
        upper = self._joint_limits[:, :, 1]
        radians = (normalized_clamped + 100.0) / 200.0 * (upper - lower) + lower
        
        # apply position commands
        self._asset.set_joint_position_target(radians, joint_ids=self._joint_ids)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
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
    """
    term: NormalizedJointPositionAction = env.action_manager.get_term(action_name)
    delays = torch.randint(
        min_delay, max_delay + 1, (env_ids.shape[0],), device=term.device
    )
    term._delay_per_env[env_ids] = delays
