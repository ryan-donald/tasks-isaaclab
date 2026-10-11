# Copyright (c) 2024-2026, Ryan Donald
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Action terms for the SO-ARM101 lift task.

The arm uses the shared normalized [-100, +100] position action. The gripper is a
binary open/close command, delayed the way the arm is.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import MISSING
from typing import TYPE_CHECKING

import torch
from isaaclab.envs.mdp.actions.actions_cfg import BinaryJointPositionActionCfg
from isaaclab.envs.mdp.actions.binary_joint_actions import BinaryJointPositionAction
from isaaclab.managers.action_manager import ActionTerm
from isaaclab.utils.configclass import configclass

# the shared arm action and delay randomization.
from ryan_tasks.mdp.actions import (  # noqa: F401
    NormalizedJointPositionAction,
    NormalizedJointPositionActionCfg,
    randomize_action_delay,
)

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


@configclass
class DelayedBinaryJointPositionActionCfg(BinaryJointPositionActionCfg):
    """Binary gripper position action with a per-env action-delay buffer, sampled on
    reset by :func:`randomize_action_delay`."""

    class_type: type[ActionTerm] = MISSING

    # maximum number of steps the gripper command can be delayed (the buffer depth).
    delay_steps: int = 0


class DelayedBinaryJointPositionAction(BinaryJointPositionAction):
    """Binary gripper position action term with a per-env action delay buffer.

    The applied command lags the commanded one by a per-env number of steps in
    [0, delay_steps], read from ``_delay_per_env`` (the same attribute the arm action
    exposes).
    """

    cfg: DelayedBinaryJointPositionActionCfg

    def __init__(self, cfg: DelayedBinaryJointPositionActionCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)

        # per-env action delay buffer, as in NormalizedJointPositionAction.
        self._max_delay = cfg.delay_steps
        if self._max_delay > 0:
            self._action_delay_buf = torch.zeros(
                self._max_delay + 1, self.num_envs, self._num_joints, device=self.device
            )
            self._delay_per_env = torch.full(
                (self.num_envs,), self._max_delay, dtype=torch.long, device=self.device
            )
            print(
                f"[DelayedBinaryJointPositionAction] Max action delay:"
                f" {self._max_delay} step(s)"
            )

    def action_bounds(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Range for the policy mean. Only the sign is executed; +/-3 is far enough
        from zero that exploration noise cannot flip a held grasp."""
        bound = torch.full((self._num_joints,), 3.0, device=self.device)
        return -bound, bound

    def process_actions(self, actions: torch.Tensor):
        # resolve the binary open/close command into self._processed_actions
        super().process_actions(actions)

        if self._max_delay > 0:
            # newest command -> slot [-1]; buffer rolls so older commands shift down
            self._action_delay_buf = torch.roll(
                self._action_delay_buf, shifts=-1, dims=0
            )
            self._action_delay_buf[-1] = self._processed_actions
            # per-env read index: delay d reads slot (max_delay - d)
            read_idx = self._max_delay - self._delay_per_env  # (num_envs,)
            env_idx = torch.arange(self.num_envs, device=self.device)
            self._processed_actions = self._action_delay_buf[read_idx, env_idx].clone()

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        super().reset(env_ids)
        if self._max_delay > 0 and env_ids is not None:
            # prime the buffer with the current command so a fresh episode starts
            # undelayed.
            for slot in range(self._max_delay + 1):
                self._action_delay_buf[slot, env_ids] = self._processed_actions[env_ids]
