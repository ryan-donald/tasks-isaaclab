# Copyright (c) 2024-2026, Ryan Donald
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""The reBot B601-DM command path, shared by the manager-based and direct tasks."""

import torch
from isaaclab.utils.noise import NoiseModelWithAdditiveBias


class ArmCommand:
    """The path from a policy action to a joint position target.

    Action 0 is the default pose, and ``scale`` is normalized units per action unit on
    each joint's [-100, 100] range. The target waits a per-env number of steps, then
    ramps at ``max_step`` radians per step. Everything is written in place, so a CUDA
    graph can capture it.
    """

    def __init__(
        self,
        limits: torch.Tensor,
        default_pos: torch.Tensor,
        joint_pos: torch.Tensor,
        scale: float,
        max_delay: int,
        max_step: float,
    ):
        self.lower, self.upper = limits[..., 0], limits[..., 1]
        self.default_pos = default_pos
        self.gain = scale / 200.0 * (self.upper - self.lower)
        self.max_step = max_step
        self.max_delay = max_delay
        num_envs = joint_pos.shape[0]
        self.delay = torch.full(
            (num_envs,), max_delay, dtype=torch.long, device=joint_pos.device
        )
        self.env_ids = torch.arange(num_envs, device=joint_pos.device)
        self.queue = joint_pos.clone().repeat(max_delay + 1, 1, 1)
        self.desired = joint_pos.clone()
        self.target = joint_pos.clone()

    def push(self, action: torch.Tensor):
        """Queue a new action and read the one each env's delay releases."""
        self.queue.copy_(torch.roll(self.queue, -1, dims=0))
        self.queue[-1] = torch.minimum(
            torch.maximum(self.default_pos + self.gain * action, self.lower), self.upper
        )
        self.desired.copy_(self.queue[self.max_delay - self.delay, self.env_ids])

    def ramp(self):
        """One step of the speed limit."""
        step = (self.desired - self.target).clamp(-self.max_step, self.max_step)
        self.target.add_(step)

    def reset(self, env_ids, joint_pos: torch.Tensor):
        """Hold ``joint_pos`` until the first action comes out of the queue."""
        self.queue[:, env_ids] = joint_pos[env_ids]
        self.desired[env_ids] = joint_pos[env_ids]
        self.target[env_ids] = joint_pos[env_ids]

    def executed_action(self) -> torch.Tensor:
        """The newest action after the joint limit clamp."""
        return (self.queue[-1] - self.default_pos) / self.gain


class GripperCommand:
    """Binary jaw target: open for action >= 0. Waits the arm's per-env delay."""

    def __init__(self, arm: ArmCommand, open_position: float):
        self.arm = arm
        self.open_position = open_position
        self.queue = torch.full(
            (arm.max_delay + 1, arm.env_ids.shape[0], 1),
            open_position,
            device=arm.env_ids.device,
        )
        self.target = torch.full_like(self.queue[0], open_position)
        self.newest = torch.ones_like(self.target)

    def push(self, action: torch.Tensor):
        self.newest.copy_(torch.where(action < 0.0, -1.0, 1.0))
        self.queue.copy_(torch.roll(self.queue, -1, dims=0))
        self.queue[-1] = (self.newest + 1.0) * 0.5 * self.open_position
        index = self.arm.max_delay - self.arm.delay
        self.target.copy_(self.queue[index, self.arm.env_ids])

    def sign(self) -> torch.Tensor:
        """The newest command: +1 open, -1 close."""
        return self.newest

    def reset(self, env_ids):
        self.queue[:, env_ids] = self.open_position
        self.target[env_ids] = self.open_position
        self.newest[env_ids] = 1.0


def randomize_delay(command: ArmCommand, env_ids, low: int, high: int):
    """Draw each env's action delay, in steps, from [low, high]."""
    command.delay[env_ids] = torch.randint_like(command.delay[env_ids], low, high + 1)


class ActionHistory:
    """The last actions, flat and oldest first, as Isaac Lab's history buffer gives."""

    def __init__(self, length: int, num_envs: int, action_dim: int, device: str):
        self.history = torch.zeros(length, num_envs, action_dim, device=device)
        self.empty = torch.ones(num_envs, dtype=torch.bool, device=device)

    def update(self, action: torch.Tensor) -> torch.Tensor:
        # an empty history, after a reset, is filled with the action.
        history = torch.where(
            self.empty[None, :, None], action.unsqueeze(0), self.history
        )
        history = torch.roll(history, -1, dims=0)
        history[-1] = action
        self.history.copy_(history)
        self.empty.fill_(False)
        return history.transpose(0, 1).flatten(1)

    def reset(self, env_ids):
        self.empty[env_ids] = True
        self.history[:, env_ids] = 0.0


def widen_bias(noise: NoiseModelWithAdditiveBias, num_components: int):
    """What the noise model does on its first call: one bias value per component."""
    if noise._sample_bias_per_component and noise._num_components is None:
        noise._num_components = num_components
        noise._bias = noise._bias.repeat(1, num_components)
        noise.reset()
