# Copyright (c) 2024-2026, Ryan Donald
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Observation terms for the push tasks."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def goal_position_in_robot_root_frame(
    env: ManagerBasedRLEnv, command_name: str = "object_pose"
) -> torch.Tensor:
    """The goal position in the robot root frame. Shape (num_envs, 3).

    The goal's orientation is constant for the cube push, and constant columns break
    the running observation normalizer, so only the position is returned.
    """
    return env.command_manager.get_command(command_name)[:, :3]
