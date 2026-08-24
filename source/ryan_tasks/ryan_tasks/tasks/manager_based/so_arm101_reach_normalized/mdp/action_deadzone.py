# Copyright (c) 2025-2026, Ryan Donald
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def action_deadzone_penalty(
    env: ManagerBasedRLEnv, threshold: float = 2.0
) -> torch.Tensor:
    """Penalize actions that are too small to move the real robot (deadzone).

    Returns 1.0 when the action magnitude is below the threshold, 0.0 otherwise,
    so use a negative weight in the RewTerm to turn this into a penalty, e.g.
    ``RewTerm(func=..., weight=-0.05)``.
    """
    # raw actions from the policy, in normalized space.
    actions = env.action_manager.action
    action_magnitude = torch.norm(actions, dim=-1)
    return torch.where(
        action_magnitude < threshold,
        torch.ones_like(action_magnitude),
        torch.zeros_like(action_magnitude),
    )
