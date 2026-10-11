# Copyright (c) 2024-2026, Ryan Donald
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Helpers shared by the direct-workflow tasks."""

import torch


class RewardTermLog:
    # the RewardManager attributes ryan_ppo reads for per-term logging.

    def __init__(self, term_names: list[str], num_envs: int, device: str):
        self.active_terms = term_names
        self._step_reward = torch.zeros(num_envs, len(term_names), device=device)
