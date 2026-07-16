# Copyright (c) 2024-2025, Ryan Donald
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Custom observation functions for normalized space [-100, +100]."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from isaaclab.assets import Articulation
from isaaclab.managers import ManagerTermBase, SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


def joint_pos_normalized_100(
    env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    # joint positions in normalized [-100, 100] space.

    asset: Articulation = env.scene[asset_cfg.name]
    joint_pos = asset.data.joint_pos.torch[:, asset_cfg.joint_ids]
    joint_limits = asset.data.soft_joint_pos_limits.torch[:, asset_cfg.joint_ids, :]
    lower = joint_limits[:, :, 0]
    upper = joint_limits[:, :, 1]

    # normalize to [-100, 100]: 200 * (pos - lower) / (upper - lower) - 100
    normalized = 200.0 * (joint_pos - lower) / (upper - lower) - 100.0
    return normalized


def joint_vel_normalized(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    velocity_scale: float = 1.0,
) -> torch.Tensor:
    # joint velocities in rad/s

    asset: Articulation = env.scene[asset_cfg.name]
    joint_vel = asset.data.joint_vel.torch[:, asset_cfg.joint_ids]
    return joint_vel * velocity_scale


class joint_vel_finite_diff(ManagerTermBase):
    # finite difference joint velocities, matches how the so101 sim2real
    # deployment works.

    def __init__(self, cfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self._asset_cfg: SceneEntityCfg = cfg.params.get(
            "asset_cfg", SceneEntityCfg("robot")
        )
        self._asset_cfg.resolve(env.scene)
        self._asset: Articulation = env.scene[self._asset_cfg.name]
        # control timestep (decimation * sim_dt), i.e. the interval between policy
        # observations — the same dt the deployment differences over.
        self._dt = env.step_dt
        num_joints = self._asset.data.joint_pos.torch[
            :, self._asset_cfg.joint_ids
        ].shape[1]
        self._prev_pos = torch.zeros(env.num_envs, num_joints, device=env.device)
        self._has_prev = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    def reset(self, env_ids=None) -> None:
        # drop the stored sample so the first velocity after a reset is zero
        if env_ids is None:
            self._has_prev[:] = False
        else:
            self._has_prev[env_ids] = False

    def __call__(
        self,
        env: ManagerBasedEnv,
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
        velocity_scale: float = 1.0,
    ) -> torch.Tensor:
        cur = self._asset.data.joint_pos.torch[:, self._asset_cfg.joint_ids]
        vel = (cur - self._prev_pos) / self._dt
        # zero until a previous sample exists (first obs of each episode)
        vel = torch.where(self._has_prev.unsqueeze(-1), vel, torch.zeros_like(vel))
        self._prev_pos = cur.clone()
        self._has_prev[:] = True
        return vel * velocity_scale


def joint_pos_normalized_100_rel(
    env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    # joint positions relative to default, normalized to [-100, 100].

    asset: Articulation = env.scene[asset_cfg.name]
    joint_pos = asset.data.joint_pos.torch[:, asset_cfg.joint_ids]
    default_pos = asset.data.default_joint_pos.torch[:, asset_cfg.joint_ids]
    joint_limits = asset.data.soft_joint_pos_limits.torch[:, asset_cfg.joint_ids, :]
    lower = joint_limits[:, :, 0]
    upper = joint_limits[:, :, 1]

    # normalize both current and default to [-100, 100], return the difference.
    current_norm = 200.0 * (joint_pos - lower) / (upper - lower) - 100.0
    default_norm = 200.0 * (default_pos - lower) / (upper - lower) - 100.0

    return current_norm - default_norm
