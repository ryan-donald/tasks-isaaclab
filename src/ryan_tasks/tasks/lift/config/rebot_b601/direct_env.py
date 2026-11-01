# Copyright (c) 2024-2026, Ryan Donald
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Pick and place, direct workflow: the manager-based task bit for bit.

It keeps the manager-based operation order and replays the per-step math as CUDA
graphs. The rest of the cost is the contact physics.
"""

from __future__ import annotations

import torch
import warp as wp
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import (
    NoiseModelWithAdditiveBias,
    NoiseModelWithAdditiveBiasCfg,
)
from isaaclab.utils.noise import UniformNoiseCfg as Unoise

from ryan_tasks.mdp.rebot_b601 import (
    ActionHistory,
    ArmCommand,
    GripperCommand,
    randomize_delay,
    widen_bias,
)
from ryan_tasks.robots.rebot_b601 import (
    ARM_JOINTS,
    GRIPPER_BODY,
    JAW_JOINT,
    JAW_OPEN,
)
from ryan_tasks.robots.so_arm101 import MAX_SLEW_RAD_S
from ryan_tasks.tasks.reach.config.so_arm101.direct_env import CapturedStep
from ryan_tasks.utils import RewardTermLog

from . import env_cfg as task
from . import params as p
from .logic import TERM_NAMES, PlaceLogic

ACTION_DIM = 5


def randomize_action_delay(env: PlaceDirectEnv, env_ids, low: int, high: int):
    randomize_delay(env.arm, env_ids, low, high)


@configclass
class PlaceDirectEventsCfg(task.PlaceEventsCfg):
    randomize_delay = EventTerm(
        func=randomize_action_delay,
        mode="reset",
        params={"low": 1, "high": p.MAX_DELAY},
    )


@configclass
class PlaceDirectEnvCfg(DirectRLEnvCfg):
    sim: SimulationCfg = task.simulation()
    scene: task.PlaceSceneCfg = task.PlaceSceneCfg(num_envs=12288, env_spacing=2.5)
    events: PlaceDirectEventsCfg = PlaceDirectEventsCfg()
    decimation = 1
    episode_length_s = p.EPISODE_S
    action_space = ACTION_DIM
    observation_space = 4 + 1 + 5 + 3 + 3 + 1 + p.HISTORY * ACTION_DIM
    state_space = 0
    joint_pos_noise: NoiseModelWithAdditiveBiasCfg = NoiseModelWithAdditiveBiasCfg(
        noise_cfg=Unoise(n_min=-p.POS_NOISE, n_max=p.POS_NOISE),
        bias_noise_cfg=Unoise(n_min=-p.POS_BIAS, n_max=p.POS_BIAS, operation="abs"),
    )


class PlaceDirectEnv(DirectRLEnv):
    cfg: PlaceDirectEnvCfg

    def __init__(
        self, cfg: PlaceDirectEnvCfg, render_mode: str | None = None, **kwargs
    ):
        super().__init__(cfg, render_mode, **kwargs)
        num_envs, device = self.num_envs, self.device
        self._robot = self.scene["robot"]
        self._cube = self.scene["object"]
        data = self._robot.data

        arm_ids = self._robot.find_joints(ARM_JOINTS)[0]
        jaw_ids = self._robot.find_joints(JAW_JOINT)[0]
        self._arm_ids = torch.tensor(arm_ids, device=device)
        self._arm_ids_wp = wp.array(arm_ids, dtype=wp.int32, device=device)
        self._jaw_ids_wp = wp.array(jaw_ids, dtype=wp.int32, device=device)
        self._jaw_id = jaw_ids[0]
        self._vel_ids = torch.tensor(arm_ids + jaw_ids, device=device)
        self._gripper_id = self._robot.find_bodies(GRIPPER_BODY)[0][0]

        limits = data.soft_joint_pos_limits.torch[:, self._arm_ids].clone()
        self._lower, self._upper = limits[..., 0], limits[..., 1]
        self._joint_span = self._upper - self._lower
        self._default_pos = data.default_joint_pos.torch[:, self._arm_ids].clone()

        self.arm = ArmCommand(
            limits,
            self._default_pos,
            data.joint_pos.torch[:, self._arm_ids],
            p.ACTION_SCALE,
            p.MAX_DELAY,
            MAX_SLEW_RAD_S * self.physics_dt,
        )
        self.gripper = GripperCommand(self.arm, JAW_OPEN)
        self._action_in = torch.zeros(num_envs, ACTION_DIM, device=device)
        self._action = torch.zeros_like(self._action_in)
        self._prev_action = torch.zeros_like(self._action_in)

        self.logic = PlaceLogic(num_envs, self.scene.env_origins)
        self._weights = torch.tensor(
            [p.WEIGHTS[name] for name in TERM_NAMES], device=device
        )

        self._pos_noise = NoiseModelWithAdditiveBias(
            cfg.joint_pos_noise, num_envs, device
        )
        self._pos_min, self._pos_range = -p.POS_NOISE, 2 * p.POS_NOISE
        self._jaw_min, self._jaw_range = -p.JAW_NOISE, 2 * p.JAW_NOISE
        self._vel_min, self._vel_range = -p.VEL_NOISE, 2 * p.VEL_NOISE
        self._cube_min, self._cube_range = -p.CUBE_NOISE, 2 * p.CUBE_NOISE
        self._pos_rand = torch.zeros(num_envs, len(arm_ids), device=device)
        self._jaw_rand = torch.zeros(num_envs, 1, device=device)
        self._vel_rand = torch.zeros(num_envs, len(arm_ids) + 1, device=device)
        self._cube_rand = torch.zeros(num_envs, 3, device=device)
        self._history = ActionHistory(p.HISTORY, num_envs, ACTION_DIM, device)
        self._obs = torch.zeros(num_envs, cfg.observation_space, device=device)

        self._reward = torch.zeros(num_envs, device=device)
        self.reward_manager = RewardTermLog(list(TERM_NAMES), num_envs, device)
        self._diverged = torch.zeros(num_envs, dtype=torch.bool, device=device)

        self._process_actions_step = CapturedStep(self._process_actions)
        self._ramp_step = CapturedStep(self.arm.ramp)
        self._dones_rewards_step = CapturedStep(self._write_dones_rewards)
        self._observations_step = CapturedStep(self._write_observations)

    def step(self, action: torch.Tensor):
        # ManagerBasedRLEnv.step.
        self._action_in.copy_(action)
        self._process_actions_step()

        self._sim_step_counter += 1
        self._ramp_step()
        targets = self._robot.actuators.target_command
        targets.set_position_index(value=self.arm.target, joint_ids=self._arm_ids_wp)
        targets.set_position_index(
            value=self.gripper.target, joint_ids=self._jaw_ids_wp
        )
        self.scene.write_data_to_sim()
        self.sim.step(render=False)
        if self.sim.is_rendering:
            self.sim.render(skip_app_pumping=not self.render_enabled)
        self.scene.update(dt=self.physics_dt)

        self.episode_length_buf += 1
        self.common_step_counter += 1
        self.reset_time_outs = self.episode_length_buf >= self.max_episode_length

        self._refresh_buffers()
        self._dones_rewards_step()
        self.reset_terminated = self._diverged

        reset_ids = (self.reset_terminated | self.reset_time_outs).nonzero().flatten()
        if len(reset_ids) > 0:
            self._reset_idx(reset_ids.int())

        self.obs_buf = self._get_observations()
        return (
            self.obs_buf,
            self._reward,
            self.reset_terminated,
            self.reset_time_outs,
            self.extras,
        )

    def _refresh_buffers(self):
        # the graphs read these buffers, which only refresh on property access.
        data = self._robot.data
        data.joint_pos, data.joint_vel, data.body_link_pose_w
        self._cube.data.root_link_pose_w

    def _process_actions(self):
        self._prev_action.copy_(self._action)
        self._action.copy_(self._action_in)
        self.arm.push(self._action_in[:, :4])
        self.gripper.push(self._action_in[:, 4:])

    def _write_dones_rewards(self):
        data = self._robot.data
        joint_pos = data.joint_pos.torch
        joint_vel = data.joint_vel.torch
        cube_pose = self._cube.data.root_link_pose_w.torch
        arm = joint_pos[:, self._arm_ids]

        cube = cube_pose[:, :3] - self.scene.env_origins
        finite = torch.isfinite(cube).all(dim=1) & torch.isfinite(joint_vel).all(dim=1)
        self._diverged.copy_(~finite | (torch.norm(cube, dim=1) > p.MAX_CUBE_RANGE))

        self.logic.write_terms(
            data.body_link_pose_w.torch[:, self._gripper_id],
            cube_pose,
            joint_pos[:, self._jaw_id],
            arm,
            self._default_pos,
            joint_vel[:, self._vel_ids],
            torch.minimum(arm - self._lower, self._upper - arm),
            self._action,
            self._prev_action,
        )
        step_reward = self.reward_manager._step_reward
        step_reward.copy_(self.logic.terms * self._weights)
        torch.sum(step_reward, dim=1, out=self._reward).mul_(self.step_dt)

    def _get_observations(self) -> dict:
        self._refresh_buffers()
        widen_bias(self._pos_noise, self._pos_rand.shape[1])
        self._pos_rand.uniform_()
        self._jaw_rand.uniform_()
        self._vel_rand.uniform_()
        self._cube_rand.uniform_()
        self._observations_step()
        # a new tensor each step, as callers keep the previous observation.
        return {"policy": self._obs.clone()}

    def _write_observations(self):
        data = self._robot.data
        joint_pos = data.joint_pos.torch
        pos = (
            200.0 * (joint_pos[:, self._arm_ids] - self._default_pos) / self._joint_span
        )
        pos = pos + (self._pos_rand * self._pos_range + self._pos_min)
        pos = pos + self._pos_noise._bias
        jaw = 100.0 * joint_pos[:, self._jaw_id].unsqueeze(1) / JAW_OPEN
        jaw = jaw + (self._jaw_rand * self._jaw_range + self._jaw_min)
        vel = data.joint_vel.torch[:, self._vel_ids]
        vel = vel + (self._vel_rand * self._vel_range + self._vel_min)
        cube = self._cube.data.root_link_pose_w.torch[:, :3] - self.scene.env_origins
        cube = cube + (self._cube_rand * self._cube_range + self._cube_min)
        action = torch.cat((self.arm.executed_action(), self.gripper.sign()), dim=1)
        self._obs.copy_(
            torch.cat(
                (
                    pos,
                    jaw,
                    vel,
                    cube,
                    self.logic.target_b,
                    self.logic.carried.float().unsqueeze(1),
                    self._history.update(action),
                ),
                dim=-1,
            )
        )

    def _reset_idx(self, env_ids):
        # ManagerBasedRLEnv._reset_idx: scene, events, observations, actions, target.
        self.scene.reset(env_ids)
        self.event_manager.apply(
            mode="reset",
            env_ids=env_ids,
            global_env_step_count=self._sim_step_counter // self.cfg.decimation,
        )
        self._history.reset(env_ids)
        self._pos_noise.reset(env_ids)
        self._prev_action[env_ids] = 0.0
        self._action[env_ids] = 0.0
        self.arm.reset(env_ids, self._robot.data.joint_pos.torch[:, self._arm_ids])
        self.gripper.reset(env_ids)
        self.logic.reset(env_ids, self._cube.data.root_pos_w.torch)
        self.episode_length_buf[env_ids] = 0

    def _pre_physics_step(self, actions: torch.Tensor):
        pass

    def _apply_action(self):
        pass

    def _get_dones(self):
        pass

    def _get_rewards(self):
        pass
