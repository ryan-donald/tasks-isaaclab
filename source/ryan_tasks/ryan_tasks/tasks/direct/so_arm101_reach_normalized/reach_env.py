# Copyright (c) 2024-2026, Ryan Donald
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Direct-workflow port of Ryan-Reach-SO-ARM101-Normalized-v0.

Keeps the manager-based task's operation order, so it reproduces it bit for bit. The
per-step math is replayed as CUDA graphs, and the reset and resample checks, which sync
the GPU, only run on steps where one can happen.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch
import warp as wp
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.utils.math import combine_frame_transforms
from isaaclab.utils.noise import NoiseModelWithAdditiveBias

if TYPE_CHECKING:
    from .reach_env_cfg import SoArm101ReachDirectEnvCfg


def _quantize(joint_pos: torch.Tensor, quantum: float) -> torch.Tensor:
    """Snap joint angles to the real encoder grid. No-op when quantum <= 0."""
    if quantum <= 0.0:
        return joint_pos
    return torch.round(joint_pos / quantum) * quantum


def randomize_action_delay(
    env: SoArm101ReachDirectEnv,
    env_ids: torch.Tensor,
    min_delay: int,
    max_delay: int,
) -> None:
    """Sample a fresh per-env action delay in [min_delay, max_delay] on reset."""
    delays = torch.randint(
        min_delay, max_delay + 1, (env_ids.shape[0],), device=env.device
    )
    env._delay_per_env[env_ids] = delays


class CapturedStep:
    # calls fn eagerly for its first calls, then captures it as a CUDA graph and
    # replays it. fn must only use fixed memory and draw no random numbers, as a
    # replay repeats them.

    def __init__(self, fn, warmup_steps: int = 3):
        self._fn = fn
        self._warmup_left = warmup_steps
        self._graph = None

    def __call__(self) -> None:
        if self._graph is not None:
            self._graph.replay()
        elif self._warmup_left > 0:
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                self._fn()
            torch.cuda.current_stream().wait_stream(stream)
            self._warmup_left -= 1
        else:
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                self._fn()
            # capture only records the kernels, so replay to actually take this step.
            graph.replay()
            self._graph = graph


class RewardTermLog:
    # the RewardManager attributes ryan_ppo reads for per-term logging.

    def __init__(self, term_names: list[str], num_envs: int, device: str):
        self.active_terms = term_names
        self._step_reward = torch.zeros(num_envs, len(term_names), device=device)


class SoArm101ReachDirectEnv(DirectRLEnv):
    cfg: SoArm101ReachDirectEnvCfg

    def __init__(
        self, cfg: SoArm101ReachDirectEnvCfg, render_mode: str | None = None, **kwargs
    ):
        # the base class sets episode_length_buf, which resets this.
        self._next_reset_step = 0
        super().__init__(cfg, render_mode, **kwargs)

        self._robot: Articulation = self.scene["robot"]
        num_envs, device = self.num_envs, self.device

        # -- actions
        action_joint_ids, _ = self._robot.find_joints(cfg.action_joint_names)
        # device copies of the joint indices, as a python list syncs on every use.
        self._action_joint_ids = torch.tensor(action_joint_ids, device=device)
        self._action_joint_ids_wp = wp.array(
            action_joint_ids, dtype=wp.int32, device=device
        )
        action_dim = len(action_joint_ids)
        self._action_in = torch.zeros(num_envs, action_dim, device=device)
        self._action = torch.zeros_like(self._action_in)
        self._prev_action = torch.zeros_like(self._action_in)
        self._processed_actions = torch.zeros_like(self._action_in)
        self._joint_targets = torch.zeros_like(self._action_in)
        self._joint_limits = self._robot.data.soft_joint_pos_limits.torch[
            :, self._action_joint_ids, :
        ].clone()
        default_pos = self._robot.data.default_joint_pos.torch[
            :, self._action_joint_ids
        ]
        lower = self._joint_limits[:, :, 0]
        upper = self._joint_limits[:, :, 1]
        # default pose is 0 in normalized space
        self._action_offset = 200.0 * (default_pos - lower) / (upper - lower) - 100.0
        # servo setpoint ramp
        self._prev_cmd = self._robot.data.joint_pos.torch[
            :, self._action_joint_ids
        ].clone()
        self._prev_vel = torch.zeros_like(self._prev_cmd)
        # action delay
        self._max_delay = cfg.max_delay_steps
        self._action_delay_buf = torch.zeros(
            self._max_delay + 1, num_envs, action_dim, device=device
        )
        self._delay_per_env = torch.full(
            (num_envs,), self._max_delay, dtype=torch.long, device=device
        )
        self._env_idx = torch.arange(num_envs, device=device)

        # -- observations
        obs_joint_ids, _ = self._robot.find_joints(cfg.obs_joint_names)
        self._obs_joint_ids = torch.tensor(obs_joint_ids, device=device)
        self._prev_obs_pos = torch.zeros(num_envs, len(obs_joint_ids), device=device)
        self._has_prev_obs_pos = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self._joint_pos_noise = NoiseModelWithAdditiveBias(
            cfg.joint_pos_noise, num_envs=num_envs, device=device
        )
        # oldest -> newest along dim 0, like IsaacLab's CircularBuffer.
        self._action_history = torch.zeros(
            cfg.action_history_length, num_envs, action_dim, device=device
        )
        self._history_empty = torch.ones(num_envs, dtype=torch.bool, device=device)
        # noise draws, made eagerly before each replay.
        self._joint_pos_rand = torch.zeros(num_envs, len(obs_joint_ids), device=device)
        self._joint_vel_rand = torch.zeros_like(self._joint_pos_rand)
        self._obs_buf = torch.zeros(num_envs, cfg.observation_space, device=device)

        # -- goal command
        self._command = cfg.ee_pose.class_type(cfg.ee_pose, self)
        self._next_resample_step = 0

        # -- rewards
        self._ee_body_id = self._robot.find_bodies(cfg.ee_body_name)[0][0]
        self._reward_buf = torch.zeros(num_envs, device=device)
        self.reward_manager = RewardTermLog(
            [
                "end_effector_position_tracking",
                "end_effector_position_tracking_fine_grained",
                "action_rate",
                "joint_vel",
                "action_l2",
                "action_rate_near",
            ],
            num_envs,
            device,
        )

        self._process_actions_step = CapturedStep(self._process_actions)
        self._joint_targets_step = CapturedStep(self._compute_joint_targets)
        self._rewards_step = CapturedStep(self._compute_rewards)
        self._observations_step = CapturedStep(self._compute_observations)

    # callers such as ryan_ppo's staggered starts replace episode_length_buf, which
    # invalidates the predicted reset step.
    @property
    def episode_length_buf(self) -> torch.Tensor:
        return self._episode_length_buf

    @episode_length_buf.setter
    def episode_length_buf(self, value: torch.Tensor) -> None:
        self._episode_length_buf = value
        self._next_reset_step = 0

    """
    Stepping.
    """

    def step(self, action: torch.Tensor):
        # DirectRLEnv.step, in the manager-based task's order (command update after
        # resets) and without a reset sync every step.
        self._pre_physics_step(action.to(self.device))

        is_rendering = self.sim.is_rendering
        for _ in range(self.cfg.decimation):
            self._sim_step_counter += 1
            self._apply_action()
            self.scene.write_data_to_sim()
            self.sim.step(render=False)
            if (
                self._sim_step_counter % self.cfg.sim.render_interval == 0
                and is_rendering
            ):
                self.sim.render(skip_app_pumping=not self.render_enabled)
            self.scene.update(dt=self.physics_dt)

        self._episode_length_buf += 1
        self.common_step_counter += 1

        self.reset_terminated[:], self.reset_time_outs[:] = self._get_dones()
        self.reset_buf = self.reset_terminated | self.reset_time_outs
        self.reward_buf = self._get_rewards()

        # timeouts are the only termination, so no env resets before the oldest episode
        # reaches max_episode_length.
        if self.common_step_counter >= self._next_reset_step:
            reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1).int()
            if len(reset_env_ids) > 0:
                self._reset_idx(reset_env_ids)
            steps_left = self.max_episode_length - int(self._episode_length_buf.max())
            self._next_reset_step = self.common_step_counter + steps_left

        self._update_command()
        self.obs_buf = self._get_observations()
        return (
            self.obs_buf,
            self.reward_buf,
            self.reset_terminated,
            self.reset_time_outs,
            self.extras,
        )

    def _pre_physics_step(self, actions: torch.Tensor):
        self._action_in.copy_(actions)
        self._process_actions_step()

    def _process_actions(self):
        self._prev_action.copy_(self._action)
        self._action.copy_(self._action_in)

        processed = self.cfg.action_scale * self._action + self._action_offset
        # newest action -> slot [-1]; delay d reads slot (max_delay - d)
        self._action_delay_buf.copy_(torch.roll(self._action_delay_buf, -1, dims=0))
        self._action_delay_buf[-1] = processed
        read_idx = self._max_delay - self._delay_per_env
        self._processed_actions.copy_(self._action_delay_buf[read_idx, self._env_idx])

    def _apply_action(self):
        if self.cfg.deadband > 0.0:
            self._robot.data.joint_pos  # refresh the buffer the graph reads
        self._joint_targets_step()
        self._robot.set_joint_position_target_index(
            target=self._joint_targets, joint_ids=self._action_joint_ids_wp
        )

    def _compute_joint_targets(self):
        # clamp to [-100, 100] range
        normalized_clamped = torch.clamp(self._processed_actions, -100.0, 100.0)

        lower = self._joint_limits[:, :, 0]
        upper = self._joint_limits[:, :, 1]

        if self.cfg.deadband > 0.0:
            current = self._robot.data.joint_pos.torch[:, self._action_joint_ids]
            current_norm = 200.0 * (current - lower) / (upper - lower) - 100.0
            inside = (normalized_clamped - current_norm).abs() < self.cfg.deadband
            normalized_clamped = torch.where(inside, current_norm, normalized_clamped)

        # convert from normalized [-100, 100] to radians
        radians = (normalized_clamped + 100.0) / 200.0 * (upper - lower) + lower

        # servo setpoint limits, see NormalizedJointPositionAction.
        if self.cfg.max_accel_rad_s2 > 0.0:
            dt = self.physics_dt
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
            self._prev_vel.copy_(torch.where(snap, err / dt, self._prev_vel))
            self._prev_cmd.copy_(radians)
        elif self.cfg.max_slew_rad_s > 0.0:
            step = self.cfg.max_slew_rad_s * self.physics_dt
            radians = self._prev_cmd + (radians - self._prev_cmd).clamp(-step, step)
            self._prev_cmd.copy_(radians)

        self._joint_targets.copy_(radians)

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self._episode_length_buf >= self.max_episode_length
        return torch.zeros_like(time_out), time_out

    def _get_rewards(self) -> torch.Tensor:
        # refresh the buffers the graph reads
        data = self._robot.data
        data.root_pos_w, data.root_quat_w, data.body_pos_w, data.joint_vel
        self._rewards_step()
        return self._reward_buf

    def _compute_rewards(self):
        cfg = self.cfg
        data = self._robot.data

        # end effector distance to the goal
        des_pos_w, _ = combine_frame_transforms(
            data.root_pos_w.torch, data.root_quat_w.torch, self._command.command[:, :3]
        )
        curr_pos_w = data.body_pos_w.torch[:, self._ee_body_id]
        distance = torch.linalg.norm(curr_pos_w - des_pos_w, dim=1)
        action_rate = torch.sum(torch.square(self._action - self._prev_action), dim=1)
        near_goal = (distance < cfg.near_goal_radius).float()

        terms = (
            (cfg.rew_position_tracking, distance),
            (
                cfg.rew_position_tracking_fine_grained,
                1 - torch.tanh(distance / cfg.fine_grained_std),
            ),
            (cfg.rew_action_rate, action_rate),
            (cfg.rew_joint_vel, torch.sum(torch.square(data.joint_vel.torch), dim=1)),
            (cfg.rew_action_l2, torch.sum(torch.square(self._action), dim=1)),
            (cfg.rew_action_rate_near, action_rate * near_goal),
        )

        # RewardManager.compute's accumulation, which scales each term by dt.
        step_reward = self.reward_manager._step_reward
        self._reward_buf[:] = 0.0
        for idx, (weight, term) in enumerate(terms):
            if weight == 0.0:
                step_reward[:, idx] = 0.0
                continue
            value = term * weight * self.step_dt
            self._reward_buf += value
            step_reward[:, idx] = value / self.step_dt

    def _update_command(self):
        # CommandTerm.compute. goals resample on a fixed timer, so the resample check
        # only runs from two steps before the earliest one can expire (float drift).
        command = self._command
        command.time_left -= self.step_dt
        if self.common_step_counter >= self._next_resample_step:
            resample_env_ids = (command.time_left <= 0.0).nonzero().flatten()
            if len(resample_env_ids) > 0:
                command._resample(resample_env_ids)
            steps_left = int(command.time_left.min() / self.step_dt) - 2
            self._next_resample_step = self.common_step_counter + max(steps_left, 1)

    def _get_observations(self) -> dict:
        self._robot.data.joint_pos  # refresh the buffer the graph reads

        # draws in NoiseModelWithAdditiveBias's order. its first call widens the bias
        # to one value per joint and resamples it.
        noise = self._joint_pos_noise
        if noise._sample_bias_per_component and noise._num_components is None:
            noise._num_components = self._joint_pos_rand.shape[1]
            noise._bias = noise._bias.repeat(1, noise._num_components)
            noise.reset()
        self._joint_pos_rand.uniform_()
        self._joint_vel_rand.uniform_()

        self._observations_step()
        # a new tensor each step, as callers keep the previous observation.
        return {"policy": self._obs_buf.clone()}

    def _compute_observations(self):
        cfg = self.cfg
        data = self._robot.data
        ids = self._obs_joint_ids

        # joint positions relative to default, normalized to [-100, 100], with uniform
        # noise and the per-episode bias
        joint_pos = _quantize(data.joint_pos.torch[:, ids], cfg.pos_quantum)
        default_pos = data.default_joint_pos.torch[:, ids]
        joint_limits = data.soft_joint_pos_limits.torch[:, ids, :]
        lower = joint_limits[:, :, 0]
        upper = joint_limits[:, :, 1]
        current_norm = 200.0 * (joint_pos - lower) / (upper - lower) - 100.0
        default_norm = 200.0 * (default_pos - lower) / (upper - lower) - 100.0
        pos_noise = cfg.joint_pos_noise.noise_cfg
        pos_range = pos_noise.n_max - pos_noise.n_min
        obs_joint_pos = current_norm - default_norm
        obs_joint_pos = (
            obs_joint_pos + self._joint_pos_rand * pos_range + pos_noise.n_min
        )
        obs_joint_pos = obs_joint_pos + self._joint_pos_noise._bias

        # finite difference joint velocities over the quantized positions, with noise
        vel = (joint_pos - self._prev_obs_pos) / self.step_dt
        vel = torch.where(self._has_prev_obs_pos.unsqueeze(-1), vel, 0.0)
        self._prev_obs_pos.copy_(joint_pos)
        self._has_prev_obs_pos.fill_(True)
        vel_noise = cfg.joint_vel_noise
        vel_range = vel_noise.n_max - vel_noise.n_min
        obs_joint_vel = vel + self._joint_vel_rand * vel_range + vel_noise.n_min

        # last 6 actions. an empty history (after reset) is filled with the action.
        history = torch.where(
            self._history_empty[None, :, None],
            self._action.unsqueeze(0),
            self._action_history,
        )
        history = torch.roll(history, -1, dims=0)
        history[-1] = self._action
        self._action_history.copy_(history)
        self._history_empty.fill_(False)
        obs_actions = history.transpose(0, 1).reshape(self.num_envs, -1)

        self._obs_buf.copy_(
            torch.cat(
                (obs_joint_pos, obs_joint_vel, self._command.command, obs_actions),
                dim=-1,
            )
        )

    """
    Resets.
    """

    def _reset_idx(self, env_ids: Sequence[int]):
        # ManagerBasedRLEnv._reset_idx order: scene, events, observations, actions,
        # command, episode length.
        self.scene.reset(env_ids)
        env_step_count = self._sim_step_counter // self.cfg.decimation
        self.event_manager.apply(
            mode="reset", env_ids=env_ids, global_env_step_count=env_step_count
        )

        # -- observations
        self._has_prev_obs_pos[env_ids] = False
        self._history_empty[env_ids] = True
        self._action_history[:, env_ids] = 0.0
        self._joint_pos_noise.reset(env_ids)

        # -- actions
        self._prev_action[env_ids] = 0.0
        self._action[env_ids] = 0.0
        self._prev_cmd[env_ids] = self._robot.data.joint_pos.torch[
            :, self._action_joint_ids
        ][env_ids]
        self._prev_vel[env_ids] = 0.0
        for slot in range(self._max_delay + 1):
            self._action_delay_buf[slot, env_ids] = self._processed_actions[env_ids]

        # -- goal command. new goal timers invalidate the predicted resample step.
        self._command.reset(env_ids)
        self._next_resample_step = 0

        self._episode_length_buf[env_ids] = 0
        self._next_reset_step = 0
