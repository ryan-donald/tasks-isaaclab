# Copyright (c) 2024-2026, Ryan Donald
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Config for the direct-workflow SO-ARM101 normalized reach task.

Every value mirrors SoArm101ReachNormalizedEnvCfg (the manager-based task); the hardware
constants are imported from it so the two cannot drift apart.
"""

import math

import isaaclab.envs.mdp as mdp
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.envs.mdp.commands.commands_cfg import UniformPoseCommandCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils.configclass import configclass
from isaaclab.utils.noise import NoiseModelWithAdditiveBiasCfg
from isaaclab.utils.noise import UniformNoiseCfg as Unoise
from isaaclab_tasks.manager_based.manipulation.reach.reach_env_cfg import (
    ReachPhysicsCfg,
    ReachSceneCfg,
)

from ryan_tasks.tasks.manager_based.so_arm101_reach_normalized.joint_pos_normalized_env_cfg import (  # noqa: E501
    ACTION_DEADBAND,
    MAX_ACCEL_RAD_S2,
    MAX_SLEW_RAD_S,
    POS_QUANTUM,
)
from ryan_tasks.tasks.manager_based.so_arm101_reach_normalized.mdp import (
    UniformPoseCommandNoMetrics,
)
from ryan_tasks.tasks.robots.so_arm101_urdf_cfg import SO_ARM101_URDF_CFG

from .reach_env import randomize_action_delay

ARM_JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex"]


@configclass
class EventCfg:
    # same terms, order and ranges as the manager-based task, so reset randomization
    # draws from the RNG in the same sequence.

    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=ARM_JOINTS),
            "position_range": (-0.2, 0.2),
            "velocity_range": (0.0, 0.0),
        },
    )

    randomize_gains = EventTerm(
        func=mdp.randomize_actuator_gains,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "stiffness_distribution_params": (0.75, 1.25),
            "damping_distribution_params": (0.5, 1.5),
            "operation": "scale",
            "distribution": "uniform",
        },
    )

    randomize_joint_friction = EventTerm(
        func=mdp.randomize_joint_parameters,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "friction_distribution_params": (0.7, 1.3),
            "operation": "scale",
            "distribution": "uniform",
        },
    )

    randomize_action_delay = EventTerm(
        func=randomize_action_delay,
        mode="reset",
        params={"min_delay": 1, "max_delay": 3},
    )


@configclass
class SoArm101ReachDirectEnvCfg(DirectRLEnvCfg):
    # env
    decimation = 1
    episode_length_s = 12.0
    action_space = 4
    observation_space = 39
    state_space = 0

    # simulation, 60 hz
    sim: SimulationCfg = SimulationCfg(dt=1.0 / 60.0, render_interval=1)

    # scene
    scene: ReachSceneCfg = ReachSceneCfg(num_envs=4096, env_spacing=2.5)

    # events
    events: EventCfg = EventCfg()

    # goal pose command
    ee_pose: UniformPoseCommandCfg = UniformPoseCommandCfg(
        class_type=UniformPoseCommandNoMetrics,
        asset_name="robot",
        body_name="gripper_frame_link",
        resampling_time_range=(4.0, 4.0),
        debug_vis=True,
        position_success_threshold=0.05,
        ranges=UniformPoseCommandCfg.Ranges(
            pos_x=(0.1, 0.25),
            pos_y=(-0.2, 0.2),
            pos_z=(0.1, 0.4),
            roll=(0.0, 0.0),
            pitch=(math.pi, math.pi),
            yaw=(-3.14, 3.14),
        ),
    )

    # actions: normalized [-100, 100] joint position targets
    action_joint_names = ["shoulder_.*", "elbow_flex", "wrist_flex"]
    action_scale = 100.0
    max_delay_steps = 3
    deadband = ACTION_DEADBAND
    max_slew_rad_s = MAX_SLEW_RAD_S
    max_accel_rad_s2 = MAX_ACCEL_RAD_S2

    # observations
    obs_joint_names = ARM_JOINTS
    pos_quantum = POS_QUANTUM
    action_history_length = 6
    # per-step noise plus a per-episode calibration offset on joint positions.
    joint_pos_noise: NoiseModelWithAdditiveBiasCfg = NoiseModelWithAdditiveBiasCfg(
        noise_cfg=Unoise(n_min=-0.3, n_max=0.3),
        bias_noise_cfg=Unoise(n_min=-0.7, n_max=0.7, operation="abs"),
    )
    joint_vel_noise: Unoise = Unoise(n_min=-0.01, n_max=0.01)

    # rewards, in the manager-based task's term order
    ee_body_name = "gripper_frame_link"
    rew_position_tracking = -0.2
    rew_position_tracking_fine_grained = 0.1
    fine_grained_std = 0.1
    rew_action_rate = -0.001
    rew_joint_vel = -0.001
    rew_action_l2 = 0.0
    rew_action_rate_near = -0.15
    near_goal_radius = 0.05

    def __post_init__(self):
        self.sim.physics = ReachPhysicsCfg()
        # randomized reset poses can spawn links in contact, overflowing the default
        # 163840-patch GPU buffer
        self.sim.physics.physx.gpu_max_rigid_patch_count = 2**18
        self.sim.physics.default.gpu_max_rigid_patch_count = 2**18

        self.scene.robot = SO_ARM101_URDF_CFG.replace(
            prim_path="{ENV_REGEX_NS}/Robot",
            actuators={
                **SO_ARM101_URDF_CFG.actuators,
                "arm": SO_ARM101_URDF_CFG.actuators["arm"].replace(friction=0.0),
            },
        )
        self.scene.table = None
        self.scene.ground.init_state.pos = (0.0, 0.0, 0.0)


@configclass
class SoArm101ReachDirectNewtonEnvCfg(SoArm101ReachDirectEnvCfg):
    # same env on the newton (mjwarp) backend, with the manager-based Newton task's
    # solver settings.

    def __post_init__(self):
        super().__post_init__()
        self.sim.physics = ReachPhysicsCfg().newton_mjwarp
        self.sim.physics.num_substeps = 2
        self.sim.physics.solver_cfg.njmax = 200
        self.sim.physics.collision_decimation = 4
        self.sim.physics.solver_cfg.integrator = "implicitfast"
