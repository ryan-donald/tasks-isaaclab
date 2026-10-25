# Copyright (c) 2024-2026, Ryan Donald
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""SO-ARM101 reach with normalized joint observations and actions in [-100, +100],
matching the real robot's servo interface."""

import math

from isaaclab.managers import (
    EventTermCfg as EventTerm,
)
from isaaclab.managers import (
    ObservationGroupCfg as ObsGroup,
)
from isaaclab.managers import (
    ObservationTermCfg as ObsTerm,
)
from isaaclab.managers import (
    RewardTermCfg as RewTerm,
)
from isaaclab.managers import (
    SceneEntityCfg,
)
from isaaclab.utils.configclass import configclass
from isaaclab.utils.noise import NoiseModelWithAdditiveBiasCfg
from isaaclab.utils.noise import UniformNoiseCfg as Unoise
from isaaclab_tasks.core.reach.reach_env_cfg import (
    ReachEnvCfg,
    ReachPhysicsCfg,
)

from ryan_tasks.robots.so_arm101 import (
    ACTION_DEADBAND,
    MAX_ACCEL_RAD_S2,
    MAX_SLEW_RAD_S,
    POS_QUANTUM,
    SO_ARM101_URDF_CFG,
)
from ryan_tasks.tasks.reach import mdp


def reach_physics() -> ReachPhysicsCfg:
    """Isaac Lab's reach physics presets with Isaac Sim PhysX as the default.

    ``physics=newton_mjwarp`` selects Newton (MJWarp) with the solver settings tuned
    for this arm.
    """
    physics = ReachPhysicsCfg()
    # Isaac Sim PhysX by name (PhysxAutoCfg picks OvPhysX when Kit is not running).
    physx = physics.isaacsim_physx
    # randomized reset poses can spawn links in contact, overflowing the default
    # 163840-patch GPU buffer.
    physx.gpu_max_rigid_patch_count = 2**18
    physics.physx = physx
    physics.default = physx
    newton = physics.newton_mjwarp
    newton.num_substeps = 2
    newton.solver_cfg.njmax = 200
    newton.collision_decimation = 4
    newton.solver_cfg.integrator = "implicitfast"
    return physics


@configclass
class SoArm101ReachNormalizedEnvCfg(ReachEnvCfg):
    """SO-ARM101 reach environment with normalized observations and actions.

    Observations: joint positions in normalized [-100, +100] space (not radians);
    velocities are still in rad/s. Actions: joint position commands in normalized
    [-100, +100] space.
    """

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 8192

        self.scene.robot = SO_ARM101_URDF_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        self.scene.table = None
        self.scene.ground.init_state.pos = (0.0, 0.0, 0.0)

        # end effector: the gripper tip.
        self.rewards.end_effector_position_tracking.params["asset_cfg"].body_names = [
            "gripper_frame_link"
        ]
        # fine-grained position tracking near the goal.
        self.rewards.end_effector_position_tracking_fine_grained = RewTerm(
            func=mdp.position_command_error_tanh,
            weight=0.1,
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=["gripper_frame_link"]),
                "std": 0.1,
                "command_name": "ee_pose",
            },
        )
        self.rewards.end_effector_orientation_tracking.params[
            "asset_cfg"
        ].body_names = ["gripper_frame_link"]

        # goal workspace.
        self.commands.ee_pose.ranges.pos_x = (0.1, 0.25)
        self.commands.ee_pose.ranges.pos_y = (-0.2, 0.2)
        self.commands.ee_pose.ranges.pos_z = (0.02, 0.4)

        self.commands.ee_pose.ranges.pitch = (math.pi, math.pi)

        self.commands.ee_pose.body_name = "gripper_frame_link"
        # skips the per-step error metrics, which only feed logging.
        self.commands.ee_pose.class_type = mdp.UniformPoseCommandNoMetrics

        self.sim.physics = reach_physics()

        # 60 Hz physics and control.
        hz = 60.0
        self.sim.dt = 1.0 / hz
        self.decimation = 1
        self.sim.render_interval = self.decimation
        self.episode_length_s = 12.0

        # smoothness pressure from step 0, since the linear policy head has no bound
        # on its commands.
        self.rewards.action_rate.weight = -0.001
        self.rewards.joint_vel.weight = -0.001
        self.rewards.action_l2 = RewTerm(func=mdp.action_l2, weight=0.0)

        # unused base terms: action_magnitude duplicates action_l2, and the success
        # bonus/termination would end episodes early.
        self.rewards.action_magnitude = None
        self.rewards.success = None
        self.terminations.success = None

        # strong action-rate penalty within 5 cm of the goal, so the approach stays
        # fast and the arm settles at the goal.
        self.rewards.action_rate_near = RewTerm(
            func=mdp.action_rate_l2_near_goal,
            weight=-0.15,
            params={
                "command_name": "ee_pose",
                "radius": 0.05,
                "asset_cfg": SceneEntityCfg("robot", body_names=["gripper_frame_link"]),
            },
        )

        # bonus for holding a still arm command at the goal: 1 cm goal kernel; 3
        # normalized units (0.03 action units) of command change per step.
        self.rewards.settled_at_goal = RewTerm(
            func=mdp.settled_at_goal,
            weight=0.1,
            params={
                "command_name": "ee_pose",
                "asset_cfg": SceneEntityCfg("robot", body_names=["gripper_frame_link"]),
                "goal_std": 0.01,
                "rate_scale": 0.03,
            },
        )

        self.curriculum.action_rate = None
        self.curriculum.joint_vel = None

        # arm position control in normalized [-100, 100] space.
        self.actions.arm_action = mdp.NormalizedJointPositionActionCfg(
            class_type=mdp.NormalizedJointPositionAction,
            asset_name="robot",
            joint_names=["shoulder_.*", "elbow_flex", "wrist_flex"],
            scale=100.0,
            use_default_offset=True,
        )

        # delay buffer depth; must be >= the max_delay of randomize_action_delay.
        self.actions.arm_action.delay_steps = 3

        # servo deadband, speed and acceleration limits.
        self.actions.arm_action.deadband = ACTION_DEADBAND
        self.actions.arm_action.max_slew_rad_s = MAX_SLEW_RAD_S
        self.actions.arm_action.max_accel_rad_s2 = MAX_ACCEL_RAD_S2

        # position only: the gripper orientation is not scored.
        self.rewards.end_effector_orientation_tracking = None

        obs_joints = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex"]

        @configclass
        class NormalizedPolicyCfg(ObsGroup):
            """Policy observations in normalized [-100, +100] space."""

            # joint positions relative to default, snapped to the encoder step.
            joint_pos = ObsTerm(
                func=mdp.joint_pos_normalized_100_rel,
                params={
                    "asset_cfg": SceneEntityCfg("robot", joint_names=obs_joints),
                    "pos_quantum": POS_QUANTUM,
                },
                # per-step noise plus a per-episode calibration offset.
                noise=NoiseModelWithAdditiveBiasCfg(
                    noise_cfg=Unoise(n_min=-0.3, n_max=0.3),
                    bias_noise_cfg=Unoise(n_min=-0.7, n_max=0.7, operation="abs"),
                ),
            )

            # joint velocities (rad/s), a finite difference of quantized positions, as
            # on the real robot.
            joint_vel = ObsTerm(
                func=mdp.joint_vel_finite_diff,
                params={
                    "velocity_scale": 1.0,
                    "asset_cfg": SceneEntityCfg("robot", joint_names=obs_joints),
                    "pos_quantum": POS_QUANTUM,
                },
                noise=Unoise(n_min=-0.01, n_max=0.01),
            )

            # goal pose.
            pose_command = ObsTerm(
                func=mdp.generated_commands, params={"command_name": "ee_pose"}
            )

            # the last 6 actions, so the policy can account for the action delay.
            actions = ObsTerm(func=mdp.last_action, history_length=6)

            def __post_init__(self):
                self.enable_corruption = True
                self.concatenate_terms = True

        self.observations.policy = NormalizedPolicyCfg()

        # randomized starting pose.
        self.events.reset_robot_joints = EventTerm(
            func=mdp.reset_joints_by_offset,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    joint_names=[
                        "shoulder_pan",
                        "shoulder_lift",
                        "elbow_flex",
                        "wrist_flex",
                    ],
                ),
                "position_range": (-0.2, 0.2),
                "velocity_range": (0.0, 0.0),
            },
        )

        # PD gain domain randomization: stiffness 75-125%, damping 50-150%.
        self.events.randomize_gains = EventTerm(
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

        # joint friction domain randomization.
        self.events.randomize_joint_friction = EventTerm(
            func=mdp.randomize_joint_parameters,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
                "friction_distribution_params": (0.7, 1.3),
                "operation": "scale",
                "distribution": "uniform",
            },
        )

        # armature domain randomization, 0.5-2x: the fit trades it against the latency.
        self.events.randomize_armature = EventTerm(
            func=mdp.randomize_joint_parameters,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
                "armature_distribution_params": (0.5, 2.0),
                "operation": "scale",
                "distribution": "log_uniform",
            },
        )

        # action delay of 1-3 steps, around the real robot's 37.6 ms latency (2.25
        # steps at 60 Hz).
        self.events.randomize_action_delay = EventTerm(
            func=mdp.randomize_action_delay,
            mode="reset",
            params={"action_name": "arm_action", "min_delay": 1, "max_delay": 3},
        )

        for term in vars(self.rewards).values():
            if term is not None:
                term.weight *= 100.0


@configclass
class SoArm101ReachNormalizedEnvCfg_PLAY(SoArm101ReachNormalizedEnvCfg):
    """Evaluation config: no observation noise or domain randomization, delay fixed
    at the measured 2 steps."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 1.0
        self.observations.policy.enable_corruption = False
        self.events.randomize_gains = None
        self.events.randomize_joint_friction = None
        self.events.randomize_armature = None
        self.events.randomize_action_delay = EventTerm(
            func=mdp.randomize_action_delay,
            mode="reset",
            params={"action_name": "arm_action", "min_delay": 2, "max_delay": 2},
        )

        self.observations.policy.joint_pos.noise = Unoise(n_min=-0.0, n_max=0.0)

        # Newton evaluation uses Isaac Lab's default Newton cfg at 8 substeps.
        self.sim.physics.newton_mjwarp = ReachPhysicsCfg().newton_mjwarp
        self.sim.physics.newton_mjwarp.num_substeps = 8

        # smaller goal/current pose markers (default 0.1).
        marker_scale = (0.04, 0.04, 0.04)
        self.commands.ee_pose.goal_pose_visualizer_cfg.markers[
            "frame"
        ].scale = marker_scale
        self.commands.ee_pose.current_pose_visualizer_cfg.markers[
            "frame"
        ].scale = marker_scale
