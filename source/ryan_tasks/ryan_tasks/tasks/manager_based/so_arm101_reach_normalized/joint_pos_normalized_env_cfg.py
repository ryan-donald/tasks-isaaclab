# Copyright (c) 2024-2026, Ryan Donald
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Normalized joint position environment configuration for SO-ARM101.

Uses normalized observations and actions in the [-100, +100] range to match the
real SO-101 robot hardware format.
"""

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

from ryan_tasks.tasks.robots.so_arm101_urdf_cfg import SO_ARM101_URDF_CFG

from . import mdp

# hardware-model terms. quantum is the measured encoder resolution, rad. the
# deadband would be 2.0, the action_deadzone.py threshold (not measured), off.
POS_QUANTUM = 0.00182
ACTION_DEADBAND = 0.0

# servo speed limit, rad/s. Goal_Velocity 1000 -> 1.464, measured with
# so101_testing/velocity_limit_bench.py (2.729 uncapped). 0.0 disables.
MAX_SLEW_RAD_S = 1.464

# servo setpoint acceleration limit, rad/s^2, fitted to open-loop chirps on all four
# policy joints (60 best for pan/elbow/wrist, 75 for lift). 0.0 disables.
MAX_ACCEL_RAD_S2 = 60.0


@configclass
class SoArm101ReachNormalizedEnvCfg(ReachEnvCfg):
    """SO-ARM101 reach environment with normalized observations and actions.

    Observations: joint positions in normalized [-100, +100] space (not radians);
    velocities are still in rad/s. Actions: joint position commands in normalized
    [-100, +100] space.
    """

    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # use calibrated URDF directly to match LeRobot FK coordinate frame exactly.
        # updated specifically for my robot.
        self.scene.robot = SO_ARM101_URDF_CFG.replace(
            prim_path="{ENV_REGEX_NS}/Robot",
            actuators={
                **SO_ARM101_URDF_CFG.actuators,
                "arm": SO_ARM101_URDF_CFG.actuators["arm"].replace(friction=0.0),
            },
        )

        self.scene.table = None
        self.scene.ground.init_state.pos = (0.0, 0.0, 0.0)

        # set the body name for the end effector (gripper tip) in rewards.
        self.rewards.end_effector_position_tracking.params["asset_cfg"].body_names = [
            "gripper_frame_link"
        ]
        # isaaclab 3.0-EA dropped this term from the base reach cfg, so define it here
        # with the weight/std the task was tuned on.
        self.rewards.end_effector_position_tracking_fine_grained = RewTerm(
            func=mdp.position_command_error_tanh,
            weight=0.1,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot", body_names=["gripper_frame_link"]
                ),
                "std": 0.1,
                "command_name": "ee_pose",
            },
        )
        self.rewards.end_effector_orientation_tracking.params[
            "asset_cfg"
        ].body_names = ["gripper_frame_link"]

        # workspace bounds for SO-ARM101 end-effector targets.
        self.commands.ee_pose.ranges.pos_x = (0.1, 0.25)
        self.commands.ee_pose.ranges.pos_y = (-0.2, 0.2)
        self.commands.ee_pose.ranges.pos_z = (0.1, 0.4)

        self.commands.ee_pose.ranges.pitch = (math.pi, math.pi)

        # set command generator body name.
        self.commands.ee_pose.body_name = "gripper_frame_link"
        # skip the per-step error metrics, which only feed extras logging.
        self.commands.ee_pose.class_type = mdp.UniformPoseCommandNoMetrics

        # randomized reset poses (reset_joints_by_offset) can spawn links in contact,
        # overflowing the default 163840-patch GPU buffer
        self.sim.physics.physx.gpu_max_rigid_patch_count = 2**18
        self.sim.physics.default.gpu_max_rigid_patch_count = 2**18

        # simulation update rates. 60hz default.
        hz = 60.0
        self.sim.dt = 1.0 / hz
        self.decimation = 1
        self.sim.render_interval = self.decimation
        self.episode_length_s = 12.0

        # LINEAR-HEAD SAFETY: the tanh policy head structurally bounded commands to
        # the joint range, so the smoothness penalties could be tiny and curriculum-
        # delayed. a linear head has no such bound, so it needs action-rate, velocity
        # and magnitude pressure from step 0 or it learns a saturating, non-settling
        # policy. make the smoothness terms active from the start and add an explicit
        # action-magnitude penalty (the direct replacement for the tanh bound, which
        # penalizes commanding position targets far from the default pose).
        self.rewards.action_rate.weight = -0.001
        self.rewards.joint_vel.weight = -0.001
        self.rewards.action_l2 = RewTerm(func=mdp.action_l2, weight=0.0)

        # isaaclab 3.0-EA added these to the base reach cfg; drop them to keep the
        # tuned mdp. action_magnitude duplicates action_l2 above, and the success
        # bonus/termination would end episodes early.
        self.rewards.action_magnitude = None
        self.rewards.success = None
        self.terminations.success = None

        # stepped smoothness: a strong action-rate penalty only within 5 cm of the goal,
        # so the approach stays fast while settling at the goal is enforced.
        self.rewards.action_rate_near = RewTerm(
            func=mdp.action_rate_l2_near_goal,
            weight=-0.15,
            params={
                "command_name": "ee_pose",
                "radius": 0.05,
                "asset_cfg": SceneEntityCfg("robot", body_names=["gripper_frame_link"]),
            },
        )

        self.curriculum.action_rate = None
        self.curriculum.joint_vel = None

        # arm is controlled using position control, in normalized ranges [-100, 100].
        # matches lerobot.
        self.actions.arm_action = mdp.NormalizedJointPositionActionCfg(
            class_type=mdp.NormalizedJointPositionAction,
            asset_name="robot",
            joint_names=["shoulder_.*", "elbow_flex", "wrist_flex"],
            scale=100.0,
            use_default_offset=True,
        )

        # maximum number of steps for action delay (the delay buffer depth, which
        # must be >= the max_delay sampled by randomize_action_delay below).
        self.actions.arm_action.delay_steps = 3

        # servo stiction and speed limit, see the constants above.
        self.actions.arm_action.deadband = ACTION_DEADBAND
        self.actions.arm_action.max_slew_rad_s = MAX_SLEW_RAD_S
        self.actions.arm_action.max_accel_rad_s2 = MAX_ACCEL_RAD_S2

        # ignore gripper orientation, just need to get the tip to the correct location.
        self.rewards.end_effector_orientation_tracking = None

        obs_joints = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex"]

        @configclass
        class NormalizedPolicyCfg(ObsGroup):
            """Observations for policy group - all in normalized [-100, +100] space."""

            # joint positions relative to default (0 is center of range of motion for
            # all except gripper, which is the closed position), in normalized space.
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

            # joint velocities (rad/s), computed as a finite difference in the same
            # manner as the lerobot deployment script.
            joint_vel = ObsTerm(
                func=mdp.joint_vel_finite_diff,
                params={
                    "velocity_scale": 1.0,
                    "asset_cfg": SceneEntityCfg("robot", joint_names=obs_joints),
                    "pos_quantum": POS_QUANTUM,
                },
                noise=Unoise(n_min=-0.01, n_max=0.01),
            )

            # target pose command, same as standard environments.
            pose_command = ObsTerm(
                func=mdp.generated_commands, params={"command_name": "ee_pose"}
            )

            # action buffer including the last 6 actions taken by the actor. allows
            # the network to learn through the real-world robot motor delays.
            actions = ObsTerm(func=mdp.last_action, history_length=6)

            def __post_init__(self):
                self.enable_corruption = True
                self.concatenate_terms = True

        # replaces observation with normalized observation matching lerobot.
        self.observations.policy = NormalizedPolicyCfg()

        # reset joints with an additive offset.
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

        # PD gain domain randomization, in the range [50%, 150%] of the nominal value
        # for damping and [75%, 125%] of the nominal value for stiffness.
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

        # action delay domain randomization. Real-robot latency (observation ->
        # servo starts tracking the new goal, excluding the servo's own rise time,
        # which the actuator gains already model) was measured 2026-08-26 at
        # 37.6 ms (p05 27, p95 46) by three independent methods -- step-onset
        # back-extrapolation, a chirp FRF fit, and the host pipeline timing.
        # At 60 Hz that is 2.25 steps, so 1-3.
        # This replaces an earlier 4-6, which came from a 5%-threshold reading of a
        # step response -- on a soft servo that threshold is set by plant dynamics,
        # not latency, and overestimated by ~2x. See so101_testing/
        # measure_action_latency.py and its action_latency.npy.
        self.events.randomize_action_delay = EventTerm(
            func=mdp.randomize_action_delay,
            mode="reset",
            params={"action_name": "arm_action", "min_delay": 1, "max_delay": 3},
        )


@configclass
class SoArm101ReachNormalizedEnvCfg_PLAY(SoArm101ReachNormalizedEnvCfg):
    # configuration of task used for play. disables domain randomization.

    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 1.0
        # disable randomization for play
        self.observations.policy.enable_corruption = False
        # disable sim2real domain randomization so play runs under nominal
        self.events.randomize_gains = None
        self.events.randomize_joint_friction = None
        self.events.randomize_action_delay = EventTerm(
            func=mdp.randomize_action_delay,
            mode="reset",
            params={"action_name": "arm_action", "min_delay": 2, "max_delay": 2},
        )

        self.observations.policy.joint_pos.noise = Unoise(n_min=-0.0, n_max=0.0)

        # shrink the goal/current pose frame markers (default is 0.1)
        marker_scale = (0.04, 0.04, 0.04)
        self.commands.ee_pose.goal_pose_visualizer_cfg.markers[
            "frame"
        ].scale = marker_scale
        self.commands.ee_pose.current_pose_visualizer_cfg.markers[
            "frame"
        ].scale = marker_scale


@configclass
class SoArm101ReachNormalizedNewtonEnvCfg(SoArm101ReachNormalizedEnvCfg):
    # reach env for newton backend, instead of default physx

    def __post_init__(self):
        super().__post_init__()
        # force the Newton MJWarp backend, reusing the parent preset's tuned cfg
        self.sim.physics = ReachPhysicsCfg().newton_mjwarp
        self.sim.physics.num_substeps = 2
        self.sim.physics.solver_cfg.njmax = 200
        self.sim.physics.collision_decimation = 4
        self.sim.physics.solver_cfg.integrator = "implicitfast"


@configclass
class SoArm101ReachNormalizedNewtonEnvCfg_PLAY(SoArm101ReachNormalizedEnvCfg_PLAY):
    # play/eval env for newton backend

    def __post_init__(self):
        super().__post_init__()
        self.sim.physics = ReachPhysicsCfg().newton_mjwarp
        self.sim.physics.num_substeps = 8
