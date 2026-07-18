# Copyright (c) 2025-2026, Ryan Donald
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Normalized joint position environment configuration for SO-ARM101.

Uses normalized observations and actions in the [-100, +100] range to match the
real SO-101 robot hardware format.
"""

import math

from isaaclab.managers import (
    CurriculumTermCfg as CurrTerm,
)
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
from isaaclab_tasks.core.reach.reach_env_cfg import ReachEnvCfg, ReachPhysicsCfg

from ryan_tasks.tasks.robots.so_arm101_urdf_cfg import SO_ARM101_URDF_CFG

from . import mdp


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
        self.scene.robot = SO_ARM101_URDF_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        self.scene.table = None
        self.scene.ground.init_state.pos = (0.0, 0.0, 0.0)

        # set the body name for the end effector (gripper tip) in rewards.
        self.rewards.end_effector_position_tracking.params["asset_cfg"].body_names = [
            "gripper_frame_link"
        ]
        self.rewards.end_effector_position_tracking_fine_grained.params[
            "asset_cfg"
        ].body_names = ["gripper_frame_link"]
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

        # curriculum terms that allow the robot to first learn how to reach the goal,
        # then progressively learn smoother and slower motions to reach it.
        self.curriculum.action_rate.params["num_steps"] = 4_000 * 24
        self.curriculum.action_rate.params["weight"] = -0.001
        # self.curriculum.action_rate_s2 = CurrTerm(
        #     func=mdp.modify_reward_weight,
        #     params={
        #         "term_name": "action_rate",
        #         "weight": -0.005,
        #         "num_steps": 8_000 * 24,
        #     },
        # )
        # self.curriculum.action_rate_s3 = CurrTerm(
        #     func=mdp.modify_reward_weight,
        #     params={
        #         "term_name": "action_rate",
        #         "weight": -0.01,
        #         "num_steps": 12_000 * 24,
        #     },
        # )

        self.curriculum.joint_vel.params["num_steps"] = 4_000 * 24
        self.curriculum.joint_vel.params["weight"] = -0.001
        # self.curriculum.joint_vel_s2 = CurrTerm(
        #     func=mdp.modify_reward_weight,
        #     params={
        #         "term_name": "joint_vel",
        #         "weight": -0.004,
        #         "num_steps": 8_000 * 24,
        #     },
        # )
        # self.curriculum.joint_vel_s3 = CurrTerm(
        #     func=mdp.modify_reward_weight,
        #     params={
        #         "term_name": "joint_vel",
        #         "weight": -0.008,
        #         "num_steps": 12_000 * 24,
        #     },
        # )

        # arm is controlled using position control, in normalized ranges [-100, 100].
        # matches lerobot.
        self.actions.arm_action = mdp.NormalizedJointPositionActionCfg(
            class_type=mdp.NormalizedJointPositionAction,
            asset_name="robot",
            joint_names=["shoulder_.*", "elbow_flex", "wrist_flex"],
            scale=100.0,
            use_default_offset=True,
        )

        # maximum number of steps for action delay.
        self.actions.arm_action.delay_steps = 6

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
                params={"asset_cfg": SceneEntityCfg("robot", joint_names=obs_joints)},
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

        # action delay domain randomization, on the real robot I measured around 50-70ms
        # of delay.
        self.events.randomize_action_delay = EventTerm(
            func=mdp.randomize_action_delay,
            mode="reset",
            params={"action_name": "arm_action", "min_delay": 4, "max_delay": 6},
        )


@configclass
class SoArm101ReachNormalizedEnvCfg_PLAY(SoArm101ReachNormalizedEnvCfg):
    # configuration of task used for play. disables domain randomization.

    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # disable randomization for play
        self.observations.policy.enable_corruption = False
        # disable sim2real domain randomization so play runs under nominal
        self.events.randomize_gains = None
        self.events.randomize_joint_friction = None
        self.events.randomize_action_delay = EventTerm(
            func=mdp.randomize_action_delay,
            mode="reset",
            params={"action_name": "arm_action", "min_delay": 5, "max_delay": 5},
        )

        self.observations.policy.joint_pos.noise = Unoise(n_min=-0.0, n_max=0.0)


# domain-randomization isolation configs. each is the clean PLAY config (nominal
# gains, no obs noise, fixed delay) with exactly ONE training-time randomization
# re-enabled at its train setting. playing the same agent across these three
# isolates which term produces the jitter/orbit seen in the train task.


@configclass
class SoArm101ReachNormalizedPlayNoiseEnvCfg(SoArm101ReachNormalizedEnvCfg_PLAY):
    # PLAY + training observation noise (joint_pos +/-1.0) only.
    def __post_init__(self):
        super().__post_init__()
        self.observations.policy.enable_corruption = True
        self.observations.policy.joint_pos.noise = Unoise(n_min=-1.0, n_max=1.0)


@configclass
class SoArm101ReachNormalizedPlayGainsEnvCfg(SoArm101ReachNormalizedEnvCfg_PLAY):
    # PLAY + training PD-gain randomization (stiffness [0.75, 1.25], damping
    # [0.5, 1.5]) only.
    def __post_init__(self):
        super().__post_init__()
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


@configclass
class SoArm101ReachNormalizedPlayDelayEnvCfg(SoArm101ReachNormalizedEnvCfg_PLAY):
    # PLAY + training action-delay randomization (4-6 steps) only.
    def __post_init__(self):
        super().__post_init__()
        self.events.randomize_action_delay = EventTerm(
            func=mdp.randomize_action_delay,
            mode="reset",
            params={"action_name": "arm_action", "min_delay": 4, "max_delay": 6},
        )


@configclass
class SoArm101ReachNormalizedEnvCfg_FIXEDDELAY(SoArm101ReachNormalizedEnvCfg):
    # version identical to the default version, but with action delay fixed at 5

    def __post_init__(self):
        super().__post_init__()
        self.events.randomize_action_delay = EventTerm(
            func=mdp.randomize_action_delay,
            mode="reset",
            params={"action_name": "arm_action", "min_delay": 5, "max_delay": 5},
        )

        # tight tracking term with a small std, gives a strong gradient within the
        # last couple cm of the goal so the agent locks on instead of hovering.
        self.rewards.end_effector_position_tracking_tight = RewTerm(
            func=mdp.position_command_error_tanh,
            weight=0.05,
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=["gripper_frame_link"]),
                "std": 0.02,
                "command_name": "ee_pose",
            },
        )
        self.rewards.end_effector_position_tracking_fine_grained.weight = 0.05

        self.observations.policy.joint_pos.noise = Unoise(n_min=-0.3, n_max=0.3)


@configclass
class SoArm101ReachNormalizedFinetuneEnvCfg(SoArm101ReachNormalizedEnvCfg):
    # fine tuning task specifically for using a pretrained checkpoint from the standard
    # task, and enhancing it for better accuracy at the goal.

    def __post_init__(self):
        super().__post_init__()

        # adjusting the reward function to provide a stronger signal to the agent for
        # placing the end-effector directly at the goal.
        self.rewards.end_effector_position_tracking_fine_grained.weight = 0.05
        self.rewards.end_effector_position_tracking_tight = RewTerm(
            func=mdp.position_command_error_tanh,
            weight=0.05,
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=["gripper_frame_link"]),
                "std": 0.03,
                "command_name": "ee_pose",
            },
        )


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
