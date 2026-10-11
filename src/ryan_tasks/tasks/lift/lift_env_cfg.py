# Copyright (c) 2024-2026, Ryan Donald
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Robot-agnostic base for the cube lift tasks."""

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils.configclass import configclass
from isaaclab_tasks.core.lift.lift_env_cfg import LiftEnvCfg as CoreLiftEnvCfg

from ryan_tasks.tasks.lift import mdp


@configclass
class LiftEnvCfg(CoreLiftEnvCfg):
    """Manager-based cube lift base.

    Isaac Lab 3.0-EA's ``core.lift.LiftEnvCfg`` is a reorient task. This removes its
    reorient-specific rewards, randomization, observations and terminations and adds
    the lift reward slots, resets, termination and goal resampling the lift tasks use.
    """

    def __post_init__(self):
        super().__post_init__()

        # conditional_reset prefills a bank of grasp states and can loop forever when
        # it cannot fill it. The rest is the base's randomization suite (materials,
        # inertia and mass, joint gains and friction, gravity).
        self.events.conditional_reset = None
        self.events.joint_friction = None
        self.events.joint_stiffness_and_damping = None
        self.events.object_physics_inertia = None
        self.events.object_physics_material = None
        self.events.object_scale_mass = None
        self.events.robot_physics_material = None
        self.events.variable_gravity = None
        # scene and object resets.
        self.events.reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")
        self.events.reset_object_position = EventTerm(
            func=mdp.reset_root_state_uniform,
            mode="reset",
            params={
                "pose_range": {"x": (-0.1, 0.1), "y": (-0.25, 0.25), "z": (0.0, 0.0)},
                "velocity_range": {},
                "asset_cfg": SceneEntityCfg("object"),
            },
        )

        # the base's reorient rewards. Subclasses add an arm-only action_l2.
        self.rewards.fingers_to_object = None
        self.rewards.position_tracking = None
        self.rewards.orientation_tracking = None
        self.rewards.success = None
        self.rewards.early_termination = None
        self.rewards.action_l2 = None
        # lift reward slots, at default weights.
        self.rewards.reaching_object = RewTerm(
            func=mdp.object_ee_distance, params={"std": 0.1}, weight=1.0
        )
        self.rewards.lifting_object = RewTerm(
            func=mdp.object_is_lifted, params={"minimal_height": 0.04}, weight=15.0
        )
        self.rewards.action_rate = RewTerm(func=mdp.action_rate_l2, weight=-1e-4)
        self.rewards.joint_vel = RewTerm(
            func=mdp.joint_vel_l2,
            weight=-1e-4,
            params={"asset_cfg": SceneEntityCfg("robot")},
        )

        # the base's reorient observation groups, unused here.
        self.observations.perception = None
        self.observations.proprio = None

        # the base's out-of-bounds box requires z > 0.3, which a cube on the ground
        # fails. The episode ends only if the cube falls through the floor.
        self.terminations.object_out_of_bound = None
        self.terminations.abnormal_robot = None
        self.terminations.object_dropping = DoneTerm(
            func=mdp.root_height_below_minimum,
            params={"minimum_height": -0.05, "asset_cfg": SceneEntityCfg("object")},
        )
        # diverged physics, NaN states or a flung cube.
        self.terminations.physics_blowup = DoneTerm(func=mdp.physics_blowup)

        # goal resampled every 5 s, position only.
        self.commands.object_pose.resampling_time_range = (5.0, 5.0)
        self.commands.object_pose.ranges.roll = (0.0, 0.0)
        self.commands.object_pose.ranges.pitch = (0.0, 0.0)

        # the base's curriculum schedules what is cleared above; subclasses set their
        # own.
        self.curriculum = None

        self.scene.env_spacing = 2.5
