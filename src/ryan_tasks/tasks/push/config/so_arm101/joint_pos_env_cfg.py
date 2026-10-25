# Copyright (c) 2024-2026, Ryan Donald
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""SO-ARM101 cube push with normalized joint observations and actions.

Push a cube across the ground plane to a planar goal. The arm is observed and
commanded in normalized [-100, +100] space, as in the reach task. The gripper is not
an action: it is held closed and used as a rigid pusher.
"""

import copy

from isaaclab.assets import RigidObjectCfg
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
from isaaclab.sensors import ContactSensorCfg
from isaaclab.sim.schemas import MassCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.sim.spawners.materials.physics_materials_cfg import RigidBodyMaterialCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.configclass import configclass
from isaaclab.utils.noise import NoiseModelWithAdditiveBiasCfg
from isaaclab.utils.noise import UniformNoiseCfg as Unoise
from isaaclab_physx.sim.schemas import PhysxRigidBodyCfg
from isaaclab_tasks.core.lift.lift_env_cfg import LiftEnvCfg

from ryan_tasks.robots.so_arm101 import (
    GRIPPER_PRIM,
    JAW_PRIM,
    MAX_ACCEL_RAD_S2,
    MAX_SLEW_RAD_S,
    SO_ARM101_URDF_CFG,
    enable_nested_contact_reporting,
)
from ryan_tasks.tasks.lift.config.so_arm101.joint_pos_env_cfg import (
    SoArm101LiftPhysicsCfg,
)
from ryan_tasks.tasks.push import mdp

# gripper target, just off the lower stop so the soft limit does not carry the
# pushing load.
_GRIPPER_CLOSED = -0.17

# fixed wrist pose: the jaws side by side, presenting a flat face to the cube.
_WRIST_ROLL = -1.5221
_WRIST_FLEX = -0.3491

# 3 cm cube (the 6 cm DexCube at scale 0.5).
_CUBE_HALF_Z = 0.015
_CUBE_MASS = 0.07

# pusher contact point in the gripper_link frame, where the closed jaws meet.
_EE_OFFSET = (0.015, 0.0, -0.098)


@configclass
class SmoothnessCurriculumCfg:
    """Phases in the downforce, smoothness and cube-speed penalties after the push is
    learned (4000 iterations), at full weight from 7000.

    The weights are sized together: their combined cost stays near 30% of the
    task's return, so pushing remains worth more than avoiding the cube.
    ``num_steps`` counts env steps (iterations x 24); later terms override earlier
    ones once their threshold passes.
    """

    downforce_partial = CurrTerm(
        func=mdp.modify_reward_weight,
        params={
            "term_name": "gripper_downforce",
            "weight": -0.5,
            "num_steps": 4000 * 24,
        },
    )
    downforce_full = CurrTerm(
        func=mdp.modify_reward_weight,
        params={
            "term_name": "gripper_downforce",
            "weight": -1.5,
            "num_steps": 7000 * 24,
        },
    )

    # speed and chatter.
    joint_vel_partial = CurrTerm(
        func=mdp.modify_reward_weight,
        params={"term_name": "joint_vel", "weight": -0.06, "num_steps": 4000 * 24},
    )
    joint_vel_full = CurrTerm(
        func=mdp.modify_reward_weight,
        params={"term_name": "joint_vel", "weight": -0.2, "num_steps": 7000 * 24},
    )
    action_rate_partial = CurrTerm(
        func=mdp.modify_reward_weight,
        params={"term_name": "action_rate", "weight": -0.2, "num_steps": 4000 * 24},
    )
    action_rate_full = CurrTerm(
        func=mdp.modify_reward_weight,
        params={"term_name": "action_rate", "weight": -0.6, "num_steps": 7000 * 24},
    )

    object_lin_speed_partial = CurrTerm(
        func=mdp.modify_reward_weight,
        params={
            "term_name": "object_lin_speed",
            "weight": -0.25,
            "num_steps": 4000 * 24,
        },
    )
    object_lin_speed_full = CurrTerm(
        func=mdp.modify_reward_weight,
        params={
            "term_name": "object_lin_speed",
            "weight": -0.6,
            "num_steps": 7000 * 24,
        },
    )
    object_ang_speed_partial = CurrTerm(
        func=mdp.modify_reward_weight,
        params={
            "term_name": "object_ang_speed",
            "weight": -0.08,
            "num_steps": 4000 * 24,
        },
    )
    # gripper hull close to the floor.
    surface_proximity_partial = CurrTerm(
        func=mdp.modify_reward_weight,
        params={
            "term_name": "gripper_surface_proximity",
            "weight": -6.0,
            "num_steps": 4000 * 24,
        },
    )
    surface_proximity_full = CurrTerm(
        func=mdp.modify_reward_weight,
        params={
            "term_name": "gripper_surface_proximity",
            "weight": -18.0,
            "num_steps": 7000 * 24,
        },
    )

    object_ang_speed_full = CurrTerm(
        func=mdp.modify_reward_weight,
        params={
            "term_name": "object_ang_speed",
            "weight": -0.2,
            "num_steps": 7000 * 24,
        },
    )


@configclass
class SoArm101PushNormalizedEnvCfg(LiftEnvCfg):
    """SO-ARM101 push task, position-controlled arm joints in normalized [-100, 100].

    Built on the lift scene (robot, free rigid object, goal-pose command), with no
    gripper action, no lift or grasp rewards, and a planar goal on the ground.
    """

    def __post_init__(self):
        super().__post_init__()

        # turn off what Isaac Lab 3.0's lift base adds for its reorient task.
        # conditional_reset prefills a bank of grasp states and can loop forever
        # when it cannot fill it; a push needs none.
        self.events.conditional_reset = None
        # randomization replaced by this task's own events, or wrong for it (mass
        # scaling, variable gravity).
        self.events.joint_friction = None
        self.events.joint_stiffness_and_damping = None
        self.events.object_physics_inertia = None
        self.events.object_physics_material = None
        self.events.object_scale_mass = None
        self.events.robot_physics_material = None
        self.events.variable_gravity = None
        # scene and object resets. reset_all runs first, so the per-asset events
        # below offset a clean state.
        self.events.reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")
        self.events.reset_object_position = EventTerm(
            func=mdp.reset_root_state_uniform,
            mode="reset",
            params={
                "pose_range": {},
                "velocity_range": {},
                "asset_cfg": SceneEntityCfg("object"),
            },
        )

        # the base's reorient rewards; this task defines its own below.
        self.rewards.fingers_to_object = None
        self.rewards.position_tracking = None
        self.rewards.orientation_tracking = None
        self.rewards.success = None
        self.rewards.early_termination = None
        # reward slots the rest of this config and the curriculum refer to by name.
        self.rewards.reaching_object = RewTerm(func=mdp.object_ee_distance, weight=0.0)
        self.rewards.action_rate = RewTerm(func=mdp.action_rate_l2, weight=0.0)
        self.rewards.joint_vel = RewTerm(
            func=mdp.joint_vel_l2,
            params={"asset_cfg": SceneEntityCfg("robot")},
            weight=0.0,
        )

        # the base's reorient observation groups, unused here.
        self.observations.perception = None
        self.observations.proprio = None

        # out-of-bounds box around the SO-101 workspace, with the cube on the ground.
        self.terminations.object_out_of_bound.params["in_bound_range"] = {
            "x": (-0.5, 0.8),
            "y": (-0.6, 0.6),
            "z": (-0.05, 0.5),
        }

        # cube-on-ground friction, low enough for the cube to slide when pushed.
        self.sim.physics_material = RigidBodyMaterialCfg(
            static_friction=0.5,
            dynamic_friction=0.4,
            restitution=0.0,
        )
        self.sim.physics = SoArm101LiftPhysicsCfg()

        self.scene.robot = SO_ARM101_URDF_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.robot.init_state = copy.deepcopy(self.scene.robot.init_state)
        self.scene.robot.init_state.joint_pos = {
            **self.scene.robot.init_state.joint_pos,
            "gripper": _GRIPPER_CLOSED,
            "wrist_roll": _WRIST_ROLL,
            "wrist_flex": _WRIST_FLEX,
        }

        # contact sensing on the gripper needs the report API on the nested links.
        self.scene.robot.spawn = copy.deepcopy(self.scene.robot.spawn)
        self.scene.robot.spawn.activate_contact_sensors = True
        enable_nested_contact_reporting(self.scene.robot.spawn)

        # the sensors filter on the cube, so the ground force is net minus cube force
        # (the ground plane is a static collider and cannot be filtered against).
        self.scene.contact_gripper = ContactSensorCfg(
            prim_path=GRIPPER_PRIM,
            history_length=1,
            filter_prim_paths_expr=["{ENV_REGEX_NS}/Object"],
        )
        self.scene.contact_jaw = ContactSensorCfg(
            prim_path=JAW_PRIM,
            history_length=1,
            filter_prim_paths_expr=["{ENV_REGEX_NS}/Object"],
        )

        # no table: the ground plane at z=0 is the pushing surface.
        self.scene.table = None
        self.scene.table_material = None
        self.scene.plane.init_state.pos = (0.0, 0.0, 0.0)
        self.scene.ee_frame = None

        self.scene.object = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Object",
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=[0.25, 0.0, _CUBE_HALF_Z], rot=[0, 0, 0, 1]
            ),
            spawn=UsdFileCfg(
                usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/DexCube/dex_cube_instanceable.usd",
                scale=(0.5, 0.5, 0.5),
                mass_props=MassCfg(mass=_CUBE_MASS),
                rigid_props=PhysxRigidBodyCfg(
                    solver_position_iteration_count=16,
                    solver_velocity_iteration_count=1,
                    max_angular_velocity=1000.0,
                    max_linear_velocity=1000.0,
                    max_depenetration_velocity=5.0,
                    disable_gravity=False,
                ),
            ),
        )

        # 60 Hz physics and control.
        hz = 60.0
        self.sim.dt = 1.0 / hz
        self.decimation = 1
        self.sim.render_interval = self.decimation
        self.episode_length_s = 10.0

        # planar goal on the ground, in the robot root frame, at the cube's resting
        # height. The goal box is wider than the spawn box, so pushes go toward,
        # away from and across the base.
        self.commands.object_pose.body_name = ["gripper_link"]
        self.commands.object_pose.ranges.pos_x = (0.18, 0.30)
        self.commands.object_pose.ranges.pos_y = (-0.10, 0.10)
        self.commands.object_pose.ranges.pos_z = (_CUBE_HALF_Z, _CUBE_HALF_Z)
        # one goal per episode: a resampling time strictly longer than the episode
        # leaves the reset as the only resample.
        self.commands.object_pose.resampling_time_range = (
            self.episode_length_s + 1.0,
            self.episode_length_s + 1.0,
        )

        # arm position control in normalized space, as in the reach task. wrist_roll
        # is left out: no reward depends on it.
        self.actions.arm_action = mdp.NormalizedJointPositionActionCfg(
            class_type=mdp.NormalizedJointPositionAction,
            asset_name="robot",
            joint_names=["shoulder_.*", "elbow_flex", "wrist_flex"],
            scale=100.0,
            use_default_offset=True,
            delay_steps=3,
            max_slew_rad_s=MAX_SLEW_RAD_S,
            max_accel_rad_s2=MAX_ACCEL_RAD_S2,
        )
        # the gripper is a pusher, not an action.
        self.actions.gripper_action = None

        # rewards.
        # reach the cube's face.
        self.rewards.reaching_object.func = mdp.object_ee_distance
        self.rewards.reaching_object.params = {
            "std": 0.15,
            "ee_cfg": SceneEntityCfg("robot", body_names=["gripper_link"]),
            "ee_offset": _EE_OFFSET,
        }
        self.rewards.reaching_object.weight = 5.0

        self.rewards.lifting_object = None

        # progress toward the goal. It sums to the distance closed, independent of
        # time, so the speed penalties set the pace. Weight is reward per metre.
        self.rewards.object_goal_tracking = RewTerm(
            func=mdp.object_goal_progress,
            params={"command_name": "object_pose", "max_closing_speed": 2.0},
            weight=3000.0,
        )
        self.rewards.object_goal_tracking_fine_grained = None

        # per-step bonus within 2 cm of the goal, for precision near zero error.
        self.rewards.object_at_goal = RewTerm(
            func=mdp.object_at_goal,
            params={"threshold": 0.02, "command_name": "object_pose"},
            weight=1.0,
        )

        # smoothness and magnitude pressure from step 0 bound the linear policy head.
        # SmoothnessCurriculumCfg raises action_rate and joint_vel later.
        self.rewards.action_rate.weight = -0.001
        self.rewards.joint_vel.weight = -0.001
        # action magnitude: commands past the joint clamp get no task gradient, so
        # this is what pulls them back.
        self.rewards.action_l2 = RewTerm(
            func=mdp.action_l2,
            weight=-0.15,
        )

        # pressing the gripper down into the cube (vertical force only; the
        # horizontal force is the push).
        self.rewards.gripper_downforce = RewTerm(
            func=mdp.gripper_downforce_penalty,
            params={
                "sensor_names": ("contact_gripper", "contact_jaw"),
                "ref_force": 5.0,
                "max_cost": 400.0,
            },
            # phased in by SmoothnessCurriculumCfg.
            weight=0.0,
        )

        # the cube's own linear and angular speed.
        self.rewards.object_lin_speed = RewTerm(
            func=mdp.object_lin_speed_penalty,
            params={"ref_speed": 0.1, "max_cost": 100.0},
            # phased in by SmoothnessCurriculumCfg.
            weight=0.0,
        )
        self.rewards.object_ang_speed = RewTerm(
            func=mdp.object_ang_speed_penalty,
            params={"ref_speed": 5.0, "max_cost": 100.0},
            weight=0.0,
        )

        # the gripper hull approaching the floor, before any contact. body_cfg goes
        # through params so the manager resolves body_ids.
        self.rewards.gripper_surface_proximity = RewTerm(
            func=mdp.gripper_surface_proximity_penalty,
            params={
                "body_cfg": SceneEntityCfg(
                    "robot", body_names=["gripper_link", "moving_jaw_so101_v1_link"]
                ),
                "clearance": 0.015,
                "surface_z": 0.0,
                "max_cost": 25.0,
            },
            # phased in by SmoothnessCurriculumCfg.
            weight=0.0,
        )

        self.curriculum = SmoothnessCurriculumCfg()

        # cube spawn, relative to the init state: x 0.22-0.28, y +/-0.05.
        self.events.reset_object_position.params["pose_range"] = {
            "x": (-0.03, 0.03),
            "y": (-0.05, 0.05),
            "z": (0.0, 0.0),
        }

        # randomized starting pose. Only the reset state is offset; default_joint_pos,
        # and with it the action offset and relative observations, is unchanged.
        # Declared after reset_all so it is not overwritten.
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

        # randomized reset poses can spawn links in contact, overflowing the default
        # 163840-patch GPU buffer.
        self.sim.physics.physx.gpu_max_rigid_patch_count = 2**18
        self.sim.physics.default.gpu_max_rigid_patch_count = 2**18

        # wrist_roll and the gripper have no action term, so their position targets
        # are held here; otherwise they would be driven to 0.
        self.events.hold_wrist_roll = EventTerm(
            func=mdp.hold_joint_position_target,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=["wrist_roll"]),
                "position": _WRIST_ROLL,
            },
        )
        self.events.hold_gripper_closed = EventTerm(
            func=mdp.hold_joint_position_target,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=["gripper"]),
                "position": _GRIPPER_CLOSED,
            },
        )

        # PD gain domain randomization.
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

        # action delay of 1-3 steps, around the real robot's 37.6 ms latency (2.25
        # steps at 60 Hz).
        self.events.randomize_action_delay = EventTerm(
            func=mdp.randomize_action_delay,
            mode="reset",
            params={"action_name": "arm_action", "min_delay": 1, "max_delay": 3},
        )

        # high friction on the gripper bodies, so the cube is pushed rather than
        # sliding off the face.
        self.events.gripper_friction = EventTerm(
            func=mdp.randomize_rigid_body_material,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot", body_names=["gripper_link", "moving_jaw_so101_v1_link"]
                ),
                "static_friction_range": (2.5, 2.5),
                "dynamic_friction_range": (2.5, 2.5),
                "restitution_range": (0.0, 0.0),
                "num_buckets": 1,
            },
        )

        obs_joints = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex"]

        @configclass
        class NormalizedPolicyCfg(ObsGroup):
            """Policy observations: arm joints in normalized [-100, +100] space."""

            # arm joints relative to default, with per-step noise and a per-episode
            # calibration offset.
            joint_pos = ObsTerm(
                func=mdp.joint_pos_arm_normalized_100_rel,
                params={"joint_names": obs_joints},
                noise=NoiseModelWithAdditiveBiasCfg(
                    noise_cfg=Unoise(n_min=-0.3, n_max=0.3),
                    bias_noise_cfg=Unoise(n_min=-0.7, n_max=0.7, operation="abs"),
                ),
            )

            # joint velocities (rad/s), a finite difference of quantized positions,
            # as on the real robot.
            joint_vel = ObsTerm(
                func=mdp.joint_vel_finite_diff,
                params={
                    "velocity_scale": 1.0,
                    "asset_cfg": SceneEntityCfg("robot", joint_names=obs_joints),
                    "pos_quantum": 0.00182,  # encoder resolution, rad
                },
                noise=Unoise(n_min=-0.01, n_max=0.01),
            )

            # cube position in the robot root frame.
            object_position = ObsTerm(
                func=mdp.object_position_in_robot_root_frame,
                noise=Unoise(n_min=-0.005, n_max=0.005),
            )

            # goal position in the robot root frame.
            goal_position = ObsTerm(
                func=mdp.goal_position_in_robot_root_frame,
                params={"command_name": "object_pose"},
            )

            # the last 6 actions, so the policy can account for the action delay.
            actions = ObsTerm(func=mdp.last_action, history_length=6)

            def __post_init__(self):
                self.enable_corruption = True
                self.concatenate_terms = True

        self.observations.policy = NormalizedPolicyCfg()


@configclass
class SoArm101PushNormalizedEnvCfg_PLAY(SoArm101PushNormalizedEnvCfg):
    """Evaluation config: no observation noise or sim-to-real randomization."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 1.0
        self.observations.policy.enable_corruption = False
        self.observations.policy.joint_pos.noise = Unoise(n_min=-0.0, n_max=0.0)
        # no curriculum: weights fixed at their final training values.
        self.curriculum = None
        # Push-T has no downforce term.
        if self.rewards.gripper_downforce is not None:
            self.rewards.gripper_downforce.weight = -1.5
        self.rewards.gripper_surface_proximity.weight = -18.0
        self.rewards.joint_vel.weight = -0.2
        self.rewards.action_rate.weight = -0.6
        self.rewards.object_lin_speed.weight = -0.6
        self.rewards.object_ang_speed.weight = -0.2
        self.events.randomize_gains = None
        self.events.randomize_joint_friction = None
        # 2 steps, the measured latency.
        self.events.randomize_action_delay.params["min_delay"] = 2
        self.events.randomize_action_delay.params["max_delay"] = 2

        # smaller goal/current pose markers (default 0.1).
        marker_scale = (0.04, 0.04, 0.04)
        self.commands.object_pose.goal_pose_visualizer_cfg.markers[
            "frame"
        ].scale = marker_scale
        self.commands.object_pose.curr_pose_visualizer_cfg.markers[
            "frame"
        ].scale = marker_scale
