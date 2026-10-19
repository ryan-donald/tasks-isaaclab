# Copyright (c) 2024-2026, Ryan Donald
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""SO-ARM101 cube lift with normalized joint observations and actions.

Arm joints are observed and commanded in normalized [-100, +100] space. The gripper
is a binary open/close action, observed in [0, 100] (0 = closed, 100 = open).
Joint velocities are in rad/s.
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
from isaaclab.sim.schemas.schemas_cfg import MassPropertiesCfg, RigidBodyPropertiesCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.sim.spawners.materials.physics_materials_cfg import RigidBodyMaterialCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.configclass import configclass
from isaaclab.utils.noise import NoiseModelWithAdditiveBiasCfg
from isaaclab.utils.noise import UniformNoiseCfg as Unoise
from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg
from isaaclab_physx.physics import PhysxCfg
from isaaclab_tasks.utils import PresetCfg

from ryan_tasks.robots.so_arm101 import (
    GRIPPER_PRIM,
    JAW_PRIM,
    MAX_ACCEL_RAD_S2,
    MAX_SLEW_RAD_S,
    POS_QUANTUM,
    SO_ARM101_URDF_CFG,
    enable_nested_contact_reporting,
)
from ryan_tasks.tasks.lift import mdp
from ryan_tasks.tasks.lift.lift_env_cfg import LiftEnvCfg


@configclass
class SoArm101LiftPhysicsCfg(PresetCfg):
    """Physics presets for the lift and push tasks. PhysX by default;
    ``physics=newton_mjwarp`` selects Newton (MJWarp)."""

    default: PhysxCfg = PhysxCfg(
        bounce_threshold_velocity=0.01,
        gpu_found_lost_aggregate_pairs_capacity=1024 * 1024 * 4,
        gpu_total_aggregate_pairs_capacity=2**21,
        friction_correlation_distance=0.00625,
    )
    physx: PhysxCfg = default
    newton_mjwarp: NewtonCfg = NewtonCfg(
        solver_cfg=MJWarpSolverCfg(
            solver="newton",
            integrator="implicitfast",
            njmax=150,
            nconmax=100,
            impratio=10.0,
            cone="elliptic",
            iterations=100,
            ls_iterations=15,
        ),
        num_substeps=8,
        debug_mode=False,
    )


@configclass
class TableContactCurriculumCfg:
    """Phases in the table-contact penalty once the lift has been learned.

    Penalizing contact from the start stops the policy from exploring down to the
    cube. ``num_steps`` counts env steps (iterations x 24); later terms override
    earlier ones once their threshold passes.
    """

    table_contact_partial = CurrTerm(
        func=mdp.modify_reward_weight,
        params={
            "term_name": "ee_table_contact",
            "weight": -5.0,
            "num_steps": 4000 * 24,
        },
    )
    table_contact_full = CurrTerm(
        func=mdp.modify_reward_weight,
        params={
            "term_name": "ee_table_contact",
            "weight": -20.0,
            "num_steps": 6000 * 24,
        },
    )


@configclass
class SmoothnessCurriculumCfg(TableContactCurriculumCfg):
    """Adds the arm command-rate and joint-velocity penalties while the lift is being
    learned. Without them the policy converges to a bang-bang controller."""

    arm_command_rate_partial = CurrTerm(
        func=mdp.modify_reward_weight,
        params={
            "term_name": "arm_command_rate",
            "weight": -0.5,
            "num_steps": 500 * 24,
        },
    )
    arm_command_rate_full = CurrTerm(
        func=mdp.modify_reward_weight,
        params={
            "term_name": "arm_command_rate",
            "weight": -1.2,
            "num_steps": 1000 * 24,
        },
    )
    joint_vel_partial = CurrTerm(
        func=mdp.modify_reward_weight,
        params={"term_name": "joint_vel", "weight": -0.05, "num_steps": 500 * 24},
    )
    joint_vel_full = CurrTerm(
        func=mdp.modify_reward_weight,
        params={"term_name": "joint_vel", "weight": -0.15, "num_steps": 1000 * 24},
    )


@configclass
class SoArm101LiftNormalizedEnvCfg(LiftEnvCfg):
    """SO-ARM101 lift task, position-controlled joints in normalized [-100, 100]
    space."""

    def __post_init__(self):
        super().__post_init__()

        self.sim.physics_material = RigidBodyMaterialCfg(
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        )
        self.sim.physics = SoArm101LiftPhysicsCfg()

        # arm joint friction is off: PhysX joint friction keeps the arm from settling.
        self.scene.robot = SO_ARM101_URDF_CFG.replace(
            prim_path="{ENV_REGEX_NS}/Robot",
            actuators={
                **SO_ARM101_URDF_CFG.actuators,
                "arm": SO_ARM101_URDF_CFG.actuators["arm"].replace(friction=0.0),
            },
        )
        # contact sensing on the gripper needs the report API on the nested links.
        self.scene.robot.spawn = copy.deepcopy(self.scene.robot.spawn)
        self.scene.robot.spawn.activate_contact_sensors = True
        enable_nested_contact_reporting(self.scene.robot.spawn)

        # the sensors filter on the cube so the grasp force can be subtracted out;
        # the ground plane is a static collider and cannot be filtered against.
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

        # no table: the ground plane at z=0 is the support surface.
        self.scene.table = None
        self.scene.plane.init_state.pos = (0.0, 0.0, 0.0)

        # default pose: a top-down pre-grasp hover 13 cm above the spawn centre, with
        # the gripper open and wrist_flex 15 deg off its upper stop. The arm action is
        # an offset from this pose. wrist_roll is not actuated by the policy and stays
        # at -1.5221, which closes the jaws along world -y, square to the cube.
        self.scene.robot.init_state.joint_pos = {
            **self.scene.robot.init_state.joint_pos,
            "shoulder_pan": -0.0737,
            "shoulder_lift": -0.338,
            "elbow_flex": 0.0318,
            "wrist_flex": 1.4967,
            "wrist_roll": -1.5221,
            "gripper": 1.6,
        }

        # end-effector frame. must be a list.
        self.commands.object_pose.body_name = ["gripper_link"]

        # 60 Hz physics and control.
        hz = 60.0
        self.sim.dt = 1.0 / hz
        self.decimation = 1
        self.sim.render_interval = self.decimation
        self.episode_length_s = 10.0

        # goal box where the gripper can carry the cube level with every joint far
        # from its limits.
        self.commands.object_pose.ranges.pos_x = (0.28, 0.33)
        self.commands.object_pose.ranges.pos_y = (-0.05, 0.05)
        self.commands.object_pose.ranges.pos_z = (0.22, 0.27)
        # one goal per episode: a resampling time longer than the episode leaves the
        # reset as the only resample.
        self.commands.object_pose.resampling_time_range = (
            self.episode_length_s + 1.0,
            self.episode_length_s + 1.0,
        )

        # arm position control in normalized space.
        self.actions.arm_action = mdp.NormalizedJointPositionActionCfg(
            class_type=mdp.NormalizedJointPositionAction,
            asset_name="robot",
            # wrist_roll is left out: no reward depends on it.
            joint_names=["shoulder_.*", "elbow_flex", "wrist_flex"],
            # one action unit moves a joint ~0.55 rad.
            scale=30.0,
            use_default_offset=True,
            delay_steps=3,
            # half the reach task's servo speed limit, for a slower lift.
            max_slew_rad_s=0.5 * MAX_SLEW_RAD_S,
            max_accel_rad_s2=MAX_ACCEL_RAD_S2,
        )

        # binary gripper, delayed the same way as the arm.
        self.actions.gripper_action = mdp.DelayedBinaryJointPositionActionCfg(
            class_type=mdp.DelayedBinaryJointPositionAction,
            asset_name="robot",
            joint_names=["gripper"],
            open_command_expr={"gripper": 1.6},
            close_command_expr={"gripper": -0.17},
            delay_steps=3,
        )

        # 3 cm cube (the 6 cm DexCube at scale 0.5).
        self.scene.object = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Object",
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=[0.22, 0.0, 0.015], rot=[0, 0, 0, 1]
            ),
            spawn=UsdFileCfg(
                usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/DexCube/dex_cube_instanceable.usd",
                scale=(0.5, 0.5, 0.5),
                mass_props=MassPropertiesCfg(mass=0.07),
                rigid_props=RigidBodyPropertiesCfg(
                    solver_position_iteration_count=16,
                    solver_velocity_iteration_count=1,
                    max_angular_velocity=1000.0,
                    max_linear_velocity=1000.0,
                    max_depenetration_velocity=5.0,
                    disable_gravity=False,
                ),
            ),
        )

        self.scene.ee_frame = None

        # rewards, as a ladder: reach (1) < close on the cube (1) < pinch (2) <
        # lift (15 + 5).
        self.rewards.reaching_object.func = mdp.object_ee_distance
        self.rewards.reaching_object.params = {
            "std": 0.1,
            "ee_cfg": SceneEntityCfg("robot", body_names=["gripper_link"]),
            # grasp point between the jaws, low enough that holding the cube centre
            # keeps the gripper hull clear of the floor.
            "ee_offset": (0.015, 0.0, -0.098),
        }
        self.rewards.reaching_object.weight = 1.0

        self.rewards.lifting_object.func = mdp.object_lift_shaped
        self.rewards.lifting_object.params = {
            "std": 0.05,
            "half_extent": 0.015,
            "table_z": 0.0,
            "reach_std": 0.1,
            "ee_cfg": SceneEntityCfg("robot", body_names=["gripper_link"]),
            "ee_offset": (0.015, 0.0, -0.098),
        }
        self.rewards.lifting_object.weight = 15.0

        self.rewards.lifting_success = RewTerm(
            func=mdp.object_lifted_clearance,
            params={
                "clearance": 0.01,
                "half_extent": 0.015,
                "table_z": 0.0,
            },
            weight=5.0,
        )

        self.rewards.grasping = None

        # both jaws pressing on the cube from opposite sides.
        self.rewards.object_pinched = RewTerm(func=mdp.object_pinched, weight=2.0)

        # closing the jaw while the cube is at the grasp point.
        self.rewards.jaw_closing_on_cube = RewTerm(
            func=mdp.object_grasped,
            params={
                "grasp_std": 0.02,
                "gripper_open": 1.6,
                "gripper_cfg": SceneEntityCfg("robot", joint_names=["gripper"]),
                "ee_cfg": SceneEntityCfg("robot", body_names=["gripper_link"]),
                "ee_offset": (0.015, 0.0, -0.098),
            },
            weight=1.0,
        )

        self.rewards.object_goal_tracking = RewTerm(
            func=mdp.object_goal_distance,
            params={"std": 0.3, "minimal_height": 0.04, "command_name": "object_pose"},
            weight=16.0,
        )
        self.rewards.object_goal_tracking_fine_grained = RewTerm(
            func=mdp.object_goal_distance,
            params={"std": 0.05, "minimal_height": 0.04, "command_name": "object_pose"},
            weight=5.0,
        )

        self.rewards.action_rate.weight = -0.001
        self.rewards.joint_vel.weight = -0.001

        # arm-only action magnitude: with use_default_offset this penalizes targets
        # far from the default pose, which bounds the unbounded policy mean.
        self.rewards.action_l2 = RewTerm(
            func=mdp.action_l2_term,
            params={"action_name": "arm_action"},
            weight=-0.05,
        )

        # linear restoring force for commands past the joint limits, where the task
        # reward is flat. Zero for any executable command.
        self.rewards.action_past_limits = RewTerm(
            func=mdp.action_past_limits,
            weight=-1.0,
        )

        # carry posture: fingers level once lifted, and arm joints off their limits.
        self.rewards.gripper_level = RewTerm(
            func=mdp.gripper_level,
            params={
                "minimal_height": 0.04,
                "ee_cfg": SceneEntityCfg("robot", body_names=["gripper_link"]),
            },
            weight=6.0,
        )
        self.rewards.joint_near_limits = RewTerm(
            func=mdp.joint_near_limits,
            params={
                "band": 0.1745,  # 10 deg
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    joint_names=[
                        "shoulder_pan",
                        "shoulder_lift",
                        "elbow_flex",
                        "wrist_flex",
                    ],
                ),
            },
            weight=-1.0,
        )

        # penalize pressing the gripper into the floor (force, not proximity, so a
        # grasp off the floor is still allowed).
        self.rewards.ee_table_contact = RewTerm(
            func=mdp.gripper_ground_force_penalty,
            params={
                "sensor_names": ("contact_gripper", "contact_jaw"),
                "deadband": 1.0,
                "max_force": 15.0,
            },
            # phased in by TableContactCurriculumCfg.
            weight=0.0,
        )

        # change in the executed arm command; phased in by SmoothnessCurriculumCfg.
        self.rewards.arm_command_rate = RewTerm(func=mdp.arm_command_rate, weight=0.0)

        self.curriculum = SmoothnessCurriculumCfg()

        # cube spawns at x 0.22-0.27, y +/-0.05 (offsets from the init position).
        self.events.reset_object_position.params["pose_range"] = {
            "x": (0.0, 0.05),
            "y": (-0.05, 0.05),
            "z": (0.0, 0.0),
        }

        # randomized starting pose. Only the reset state is offset; default_joint_pos,
        # and with it the action offset and relative observations, is unchanged.
        # Declared after the base's reset_all so it is not overwritten.
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

        # wrist_roll has no action term, so its position target is held here;
        # otherwise it would be driven to 0.
        self.events.hold_wrist_roll = EventTerm(
            func=mdp.hold_joint_position_target,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=["wrist_roll"]),
                "position": -1.5221,
            },
        )

        # motor PD domain randomization.
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
        # steps at 60 Hz). The arm and gripper delays are drawn independently.
        self.events.randomize_action_delay = EventTerm(
            func=mdp.randomize_action_delay,
            mode="reset",
            params={"action_name": "arm_action", "min_delay": 1, "max_delay": 3},
        )

        self.events.randomize_gripper_action_delay = EventTerm(
            func=mdp.randomize_action_delay,
            mode="reset",
            params={"action_name": "gripper_action", "min_delay": 1, "max_delay": 3},
        )

        # gripper friction.
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

        @configclass
        class NormalizedPolicyCfg(ObsGroup):
            """Policy observations: arm joints in [-100, +100] relative to default,
            gripper in [0, 100]."""

            # arm joints relative to default, snapped to the encoder step, with
            # per-step noise and a per-episode calibration offset.
            joint_pos_arm = ObsTerm(
                func=mdp.joint_pos_normalized_100_rel,
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
                    "pos_quantum": POS_QUANTUM,
                },
                noise=NoiseModelWithAdditiveBiasCfg(
                    noise_cfg=Unoise(n_min=-0.3, n_max=0.3),
                    bias_noise_cfg=Unoise(n_min=-0.7, n_max=0.7, operation="abs"),
                ),
            )

            # gripper position, 0 = closed, 100 = open.
            joint_pos_gripper = ObsTerm(
                func=mdp.joint_pos_gripper_normalized_0_100,
                noise=Unoise(n_min=-1.5, n_max=1.5),
            )

            # joint velocities (rad/s), a finite difference of the quantized
            # positions, as on the real robot.
            joint_vel = ObsTerm(
                func=mdp.joint_vel_finite_diff,
                params={
                    "velocity_scale": 1.0,
                    "pos_quantum": POS_QUANTUM,
                    # explicit names: the term re-resolves this cfg and a regex is
                    # rejected as inconsistent with the resolved joint ids.
                    "asset_cfg": SceneEntityCfg(
                        "robot",
                        joint_names=[
                            "shoulder_pan",
                            "shoulder_lift",
                            "elbow_flex",
                            "wrist_flex",
                            "gripper",
                        ],
                    ),
                },
                noise=Unoise(n_min=-0.01, n_max=0.01),
            )

            # cube position in the robot root frame.
            object_position = ObsTerm(
                func=mdp.object_position_in_robot_root_frame,
                noise=Unoise(n_min=-0.005, n_max=0.005),
            )

            # goal position in the robot root frame, noise-free.
            target_object_position = ObsTerm(
                func=mdp.command_position,
                params={"command_name": "object_pose"},
            )

            # the last 6 actions as executed (clamped arm targets, gripper sign), so
            # the policy can account for the action delay.
            actions = ObsTerm(func=mdp.effective_last_action, history_length=6)

            def __post_init__(self):
                self.enable_corruption = True
                self.concatenate_terms = True

        self.observations.policy = NormalizedPolicyCfg()


@configclass
class SoArm101LiftNormalizedEnvCfg_PLAY(SoArm101LiftNormalizedEnvCfg):
    """Evaluation config: no observation noise, delays fixed at the measured value."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False
        self.events.randomize_joint_friction = None
        # 2 steps, the measured latency.
        self.events.randomize_action_delay.params["min_delay"] = 2
        self.events.randomize_action_delay.params["max_delay"] = 2
        self.events.randomize_gripper_action_delay.params["min_delay"] = 2
        self.events.randomize_gripper_action_delay.params["max_delay"] = 2
        # no curriculum: weights fixed at their final training values.
        self.curriculum = None
        self.rewards.ee_table_contact.weight = -20.0
        self.rewards.arm_command_rate.weight = -1.2
        self.rewards.joint_vel.weight = -0.15
        self.commands.object_pose.ranges.pos_x = (0.28, 0.33)
        self.commands.object_pose.ranges.pos_y = (-0.05, 0.05)
        self.commands.object_pose.ranges.pos_z = (0.22, 0.27)


@configclass
class SoArm101LiftNormalizedSettleEnvCfg(SoArm101LiftNormalizedEnvCfg):
    """Lift with a bonus for holding a still arm command at the goal.

    Uses the reach task's action scale and light smoothness terms, starts a fraction
    of episodes from a banked hold state, and drops the terms that bound the arm
    command (action_l2, action_past_limits, the arm_command_rate / joint_vel ramps).
    """

    hold_fraction: float = 0.25

    def __post_init__(self):
        super().__post_init__()
        # one action unit spans about a joint's whole range.
        self.actions.arm_action.scale = 100.0

        self.rewards.action_l2 = None
        self.rewards.action_past_limits = None
        self.rewards.arm_command_rate = None
        self.rewards.action_rate.weight = -1.0
        self.curriculum = TableContactCurriculumCfg()

        self.commands.object_pose.class_type = mdp.HoldAwarePoseCommand
        self.events.hold_start = EventTerm(
            func=mdp.hold_start_reset,
            mode="reset",
            params={"fraction": self.hold_fraction},
        )
        # 1.5 cm goal kernel; 10 normalized units (0.1 action units) of command
        # change per step.
        self.rewards.settled_at_goal = RewTerm(
            func=mdp.settled_at_goal,
            weight=6.0,
            params={
                "goal_std": 0.015,
                "rate_scale": 0.1,
                "minimal_height": 0.04,
                "command_name": "object_pose",
            },
        )


@configclass
class SoArm101LiftNormalizedSettleEnvCfg_PLAY(SoArm101LiftNormalizedSettleEnvCfg):
    """Evaluation config for the Settle task: the base Play settings, no hold starts."""

    hold_fraction: float = 0.0

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.observations.policy.enable_corruption = False
        self.events.randomize_joint_friction = None
        self.events.randomize_action_delay.params["min_delay"] = 2
        self.events.randomize_action_delay.params["max_delay"] = 2
        self.events.randomize_gripper_action_delay.params["min_delay"] = 2
        self.events.randomize_gripper_action_delay.params["max_delay"] = 2
        self.curriculum = None
        self.rewards.ee_table_contact.weight = -20.0
        self.commands.object_pose.ranges.pos_x = (0.28, 0.33)
        self.commands.object_pose.ranges.pos_y = (-0.05, 0.05)
        self.commands.object_pose.ranges.pos_z = (0.22, 0.27)
