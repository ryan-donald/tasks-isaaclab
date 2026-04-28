from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

URDF_ASSETS_DATA_DIR = Path(__file__).resolve().parent

##
# Configuration - URDF Version
##

SO_ARM101_URDF_CFG = ArticulationCfg(
    spawn=sim_utils.UrdfFileCfg(
        fix_base=True,
        replace_cylinders_with_capsules=True,
        asset_path=f"{URDF_ASSETS_DATA_DIR}/so101_new_calib.urdf",
        activate_contact_sensors=False,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=5.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=0,
        ),
        joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
            gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0, damping=0)
        ),
    ),
    # default starting state
    init_state=ArticulationCfg.InitialStateCfg(
        rot=(1.0, 0.0, 0.0, 0.0),
        joint_pos={
            "shoulder_pan": 0.0,
            "shoulder_lift": 0.0,
            "elbow_flex": -0.0,
            "wrist_flex": 0.75,
            "wrist_roll": -0.0,
            "gripper": 0.0,
        },
        joint_vel={".*": 0.0},
    ),

    
    actuators={
        # shoulder pan, shoulder lift, elbow, wrist pitch, wrist roll, gripper
        "arm": ImplicitActuatorCfg(
            joint_names_expr=["shoulder_.*", "elbow_flex", "wrist_.*"],
            effort_limit_sim=1.9,
            velocity_limit_sim=4.0,
            stiffness={
                "shoulder_pan": 400.0,
                "shoulder_lift": 400.0,
                "elbow_flex": 400.0,
                "wrist_flex": 400.0,
                "wrist_roll": 600.0,
            },
            damping={
                "shoulder_pan": 40.0,
                "shoulder_lift": 40.0,
                "elbow_flex": 40.0,
                "wrist_flex": 40.0,
                "wrist_roll": 40.0,
            },
            friction={
                "shoulder_pan": 0.005,
                "shoulder_lift": 0.005,
                "elbow_flex": 0.005,
                "wrist_flex": 0.005,
                "wrist_roll": 0.005,
            },
        ),
        "gripper": ImplicitActuatorCfg(
            joint_names_expr=["gripper"],
            effort_limit_sim=2.5,
            velocity_limit_sim=3.18,
            stiffness=400.0,
            damping=40.0,
        ),
    },
    # there is no padding with safety for joint limits with 1.0. i.e. with 0.9, sim will tell agent [-100, 100],
    # but the actual joint range will be scaled to [-90, 90].
    soft_joint_pos_limit_factor=1.0,
)
