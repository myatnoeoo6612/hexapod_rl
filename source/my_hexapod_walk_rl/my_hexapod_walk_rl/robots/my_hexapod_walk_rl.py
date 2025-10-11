from isaaclab.assets import ArticulationCfg
from isaaclab.sim.spawners.from_files import UsdFileCfg
from isaaclab.actuators import ImplicitActuatorCfg

USD_PATH = "my_hexapod_flat_walk.usd"

MY_HEXAPOD_WALK_RL_CFG = ArticulationCfg(
    prim_path="/World/envs/env_.*/Robot",
    spawn=UsdFileCfg(
        usd_path=USD_PATH,
        scale=(1.0, 1.0, 1.0),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.05),
        rot=(1.0, 0.0, 0.0, 0.0),
        joint_pos={
            "hip_1_joint": 0.0, "knee_1_joint": 0.6,
            "hip_2_joint": 0.0, "knee_2_joint": 0.6,
            "hip_3_joint": 0.0, "knee_3_joint": 0.6,
            "hip_4_joint": 0.0, "knee_4_joint": 0.6,
            "hip_5_joint": 0.0, "knee_5_joint": 0.6,
            "hip_6_joint": 0.0, "knee_6_joint": 0.6,
        },
        joint_vel={".*": 0.0},
    ),
    actuators={
        "hip_actuators": ImplicitActuatorCfg(
            joint_names_expr=[
                "hip_1_joint", "hip_2_joint", "hip_3_joint",
                "hip_4_joint", "hip_5_joint", "hip_6_joint",
            ],
            stiffness=30.0,
            damping=3.0,
            effort_limit=4.0,
        ),
        "knee_actuators": ImplicitActuatorCfg(
            joint_names_expr=[
                "knee_1_joint", "knee_2_joint", "knee_3_joint",
                "knee_4_joint", "knee_5_joint", "knee_6_joint",
            ],
            stiffness=40.0,
            damping=4.0,
            effort_limit=4.0,
        ),
    },
)

