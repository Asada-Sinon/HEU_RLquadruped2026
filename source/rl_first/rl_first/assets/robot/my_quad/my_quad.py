import os

import isaaclab.sim as sim_utils
from isaaclab.actuators import DCMotorCfg
from isaaclab.assets.articulation import ArticulationCfg


MY_QUAD_DIR = os.path.dirname(__file__).replace("\\", "/")

MY_QUAD_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{MY_QUAD_DIR}/my_12dof_quadruped.usd",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=100.0,
            max_angular_velocity=100.0,
            max_depenetration_velocity=5.0,
            enable_gyroscopic_forces=True,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=0,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.37),
        joint_pos={
            "LF_HAA": -0.027960174616949163,
            "LF_HFE":  1.0191152035320088,
            "LF_KFE": -1.5190473144732646,

            "RF_HAA":  0.027960174616949163,
            "RF_HFE":  1.0191152035320088,
            "RF_KFE": -1.5190473144732646,

            "LH_HAA": -0.027960174616949163,
            "LH_HFE":  1.0191152035320088,
            "LH_KFE": -1.5190473144732646,

            "RH_HAA":  0.027960174616949163,
            "RH_HFE":  1.0191152035320088,
            "RH_KFE": -1.5190473144732646,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "legs": DCMotorCfg(
            joint_names_expr=[".*_HAA", ".*_HFE", ".*_KFE"],
            effort_limit=23.7,
            saturation_effort=23.7,
            velocity_limit=30.0,
            stiffness=25.0,
            damping=0.5,
            friction=0.0,
        ),
    },
)
