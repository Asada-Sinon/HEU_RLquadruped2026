# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass

from . import mdp

##
# 预定义配置
##

from isaaclab_assets.robots.cartpole import CARTPOLE_CFG  # isort:skip


##
# 场景定义
##


@configclass
class RlFirstSceneCfg(InteractiveSceneCfg):
    """小车-倒立摆场景配置。

    你可以在这里改“世界里有什么”：地面、机器人、灯光等。
    这是最直观、最容易上手的一块。
    """

    # 地面：给机器人一个可交互的平面
    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(size=(100.0, 100.0)),
    )

    # 机器人本体：基于 Isaac Lab 自带的 CARTPOLE_CFG，并把它放到每个并行环境内
    # 新手常改：替换成你自己的机器人配置（只要接口兼容）
    robot: ArticulationCfg = CARTPOLE_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    # 场景灯光：只影响可视化，不直接影响策略学习
    dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(color=(0.9, 0.9, 0.9), intensity=500.0),
    )


##
# MDP（任务）设置
##


@configclass
class ActionsCfg:
    """动作空间配置。

    决定“策略输出的动作”如何施加到机器人上。
    """

    # joint_effort：让策略输出作用在 slider_to_cart 这个关节上的力/力矩
    # scale 是动作缩放系数：
    # - 变大：动作更“猛”，更容易抖动或发散
    # - 变小：动作更“稳”，但可能学得慢或控制不住
    joint_effort = mdp.JointEffortActionCfg(asset_name="robot", joint_names=["slider_to_cart"], scale=100.0)


@configclass
class ObservationsCfg:
    """观测空间配置。

    决定“策略在每一步能看到什么状态”。
    """

    @configclass
    class PolicyCfg(ObsGroup):
        """给策略网络使用的观测组。"""

        # 观测项（顺序会保留，并按该顺序拼接）
        # joint_pos_rel：关节相对位置
        joint_pos_rel = ObsTerm(func=mdp.joint_pos_rel)
        # joint_vel_rel：关节相对速度
        joint_vel_rel = ObsTerm(func=mdp.joint_vel_rel)

        def __post_init__(self) -> None:
            # 关闭观测噪声/扰动，先保证训练稳定，适合新手调试
            self.enable_corruption = False
            # 将多个观测项拼接成一个向量，供策略网络直接输入
            self.concatenate_terms = True

    # 策略使用的观测组入口
    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """事件配置（通常用于 reset 随机化）。

    事件是“在某个时机触发的一次性操作”，最常见就是每回合重置时随机初始状态。
    """

    # 重置时：随机小车位置和速度
    reset_cart_position = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=["slider_to_cart"]),
            "position_range": (-1.0, 1.0),
            "velocity_range": (-0.5, 0.5),
        },
    )

    # 重置时：随机杆子的角度和角速度
    # 这里用到了 pi，表示范围是 +-0.25*pi
    reset_pole_position = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=["cart_to_pole"]),
            "position_range": (-0.25 * math.pi, 0.25 * math.pi),
            "velocity_range": (-0.25 * math.pi, 0.25 * math.pi),
        },
    )


@configclass
class RewardsCfg:
    """奖励函数配置。

    调参最常改这里：
    - 改权重（weight）= 改每个目标的重要性
    - 加/删奖励项 = 改学习目标
    """

    # (1) 存活奖励：每步给一个正奖励，鼓励尽量不失败
    alive = RewTerm(func=mdp.is_alive, weight=1.0)
    # (2) 终止惩罚：回合结束时给负奖励，惩罚失败
    terminating = RewTerm(func=mdp.is_terminated, weight=-2.0)
    # (3) 主任务：让杆子角度接近 0（直立）
    # 使用 L2 距离，离目标越远惩罚越大
    pole_pos = RewTerm(
        func=mdp.joint_pos_target_l2,
        weight=-1.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["cart_to_pole"]), "target": 0.0},
    )
    # (4) 形状奖励：抑制小车速度，减少无效左右乱冲
    cart_vel = RewTerm(
        func=mdp.joint_vel_l1,
        weight=-0.01,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["slider_to_cart"])},
    )
    # (5) 形状奖励：抑制杆子角速度，减少剧烈摆动
    pole_vel = RewTerm(
        func=mdp.joint_vel_l1,
        weight=-0.005,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["cart_to_pole"])},
    )


@configclass
class TerminationsCfg:
    """回合终止条件配置。"""

    # (1) 超时终止：达到 episode 长度后结束
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    # (2) 越界终止：小车位置超出 [-3, 3] 则结束回合
    cart_out_of_bounds = DoneTerm(
        func=mdp.joint_pos_out_of_manual_limit,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["slider_to_cart"]), "bounds": (-3.0, 3.0)},
    )


##
# 环境总配置
##


@configclass
class RlFirstEnvCfg(ManagerBasedRLEnvCfg):
    # 场景参数
    # num_envs：并行环境数，越大训练吞吐越高，但显存占用更大
    # env_spacing：并行环境在世界中的间距，避免互相干扰
    scene: RlFirstSceneCfg = RlFirstSceneCfg(num_envs=4096, env_spacing=4.0)
    # 基础模块
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    events: EventCfg = EventCfg()
    # MDP 核心：奖励与终止
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    # 后处理：补充一些全局仿真参数
    def __post_init__(self) -> None:
        """后初始化钩子。"""
        # 通用设置
        # decimation=2 表示“策略每 2 个物理步决策一次”
        # 控制频率 = 物理频率 / decimation
        self.decimation = 2
        # 每个回合时长（秒）
        self.episode_length_s = 5
        # 可视化相机位置（只影响观看）
        self.viewer.eye = (8.0, 0.0, 5.0)
        # 仿真设置
        # dt=1/120 -> 物理仿真频率 120Hz
        self.sim.dt = 1 / 120
        # 渲染间隔：通常和 decimation 一致，减少不必要渲染开销
        self.sim.render_interval = self.decimation