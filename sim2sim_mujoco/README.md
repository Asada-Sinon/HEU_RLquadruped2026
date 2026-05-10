# MuJoCo Sim2Sim for MyQuad

这个目录是独立的 MuJoCo 验证工具，不依赖 IsaacLab 运行时，也不修改 `source/rl_first` 里的 extension 代码。

## Install

在你准备用来跑 MuJoCo 的 Python 环境里安装依赖：

```bash
python -m pip install -r sim2sim_mujoco/requirements.txt
```

当前仓库环境可能还没有 `mujoco` 和 `onnxruntime`。如果运行时报 `ModuleNotFoundError`，先安装上面的依赖，或者切换到已经安装这些包的环境。

## Run

默认使用当前项目里的机器人 URDF 和最新 `my_quad_flat_v2` 导出策略：

```bash
python sim2sim_mujoco/run_mujoco.py --config sim2sim_mujoco/configs/my_quad.yaml
```

无窗口 smoke test：

```bash
python sim2sim_mujoco/run_mujoco.py --no-viewer --duration 2
```

覆盖策略或 URDF：

```bash
python sim2sim_mujoco/run_mujoco.py --policy path/to/policy.onnx
python sim2sim_mujoco/run_mujoco.py --urdf path/to/robot.urdf
```

## Keyboard

打开 viewer 后可以实时调速度命令：

- `W/S`: 增减 `vx`
- `A/D`: 增减 `vy`
- `Q/E`: 增减 `wz`
- `Space`: 速度指令清零

默认命令是慢速向前走：`vx=0.2, vy=0.0, wz=0.0`。

## Notes

观测顺序按 IsaacLab 训练配置构造：

```text
base_lin_vel, base_ang_vel, projected_gravity, velocity_commands,
joint_pos_rel, joint_vel_rel, last_action, height_scan
```

当前策略训练时包含 `height_scan`，这里先用固定平地值填充 187 维高度扫描，占位值默认是 `0.0`，可通过 `--height-scan-value` 调整。这适合先做平地 sim2sim 排查模型、关节顺序、动作缩放和 PD 控制；后续上实机前建议重新训练一个不依赖 `height_scan` 的策略。

控制频率按训练配置设置为 50 Hz：MuJoCo 物理步长 `0.005s`，策略每 `0.02s` 推理一次。动作转换为关节目标：

```text
target_joint_pos = default_joint_pos + 0.3 * action
```

然后使用 PD 力矩控制，并裁剪到 `23.7 Nm`。

默认会在 MuJoCo 加载时给 URDF 临时补一个 `floating_base` joint 和静态平地 collision。MuJoCo 直接读取普通 URDF 时根链接可能会被固定到世界，而 IsaacLab 训练配置里 `fix_base: false`；没有地面时浮动基座又会直接下落，所以 sim2sim 需要同时补这两件事。
