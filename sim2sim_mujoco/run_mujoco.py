"""Run an IsaacLab RSL-RL policy in MuJoCo for sim2sim validation."""

from __future__ import annotations

import argparse
import math
import tempfile
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

try:
    import mujoco
    import numpy as np
    import onnxruntime as ort
    import yaml
except ModuleNotFoundError as exc:
    missing = exc.name
    raise SystemExit(
        f"Missing dependency '{missing}'. Install with: "
        "python -m pip install -r sim2sim_mujoco/requirements.txt"
    ) from exc


REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="MuJoCo sim2sim runner for the MyQuad IsaacLab policy."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("sim2sim_mujoco/configs/my_quad.yaml"),
        help="Path to the sim2sim YAML config.",
    )
    parser.add_argument(
        "--policy", type=Path, default=None, help="Override ONNX policy path."
    )
    parser.add_argument(
        "--urdf", type=Path, default=None, help="Override robot URDF path."
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=None,
        help="Optional simulation duration in seconds.",
    )
    parser.add_argument(
        "--no-viewer", action="store_true", help="Run without the MuJoCo viewer."
    )
    parser.add_argument(
        "--height-scan-value",
        type=float,
        default=None,
        help="Deprecated; ignored because the sim2real policy no longer uses height_scan.",
    )
    parser.add_argument(
        "--log",
        type=Path,
        default=None,
        help="Optional .npz path for saving policy-step trace data.",
    )
    return parser.parse_args()


def resolve_path(path: str | Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return (REPO_ROOT / candidate).resolve()


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def quat_conj(q: np.ndarray) -> np.ndarray:
    return np.array([q[0], -q[1], -q[2], -q[3]], dtype=np.float64)


def quat_mul(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        dtype=np.float64,
    )


def rotate_world_to_body(quat_wxyz: np.ndarray, vector_world: np.ndarray) -> np.ndarray:
    quat = quat_wxyz / np.linalg.norm(quat_wxyz)
    vec_quat = np.array([0.0, *vector_world], dtype=np.float64)
    return quat_mul(quat_mul(quat_conj(quat), vec_quat), quat)[1:]


class CommandState:
    def __init__(
        self,
        default_command: np.ndarray,
        limits: dict[str, list[float]],
        linear_step: float,
        angular_step: float,
    ):
        self.command = default_command.astype(np.float32)
        self.limits = limits
        self.linear_step = linear_step
        self.angular_step = angular_step

    def clip(self) -> None:
        self.command[0] = np.clip(self.command[0], *self.limits["lin_vel_x"])
        self.command[1] = np.clip(self.command[1], *self.limits["lin_vel_y"])
        self.command[2] = np.clip(self.command[2], *self.limits["ang_vel_z"])

    def on_key(self, keycode: int) -> None:
        if keycode == 32:
            self.command[:] = 0.0
        else:
            try:
                key = chr(keycode).lower()
            except ValueError:
                return
            if key == "w":
                self.command[0] += self.linear_step
            elif key == "s":
                self.command[0] -= self.linear_step
            elif key == "a":
                self.command[1] += self.linear_step
            elif key == "d":
                self.command[1] -= self.linear_step
            elif key == "q":
                self.command[2] += self.angular_step
            elif key == "e":
                self.command[2] -= self.angular_step
            else:
                return
        self.clip()
        print(
            f"[command] vx={self.command[0]: .2f}, vy={self.command[1]: .2f}, wz={self.command[2]: .2f}"
        )


class OnnxPolicy:
    def __init__(self, policy_path: Path, obs_dim: int, action_dim: int):
        if not policy_path.exists():
            raise FileNotFoundError(f"Policy file not found: {policy_path}")
        providers = ["CPUExecutionProvider"]
        self.session = ort.InferenceSession(str(policy_path), providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self._check_io_shapes()

    def _check_io_shapes(self) -> None:
        input_shape = self.session.get_inputs()[0].shape
        output_shape = self.session.get_outputs()[0].shape
        if isinstance(input_shape[-1], int) and input_shape[-1] != self.obs_dim:
            raise ValueError(
                f"ONNX input dim {input_shape[-1]} does not match configured obs dim {self.obs_dim}."
            )
        if isinstance(output_shape[-1], int) and output_shape[-1] != self.action_dim:
            raise ValueError(
                f"ONNX output dim {output_shape[-1]} does not match configured action dim {self.action_dim}."
            )

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        obs_batch = obs.astype(np.float32, copy=False).reshape(1, -1)
        action = self.session.run([self.output_name], {self.input_name: obs_batch})[0]
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        if action.shape[0] != self.action_dim:
            raise ValueError(
                f"Policy returned {action.shape[0]} actions, expected {self.action_dim}."
            )
        if not np.all(np.isfinite(action)):
            raise FloatingPointError("Policy returned non-finite actions.")
        return action


class ObservationHistory:
    """Term-wise observation history matching Isaac Lab's flattened history layout."""

    def __init__(self, term_dims: list[int], history_length: int):
        self.term_dims = term_dims
        self.history_length = max(1, int(history_length))
        self._buffers: list[np.ndarray] | None = None

    def reset(self) -> None:
        self._buffers = None

    def append_and_flatten(self, terms: list[np.ndarray]) -> np.ndarray:
        if len(terms) != len(self.term_dims):
            raise ValueError(
                f"Expected {len(self.term_dims)} observation terms, got {len(terms)}."
            )

        normalized_terms = []
        for term, expected_dim in zip(terms, self.term_dims):
            term = np.asarray(term, dtype=np.float32).reshape(-1)
            if term.shape[0] != expected_dim:
                raise ValueError(
                    f"Observation term dim {term.shape[0]} does not match expected {expected_dim}."
                )
            normalized_terms.append(term)

        if self._buffers is None:
            self._buffers = [
                np.repeat(term[None, :], self.history_length, axis=0)
                for term in normalized_terms
            ]
        else:
            for buffer, term in zip(self._buffers, normalized_terms):
                buffer[:-1] = buffer[1:]
                buffer[-1] = term

        return np.concatenate([buffer.reshape(-1) for buffer in self._buffers]).astype(
            np.float32
        )


class MujocoSim2Sim:
    def __init__(
        self,
        cfg: dict[str, Any],
        urdf_path: Path,
        policy_path: Path,
        _height_scan_value: float | None,
    ):
        self.cfg = cfg
        self.urdf_path = urdf_path
        self.policy_cfg = cfg["policy"]
        self.sim_cfg = cfg["simulation"]
        self.control_cfg = cfg["control"]
        self.joint_names = list(cfg["joint_order"])
        self.default_joint_pos = np.array(
            [cfg["default_joint_pos"][name] for name in self.joint_names],
            dtype=np.float64,
        )
        self.last_action = np.zeros(len(self.joint_names), dtype=np.float32)
        self.target_joint_pos = self.default_joint_pos.copy()
        self.obs_history = ObservationHistory(
            term_dims=[
                3,
                3,
                3,
                len(self.joint_names),
                len(self.joint_names),
                len(self.joint_names),
            ],
            history_length=int(self.policy_cfg.get("observation_history_length", 1)),
        )

        if not urdf_path.exists():
            raise FileNotFoundError(f"URDF file not found: {urdf_path}")
        self.model = self._load_model()
        self.data = mujoco.MjData(self.model)
        self.model.opt.timestep = float(self.sim_cfg["physics_dt"])

        self.free_qpos_addr, self.free_dof_addr = self._resolve_free_joint()
        self.joint_ids, self.qpos_addr, self.dof_addr = self._resolve_joints()
        self._reset_state()
        self.policy = OnnxPolicy(
            policy_path,
            obs_dim=int(self.policy_cfg["observation_dim"]),
            action_dim=int(self.policy_cfg["action_dim"]),
        )
        self._last_trace: dict[str, np.ndarray] | None = None

    def _load_model(self):
        model = mujoco.MjModel.from_xml_path(str(self.urdf_path))
        if self._model_has_free_joint(model) or not bool(
            self.sim_cfg.get("add_free_joint_if_missing", True)
        ):
            return model
        return self._load_model_with_free_joint()

    @staticmethod
    def _model_has_free_joint(model) -> bool:
        return any(
            model.jnt_type[joint_id] == mujoco.mjtJoint.mjJNT_FREE
            for joint_id in range(model.njnt)
        )

    def _load_model_with_free_joint(self):
        tree = ET.parse(self.urdf_path)
        robot = tree.getroot()
        world = robot.find("./link[@name='world']")
        if world is None:
            world = ET.SubElement(robot, "link", {"name": "world"})
        if bool(self.sim_cfg.get("add_ground_if_missing", True)):
            self._ensure_ground_collision(world)
        existing = robot.find("./joint[@name='floating_base']")
        if existing is None:
            joint = ET.SubElement(
                robot, "joint", {"name": "floating_base", "type": "floating"}
            )
            ET.SubElement(joint, "parent", {"link": "world"})
            ET.SubElement(joint, "child", {"link": "base_link"})
            ET.SubElement(joint, "origin", {"xyz": "0 0 0", "rpy": "0 0 0"})
        with tempfile.NamedTemporaryFile(
            "wb", suffix=".urdf", delete=False
        ) as tmp_file:
            tmp_path = Path(tmp_file.name)
            tree.write(tmp_file, encoding="utf-8", xml_declaration=True)
        try:
            model = mujoco.MjModel.from_xml_path(str(tmp_path))
        finally:
            tmp_path.unlink(missing_ok=True)
        print(
            "[info] Added temporary floating_base joint for MuJoCo free-base simulation."
        )
        return model

    def _ensure_ground_collision(self, world: ET.Element) -> None:
        if world.find("./collision[@name='sim2sim_ground_collision']") is None:
            sx, sy, sz = self.sim_cfg.get("ground_size", [100.0, 100.0, 0.01])
            z = -0.5 * float(sz)
            collision = ET.SubElement(
                world, "collision", {"name": "sim2sim_ground_collision"}
            )
            ET.SubElement(collision, "origin", {"xyz": f"0 0 {z}", "rpy": "0 0 0"})
            geometry = ET.SubElement(collision, "geometry")
            ET.SubElement(geometry, "box", {"size": f"{sx} {sy} {sz}"})
        if world.find("./visual[@name='sim2sim_ground_visual']") is None:
            sx, sy, sz = self.sim_cfg.get("ground_size", [100.0, 100.0, 0.01])
            z = -0.5 * float(sz)
            visual = ET.SubElement(world, "visual", {"name": "sim2sim_ground_visual"})
            ET.SubElement(visual, "origin", {"xyz": f"0 0 {z}", "rpy": "0 0 0"})
            geometry = ET.SubElement(visual, "geometry")
            ET.SubElement(geometry, "box", {"size": f"{sx} {sy} {sz}"})

    def _resolve_free_joint(self) -> tuple[int | None, int | None]:
        for joint_id in range(self.model.njnt):
            if self.model.jnt_type[joint_id] == mujoco.mjtJoint.mjJNT_FREE:
                return int(self.model.jnt_qposadr[joint_id]), int(
                    self.model.jnt_dofadr[joint_id]
                )
        return None, None

    def _resolve_joints(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        joint_ids = []
        qpos_addr = []
        dof_addr = []
        for name in self.joint_names:
            joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if joint_id < 0:
                raise ValueError(
                    f"Joint '{name}' from config was not found in the MuJoCo model."
                )
            if self.model.jnt_type[joint_id] != mujoco.mjtJoint.mjJNT_HINGE:
                raise ValueError(f"Joint '{name}' is not a hinge joint in MuJoCo.")
            joint_ids.append(joint_id)
            qpos_addr.append(self.model.jnt_qposadr[joint_id])
            dof_addr.append(self.model.jnt_dofadr[joint_id])
        return np.array(joint_ids), np.array(qpos_addr), np.array(dof_addr)

    def _reset_state(self) -> None:
        mujoco.mj_resetData(self.model, self.data)
        if self.free_qpos_addr is not None:
            self.data.qpos[self.free_qpos_addr : self.free_qpos_addr + 3] = np.array(
                self.sim_cfg["initial_base_pos"], dtype=np.float64
            )
            self.data.qpos[self.free_qpos_addr + 3 : self.free_qpos_addr + 7] = (
                np.array(self.sim_cfg["initial_base_quat_wxyz"], dtype=np.float64)
            )
        else:
            print(
                "[warn] MuJoCo model does not expose a free base qpos; locomotion may be fixed-base."
            )
        self.data.qpos[self.qpos_addr] = self.default_joint_pos
        self.data.qvel[:] = 0.0
        self.last_action[:] = 0.0
        self.target_joint_pos = self.default_joint_pos.copy()
        self.obs_history.reset()
        self.data.qfrc_applied[:] = 0.0
        mujoco.mj_forward(self.model, self.data)

    def make_observation(self, command: np.ndarray) -> np.ndarray:
        if self.free_qpos_addr is not None and self.free_dof_addr is not None:
            base_quat = self.data.qpos[
                self.free_qpos_addr + 3 : self.free_qpos_addr + 7
            ].copy()
            base_ang_vel = rotate_world_to_body(
                base_quat,
                self.data.qvel[self.free_dof_addr + 3 : self.free_dof_addr + 6],
            )
        else:
            base_quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
            base_ang_vel = np.zeros(3, dtype=np.float64)
        projected_gravity = rotate_world_to_body(
            base_quat, np.array([0.0, 0.0, -1.0], dtype=np.float64)
        )
        joint_pos_rel = self.data.qpos[self.qpos_addr] - self.default_joint_pos
        joint_vel = self.data.qvel[self.dof_addr]
        obs = self.obs_history.append_and_flatten(
            [
                base_ang_vel,
                projected_gravity,
                command,
                joint_pos_rel,
                joint_vel,
                self.last_action,
            ]
        )
        expected_dim = int(self.policy_cfg["observation_dim"])
        if obs.shape[0] != expected_dim:
            raise ValueError(
                f"Constructed obs dim {obs.shape[0]} does not match expected {expected_dim}."
            )
        if not np.all(np.isfinite(obs)):
            raise FloatingPointError("Observation contains non-finite values.")
        return obs

    def step_policy(self, command: np.ndarray) -> dict[str, np.ndarray]:
        obs = self.make_observation(command)
        action = self.policy(obs)
        action_scale = float(self.policy_cfg["action_scale"])
        self.target_joint_pos = self.default_joint_pos + action_scale * action.astype(
            np.float64
        )
        torque = self.apply_pd_control()
        self.last_action = action
        self._last_trace = {
            "time": np.array(self.data.time, dtype=np.float64),
            "command": command.copy(),
            "obs": obs.copy(),
            "raw_action": action.copy(),
            "target_joint_pos": self.target_joint_pos.astype(np.float32),
            "applied_torque": torque.astype(np.float32),
            "base_pos": self._base_pos().astype(np.float32),
            "base_quat": self._base_quat().astype(np.float32),
            "base_lin_vel": self._base_lin_vel_b().astype(np.float32),
            "base_ang_vel": self._base_ang_vel_b().astype(np.float32),
            "projected_gravity": self._projected_gravity().astype(np.float32),
            "joint_pos": self.data.qpos[self.qpos_addr].astype(np.float32),
            "joint_pos_rel": (
                self.data.qpos[self.qpos_addr] - self.default_joint_pos
            ).astype(np.float32),
            "joint_vel": self.data.qvel[self.dof_addr].astype(np.float32),
        }
        return self._last_trace

    def apply_pd_control(self) -> np.ndarray:
        kp = float(self.control_cfg["kp"])
        kd = float(self.control_cfg["kd"])
        torque_limit = float(self.control_cfg["torque_limit"])
        q = self.data.qpos[self.qpos_addr]
        dq = self.data.qvel[self.dof_addr]
        torque = kp * (self.target_joint_pos - q) - kd * dq
        torque = np.clip(torque, -torque_limit, torque_limit)
        self.data.qfrc_applied[:] = 0.0
        self.data.qfrc_applied[self.dof_addr] = torque
        return torque

    def run(
        self,
        command_state: CommandState,
        duration: float | None,
        use_viewer: bool,
        log_path: Path | None = None,
    ) -> None:
        control_dt = float(self.sim_cfg["control_dt"])
        physics_dt = float(self.sim_cfg["physics_dt"])
        substeps = max(1, int(round(control_dt / physics_dt)))
        start_sim_time = float(self.data.time)
        next_control = 0
        trace = TraceBuffer(
            metadata={
                "source": "mujoco",
                "urdf_path": str(self.urdf_path),
                "joint_names": self.joint_names,
                "default_joint_pos": self.default_joint_pos.astype(np.float32),
                "physics_dt": physics_dt,
                "control_dt": control_dt,
                "observation_history_length": self.obs_history.history_length,
            }
        )

        if use_viewer:
            from mujoco import viewer as mujoco_viewer

            with mujoco_viewer.launch_passive(
                self.model, self.data, key_callback=command_state.on_key
            ) as viewer:
                while viewer.is_running() and self._keep_running(
                    start_sim_time, duration
                ):
                    step_start = time.monotonic()
                    if next_control <= 0:
                        trace.append(self.step_policy(command_state.command))
                        next_control = substeps
                    else:
                        self.apply_pd_control()
                    mujoco.mj_step(self.model, self.data)
                    next_control -= 1
                    self._check_finite()
                    viewer.sync()
                    self._sleep_realtime(step_start, physics_dt)
        else:
            while self._keep_running(start_sim_time, duration):
                if next_control <= 0:
                    trace.append(self.step_policy(command_state.command))
                    next_control = substeps
                else:
                    self.apply_pd_control()
                mujoco.mj_step(self.model, self.data)
                next_control -= 1
                self._check_finite()
        if log_path is not None:
            trace.save(resolve_path(log_path))

    def _keep_running(self, start_sim_time: float, duration: float | None) -> bool:
        return duration is None or (float(self.data.time) - start_sim_time) < duration

    @staticmethod
    def _sleep_realtime(step_start: float, physics_dt: float) -> None:
        sleep_time = physics_dt - (time.monotonic() - step_start)
        if sleep_time > 0:
            time.sleep(sleep_time)

    def _check_finite(self) -> None:
        if not np.all(np.isfinite(self.data.qpos)) or not np.all(
            np.isfinite(self.data.qvel)
        ):
            raise FloatingPointError("MuJoCo state contains non-finite values.")
        root_z = (
            self.data.qpos[self.free_qpos_addr + 2]
            if self.free_qpos_addr is not None
            else math.inf
        )
        if root_z < -1.0:
            raise FloatingPointError("Robot base fell below z=-1.0; stopping sim.")

    def _base_pos(self) -> np.ndarray:
        if self.free_qpos_addr is None:
            return np.zeros(3, dtype=np.float64)
        return self.data.qpos[self.free_qpos_addr : self.free_qpos_addr + 3].copy()

    def _base_quat(self) -> np.ndarray:
        if self.free_qpos_addr is None:
            return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        return self.data.qpos[self.free_qpos_addr + 3 : self.free_qpos_addr + 7].copy()

    def _base_lin_vel_b(self) -> np.ndarray:
        if self.free_qpos_addr is None or self.free_dof_addr is None:
            return np.zeros(3, dtype=np.float64)
        return rotate_world_to_body(
            self._base_quat(),
            self.data.qvel[self.free_dof_addr : self.free_dof_addr + 3],
        )

    def _base_ang_vel_b(self) -> np.ndarray:
        if self.free_qpos_addr is None or self.free_dof_addr is None:
            return np.zeros(3, dtype=np.float64)
        return rotate_world_to_body(
            self._base_quat(),
            self.data.qvel[self.free_dof_addr + 3 : self.free_dof_addr + 6],
        )

    def _projected_gravity(self) -> np.ndarray:
        return rotate_world_to_body(
            self._base_quat(), np.array([0.0, 0.0, -1.0], dtype=np.float64)
        )


class TraceBuffer:
    def __init__(self, metadata: dict[str, Any]):
        self.metadata = metadata
        self._data: dict[str, list[np.ndarray]] = {}

    def append(self, values: dict[str, np.ndarray]) -> None:
        for key, value in values.items():
            self._data.setdefault(key, []).append(np.asarray(value).copy())

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        arrays = {
            key: np.stack(values, axis=0)
            for key, values in self._data.items()
            if values
        }
        for key, value in self.metadata.items():
            arrays[f"meta_{key}"] = np.asarray(value)
        np.savez_compressed(path, **arrays)
        print(f"[info] Wrote trace: {path}")


def main() -> None:
    args = parse_args()
    config_path = resolve_path(args.config)
    cfg = load_config(config_path)
    urdf_path = resolve_path(args.urdf if args.urdf is not None else cfg["urdf_path"])
    policy_path = resolve_path(
        args.policy if args.policy is not None else cfg["policy_path"]
    )
    command_step = cfg["policy"]["command_step"]
    command_state = CommandState(
        default_command=np.array(cfg["policy"]["default_command"], dtype=np.float32),
        limits=cfg["policy"]["command_limits"],
        linear_step=float(command_step["linear"]),
        angular_step=float(command_step["angular"]),
    )

    runner = MujocoSim2Sim(
        cfg,
        urdf_path=urdf_path,
        policy_path=policy_path,
        _height_scan_value=args.height_scan_value,
    )
    print(f"[info] URDF: {urdf_path}")
    print(f"[info] Policy: {policy_path}")
    print(
        f"[info] Initial command: vx={command_state.command[0]:.2f}, vy={command_state.command[1]:.2f}, wz={command_state.command[2]:.2f}"
    )
    runner.run(
        command_state,
        duration=args.duration,
        use_viewer=not args.no_viewer,
        log_path=args.log,
    )


if __name__ == "__main__":
    main()
