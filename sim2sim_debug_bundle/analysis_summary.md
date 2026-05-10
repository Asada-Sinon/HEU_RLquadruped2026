# Sim2Sim Trace Analysis

Data captured on 2026-05-09:

- IsaacLab trace: `sim2sim_debug_bundle/isaaclab/trace.npz`
- MuJoCo trace: `sim2sim_debug_bundle/mujoco/trace.npz`
- User video copy: `sim2sim_debug_bundle/mujoco/第一次mujoco乱飞.mp4`

## Main Findings

1. The original MuJoCo joint order was wrong.
   - IsaacLab actual order is:
     `LF_HAA, LH_HAA, RF_HAA, RH_HAA, LF_HFE, LH_HFE, RF_HFE, RH_HFE, LF_KFE, LH_KFE, RF_KFE, RH_KFE`
   - The first MuJoCo config used leg-by-leg order.
   - This means policy action indices were being sent to the wrong joints. I updated `sim2sim_mujoco/configs/my_quad.yaml` to match IsaacLab.

2. MuJoCo needed a floating base and ground.
   - Direct MuJoCo URDF loading had no free joint, so the robot could be fixed-base.
   - After adding a free joint, it fell because there was no ground in the MuJoCo model.
   - I updated `run_mujoco.py` to temporarily inject both a `floating_base` joint and a static ground collision during loading.

3. Height scan was not equivalent.
   - IsaacLab `height_scan` uses `sensor_z - ray_hit_z - 0.5`, so on flat ground it is roughly `base_z - 0.5`.
   - MuJoCo originally filled height scan with `0.0`.
   - I updated MuJoCo height scan to use `flat_from_base_height`.

4. IsaacLab play is still not a clean nominal comparison.
   - The trace shows observation corruption is enabled in the play config.
   - Reset randomization changes the initial joint positions.
   - Startup domain randomization changes friction, base mass, and COM.
   - These are useful for robustness but make IsaacLab-vs-MuJoCo debugging harder.

5. The MuJoCo PD controller was being updated at the wrong rate.
   - IsaacLab updates the policy target every `decimation * physics_dt = 0.02 s`, but the actuator torque is recomputed every physics step.
   - The previous MuJoCo loop computed PD torque only when the policy ran and held that torque for the next four `0.005 s` physics steps.
   - This made the robot behave too soft/late after contact, producing a crooked stance and almost no visible command response.
   - `sim2sim_mujoco/run_mujoco.py` now keeps the latest target joint position, recomputes PD torque before every MuJoCo physics step, and only refreshes the ONNX policy action at the control rate.

## Current Trace Numbers

After fixing joint order and MuJoCo height scan, but before fixing the PD update rate:

- Joint order: matched.
- Command: matched exactly.
- First-frame mean absolute action difference: about `0.163`.
- Mean action difference over 100 policy steps: about `0.708`.
- First-frame joint position difference: about `0.147 rad`, mainly from IsaacLab reset randomization.
- Height scan mean difference over the trace: about `0.123`, partly because MuJoCo base height drops more.
- MuJoCo base z over 2 seconds: `0.37 -> 0.271 m`.
- IsaacLab base z over 2 seconds: `0.375 -> 0.415 m`.

After fixing the MuJoCo PD update rate, a new trace was saved at
`sim2sim_debug_bundle/mujoco/trace_pd_every_step.npz`.

With the default command `vx=0.2, vy=0.0, wz=0.0`:

- MuJoCo base position over 2 seconds: approximately `[0.0, 0.0, 0.37] -> [0.335, 0.009, 0.342]`.
- Final body-frame linear velocity: approximately `[0.179, -0.0001, -0.023] m/s`.
- Final projected gravity: approximately `[-0.006, 0.001, -1.0]`, so the body is no longer visibly tilted.
- A zero-command check stayed near the start pose: final base position about `[0.009, 0.004, 0.347]`.
- Larger/other commands also responded: `vx=0.6` reached about `0.54 m/s`, `vy=0.15` reached about `0.14 m/s`, and `wz=0.5` reached about `0.42 rad/s`.

I also captured a nominal IsaacLab trace at `sim2sim_debug_bundle/isaaclab_nominal/trace.npz` with observation noise and most randomization disabled. In that run:

- First-frame observation difference: about `6e-7`.
- First-frame raw action difference: about `1e-6`.
- First-frame target joint position difference: about `3e-7`.

So after fixing joint order and height-scan construction, the policy input/output path is effectively aligned at the first policy step. The remaining divergence is dominated by simulator dynamics, contact, actuator, and post-step state evolution.

## Suggested Next Fixes

1. Use the updated MuJoCo joint order, free-base/ground loader, height scan, and every-physics-step PD torque update as the new baseline.
2. Re-record the MuJoCo viewer video from this baseline and compare it to IsaacLab play.
3. For debugging, create a nominal IsaacLab play mode:
   - disable observation corruption,
   - disable startup domain randomization,
   - disable reset randomization,
   - keep command fixed.
4. Then compare first-frame `obs` and `raw_action`. They should be nearly identical except for simulator-only height scan details.
5. Once nominal traces match, re-enable one difference at a time: height scan, reset randomization, then domain randomization.

## Practical Recommendation

Keep the MuJoCo-side fixes. For IsaacLab visual `play`, add a dedicated play/eval config or use the new trace options as a template:

- disable observation corruption during play/eval,
- disable startup randomization when doing simulator-to-simulator diagnosis,
- disable command debug visualization if offline,
- avoid the default ground-plane USD path if it fails in the local Isaac Sim install.

The policy itself is probably not the first problem anymore: nominal first-step `obs -> action -> target` now matches. The next debugging target is the physics layer, especially actuator implementation and contact/ground handling.
