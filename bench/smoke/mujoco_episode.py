"""MuJoCo: run one episode of a 7-DoF arm under position control and time each physics step."""

from __future__ import annotations

import math
from pathlib import Path

from ..harness import Benchmark

# Seven hinge joints alternating z/y axes, roughly the proportions of a Franka Panda.
_AXES = ("0 0 1", "0 1 0", "0 0 1", "0 1 0", "0 0 1", "0 1 0", "0 0 1")
_LENGTHS = (0.333, 0.316, 0.0, 0.384, 0.0, 0.088, 0.107)


def arm_xml(timestep: float = 0.002) -> str:
    body = ""
    close = ""
    for i, (axis, length) in enumerate(zip(_AXES, _LENGTHS)):
        seg = max(length, 0.08)
        body += (
            f'<body name="link{i + 1}" pos="0 0 {0.0 if i == 0 else _LENGTHS[i - 1] or 0.08}">'
            f'<joint name="j{i + 1}" type="hinge" axis="{axis}" range="-2.8 2.8" damping="1"/>'
            f'<geom type="capsule" fromto="0 0 0 0 0 {seg}" size="0.05" mass="2"/>'
        )
        close += "</body>"
    actuators = "".join(
        f'<position name="a{i + 1}" joint="j{i + 1}" kp="200" ctrlrange="-2.8 2.8"/>' for i in range(7)
    )
    return f"""
<mujoco model="sixgig_arm7">
  <option timestep="{timestep}" integrator="implicitfast"/>
  <worldbody>
    <geom name="floor" type="plane" size="2 2 0.1"/>
    <body name="base" pos="0 0 0">{body}{close}</body>
    <body name="box" pos="0.5 0 0.05">
      <freejoint/>
      <geom type="box" size="0.04 0.04 0.04" mass="0.2"/>
    </body>
  </worldbody>
  <actuator>{actuators}</actuator>
</mujoco>
"""


def run(quick: bool = False, suite: str = "phase0") -> Path:
    import mujoco

    timestep = 0.002
    steps = 1000 if quick else 10000
    warmup = 100
    params = {"model": "sixgig_arm7", "dof": 7, "timestep_s": timestep, "steps": steps, "integrator": "implicitfast"}
    # Physics is CPU-only here, so GPU telemetry is not needed.
    with Benchmark("mujoco_arm7_episode", suite=suite, params=params, monitor_gpu=False) as b:
        b.params["mujoco"] = mujoco.__version__
        model = mujoco.MjModel.from_xml_string(arm_xml(timestep))
        data = mujoco.MjData(model)
        mujoco.mj_resetData(model, data)
        state = {"k": 0}

        def step():
            t = state["k"] * timestep
            for j in range(7):
                data.ctrl[j] = 0.8 * math.sin(2 * math.pi * 0.5 * t + j)
            mujoco.mj_step(model, data)
            state["k"] += 1

        samples = b.measure(step, warmup=warmup, iters=steps)
        if not all(math.isfinite(v) for v in data.qpos):
            raise RuntimeError("simulation diverged: non-finite qpos")
        p50_s = sorted(samples)[len(samples) // 2] / 1e3
        b.metrics["steps_per_s_p50"] = 1.0 / p50_s
        b.metrics["realtime_factor_p50"] = timestep / p50_s
        b.metrics["sim_time_s"] = state["k"] * timestep
        b.metrics["nq"] = model.nq
        b.metrics["nv"] = model.nv
    return b.output_path
