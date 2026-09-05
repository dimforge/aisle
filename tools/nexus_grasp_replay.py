#!/usr/bin/env python3
"""Replay a recorded run's joint and gripper commands into a fresh Nexus scene.

The recorded `budget-guard__joint_cmd_safe` / `gripper_cmd_safe` traces of a
rollout are fed tick by tick to the Nexus pharmacy scene built for the same
seed, so the physics can be re-run offline under different solver settings
(`--substeps`, `--contact-frequency`, `--pgs`) without dora. Reports how far
the target box rose with the hand: the pinch-grasp fidelity check behind the
`[sim]` defaults of src/aisle/sim/nexus_physics.toml (ADR-55). CON-8: JSON on
stdout, logs on stderr, exit 0 iff the replay ran (not iff the grasp held).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def _load(path: Path) -> dict:
    import pyarrow as pa

    with pa.memory_map(str(path)) as src:
        try:
            table = pa.ipc.open_stream(src).read_all()
        except pa.ArrowInvalid:
            table = pa.ipc.open_file(src).read_all()
    return table.to_pydict()


def replay(
    run_dir: Path,
    seed: int,
    embodiment: str,
    target: str,
    steps: int,
    substeps: int | None,
    contact_frequency: float | None,
    pgs: int | None,
) -> dict:
    import aisle.sim.nexus_backend as nb
    from aisle.scenes.pharmacy import load_physics, to_numpy

    traces = run_dir / "traces"
    joints = _load(traces / "budget-guard__joint_cmd_safe.arrow")
    grips = _load(traces / "budget-guard__gripper_cmd_safe.arrow")
    jt = np.asarray(joints["sim_time_ns"], dtype=np.int64)
    gt = np.asarray(grips["sim_time_ns"], dtype=np.int64)
    profile = load_physics()["embodiment"][embodiment]
    open_m, close_m = float(profile["gripper_open_m"]), float(profile["gripper_close_m"])

    base = nb.load_nexus_physics()
    cfg = json.loads(json.dumps(base))
    if substeps is not None:
        cfg["sim"]["substeps"] = substeps
    if contact_frequency is not None:
        cfg["sim"]["contact_natural_frequency"] = contact_frequency
        cfg["sim"]["static_contact_natural_frequency"] = 2.0 * contact_frequency
    if pgs is not None:
        cfg["sim"]["internal_pgs_iterations"] = pgs
    nb.load_nexus_physics = lambda: cfg

    handle = nb.build_scene(seed=seed, embodiment=embodiment, n_envs=1, headless=True)
    robot, box = handle.robot, handle.boxes[target]
    start = to_numpy(box.get_pos()).reshape(-1).copy()
    samples = []
    for step in range(steps):
        t_ns = step * 10_000_000
        if t_ns >= jt[0]:
            cmd = np.asarray(joints["data"][int(np.searchsorted(jt, t_ns, side="right")) - 1])
            cmd = cmd.astype(np.float32).copy()
            if t_ns >= gt[0]:
                grip = float(
                    np.asarray(grips["data"][int(np.searchsorted(gt, t_ns, side="right")) - 1])[0]
                )
                cmd[len(cmd) - int(profile["gripper_dofs"]) :] = open_m + (
                    close_m - open_m
                ) * np.clip(grip, 0.0, 1.0)
            robot.control_dofs_position(cmd)
        handle.scene.step()
        if step % 50 == 49:
            pos = to_numpy(box.get_pos()).reshape(-1)
            samples.append({"t_s": (step + 1) / 100, "box_rise_m": float(pos[2] - start[2])})
    rise = max(s["box_rise_m"] for s in samples) if samples else 0.0
    return {
        "run": str(run_dir),
        "target": target,
        "solver": handle.scene.state.rbd_solver_params(),
        "samples": samples,
        "max_box_rise_m": rise,
        "final_box_rise_m": samples[-1]["box_rise_m"] if samples else 0.0,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True, help="runs/<run-id> directory")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--embodiment", default="franka")
    parser.add_argument("--target", default="amoxicillin")
    parser.add_argument("--steps", type=int, default=800, help="10 ms ticks to replay")
    parser.add_argument("--substeps", type=int, default=None)
    parser.add_argument("--contact-frequency", type=float, default=None)
    parser.add_argument("--pgs", type=int, default=None)
    args = parser.parse_args(argv)
    try:
        report = replay(
            args.run.resolve(),
            args.seed,
            args.embodiment,
            args.target,
            args.steps,
            args.substeps,
            args.contact_frequency,
            args.pgs,
        )
    except (FileNotFoundError, KeyError, ImportError) as exc:
        print(json.dumps({"ok": False, "error": str(exc)}))
        print(str(exc), file=sys.stderr)
        return 1
    print(json.dumps({"ok": True, **report}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
