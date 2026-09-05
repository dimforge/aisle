# ADR-55: a second physics engine (Nexus) behind the scene contract

Status: ACCEPTED (development); a spec-change PR is still owed (see below).

## Context

SPEC 020 names Genesis World as the simulator and SPEC 030 calls the bridge
`dora-genesis`, but the contracts themselves are engine neutral at the object
level: the bridge talks to `robot`, entity, link and camera objects through a
small duck-typed surface (`get_qpos`, `control_dofs_position`, `set_pos`,
`render(rgb, depth, segmentation)`, ...), and everything that decides a
scene (layout, seeded placements, DR draws, label textures, camera mounts)
is a pure function in the frozen `aisle.scenes` modules.

We want to run the same experiments on Nexus (`dimforge/nexus`, a GPU
rigid-body engine with Python bindings) without disturbing the measured
Genesis record.

## Decision

1. **The engine is an explicit, attested choice.** `AISLE_SIM_ENGINE`
   (`genesis` | `nexus`, default `genesis`) is declared on the bridge node
   like the perception rung, injected by `harness rollout --sim-engine`,
   and attested in `bridge_info` (`sim_engine`, with `genesis_version`
   keeping its BRG-6 name and carrying whichever engine's version ran) and
   in the run manifest (`sim_engine`). Unknown names are refused, never
   defaulted (the TC-9 rule).
2. **The frozen set is untouched (CON-7).** `src/aisle/scenes/*` keeps the
   Genesis builders byte-for-byte. The Nexus realization lives in
   `src/aisle/sim/nexus_backend.py`, outside the fence, and REUSES the
   frozen pure functions (`resolve_layout`, `sample_placements`,
   `apply_occlusion`, `label_texture_image`, `wrist_mount_transform`,
   `_assert_reachable`, the `SceneHandle`/`StoreHandle` dataclasses). Only
   the object construction is re-implemented. A sim test pins the Nexus
   placements to the frozen sampler's output.
3. **Same object surface, same conventions.** Nexus wrappers return numpy
   arrays (Genesis returned tensors; `to_numpy` passes both), quaternions in
   Genesis's (w, x, y, z) order, the same single-env/batched shape rules,
   `segmentation_idx_dict` in the `link`-level shape TC-9 documents, and a
   camera `transform` in the OpenGL look-at convention VER-8 pins, so the
   verifier's calibration module needs no change.
4. **Rendering lives in the Nexus viewer, not the physics engine.** Sensor
   cameras (offscreen surfaces with RGB, metric depth and per-body
   segmentation passes, optionally attached to a link) were added to
   `nexus_viewer3d`; the physics core only gained body/joint state read and
   write access.
5. **Nexus-only constants** (Genesis-equivalent default PD gains for URDF
   joints, the ground slab replacing the infinite plane, camera clip planes)
   live in `src/aisle/sim/nexus_physics.toml`, outside the frozen set.

## Consequences

- Results across engines are NOT comparable: contact models, solver and
  renderer differ. A Nexus run is a different environment; the frozen-set
  hash does not encode the engine, so the manifest's `sim_engine` field is
  the discriminator. Before any Nexus result enters the measured record,
  the frozen baseline must be re-established under human review (CON-7) and
  SPEC 020 / SPEC 030 wording generalized by a `spec-change` PR (CON-14).
- The Genesis-fit constants in `physics.toml` (gripper gains, the SO-101
  kinematic carry latch, convex decomposition threshold) were not re-tuned;
  the Nexus path inherits them.
- CON-5 on Nexus: the GPU broad phase and constraint coloring use atomics,
  so bitwise run-to-run reproducibility is not established. `build_scene`
  determinism (same seed, same initial state) holds; stepping determinism
  must be measured before Nexus evidence is trusted.
- Fixed upstream while doing this, both consumed by Nexus through
  `[patch.crates-io]` path overrides until releases carry them:
  rapier3d-urdf composed URDF `rpy` as intrinsic XYZ Euler angles instead of
  fixed-axis roll-pitch-yaw (`Rz * Ry * Rx`), misplacing every SO-101 link
  below the shoulder (rapier branch `fix-urdf-rpy`); and kiss3d replaced its
  global mesh/texture/material managers whenever a second window or
  offscreen surface was created, orphaning the material of objects built
  before the sensor cameras existed, whose per-object uniform buffer then
  grew until wgpu rejected the offsets (kiss3d branch
  `fix-shared-window-managers`).
- Solver settings are the engine's, not the scene's, and had to be tuned
  (`src/aisle/sim/nexus_physics.toml [sim]`): boxes at rest on a board creep
  on Nexus at a rate set only by the substep count (1 substep: 25 mm in
  0.3 s, read by the oracle as a collision; 8: 1.2 mm/s; 16: 0.33 mm/s), so
  Nexus runs 16 substeps against Genesis's one. With Nexus's default 30 Hz
  soft contacts, 5 mm penetration allowance and one PGS iteration the Franka
  pinch let the box slip on lift. Replaying the recorded T0 commands
  (`tools/nexus_grasp_replay.py`) the box follows the hand as on Genesis in
  2 of 3 replays with 240 Hz contacts, 0.5 mm allowance and eight PGS
  iterations (now the defaults; 0 of 3 with four iterations), at roughly 10 ms
  of GPU time per 10 ms tick on an M-series laptop. The pinch is therefore
  still marginal and not run-to-run deterministic: the Genesis T0 expert
  passes seed 0 (verified on this bridge code), the Nexus run executes every
  stage but has not yet held the box through the lift. Both the creep and
  the pinch robustness are Nexus solver work, tracked upstream.
- Known gaps on Nexus: `get_dofs_velocity` on robots reports zeros
  (generalized velocities are not read back), a fixed-root robot's base is
  re-based through the body buffer (SPEC 210 mobile), and URDF inertial
  `rpy` orientation is ignored by rapier3d-urdf (translation applied).

## Alternatives rejected

- Refactoring `pharmacy.py`/`store.py` into an engine-neutral builder: the
  clean design, but it edits the frozen set and re-freezes the baseline for
  a change with no measured benefit yet.
- Selecting the engine from ambient environment only: violates the
  graph-attests-everything rule that the rung and the backend already
  follow.
