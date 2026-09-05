"""Sim acceptance tests for the Nexus engine backend (ADR-55): the frozen
pharmacy scene (SPEC 020 SCN-1, SCN-3..5, SCN-7) and the bridge's object
surface (SPEC 030), realized on Nexus.

Marker `sim`: imports nexus3d (and genesis for the Franka asset), runs
headless. Skipped when the nexus3d wheel is not installed.
"""

import importlib.util

import numpy as np
import pytest

from aisle.scenes.pharmacy import MED_NAMES, load_physics, oracle_state, to_numpy

pytestmark = [
    pytest.mark.sim,
    pytest.mark.skipif(
        importlib.util.find_spec("nexus3d") is None or importlib.util.find_spec("genesis") is None,
        reason="nexus3d wheel or the sim extra not installed",
    ),
]


# Function-scoped on purpose: the shared Nexus viewer renders the LATEST
# scene only (a later build supersedes the previous one's cameras), so every
# test gets a fresh scene (about a second each) instead of a module fixture.
@pytest.fixture
def handle():
    from aisle.sim import build_scene

    return build_scene("nexus", seed=7, embodiment="franka", n_envs=1, headless=True)


@pytest.fixture
def so101_handle():
    from aisle.sim import build_scene

    return build_scene("nexus", seed=3, embodiment="so101", n_envs=1, headless=True)


def test_build_determinism(handle):
    """SCN-1, SCN-7 on Nexus: the initial oracle_state is a pure function of
    the seed (bitwise) and differs across seeds."""
    from aisle.sim import build_scene

    first = oracle_state(handle)
    again = oracle_state(build_scene("nexus", seed=7, embodiment="franka", headless=True))
    other = oracle_state(build_scene("nexus", seed=11, embodiment="franka", headless=True))
    assert first.dtype == np.float32
    assert first.shape == (len(MED_NAMES) * 7,)
    assert np.array_equal(first, again)
    assert not np.array_equal(first, other)
    # a superseded scene keeps its physics readbacks but refuses to render
    assert np.array_equal(oracle_state(handle), first)
    with pytest.raises(RuntimeError, match="superseded"):
        handle.cams["overhead"].render()


def test_placements_match_frozen_sampler(handle):
    """ADR-55: the Nexus builder places the boxes exactly where the frozen
    sampler says (same pure function, same seed), before any step."""
    from aisle.scenes.pharmacy import resolve_layout, sample_placements

    placements = {
        p.name: p for p in sample_placements(7, MED_NAMES, resolve_layout(load_physics(), "franka"))
    }
    for name, entity in handle.boxes.items():
        pos = to_numpy(entity.get_pos()).reshape(-1)[:3]
        expected = placements[name]
        assert np.allclose(pos, [expected.x, expected.y, expected.z], atol=1e-5), name


def test_reachability_and_home(handle):
    """SCN-3/SCN-4 (Nexus IK) and SCN-1: every placement admits an IK
    solution and the robot rests at its home pose."""
    assert handle.reachability_errors == []
    home = np.asarray(load_physics()["embodiment"]["franka"]["home_qpos"], dtype=np.float32)
    actual = to_numpy(handle.robot.get_qpos()).reshape(-1)[: home.shape[0]]
    assert np.allclose(actual, home, atol=1e-4)


def test_cameras_render_all_passes(handle):
    """SCN-5 / TC-9: overhead 640x480 fov 55, wrist 320x240 fov 70 attached
    to the EE link; one pass yields rgb (uint8), metric depth (float32) and
    a segmentation map whose ids are the scene's own map (background -1)."""
    overhead, wrist = handle.cams["overhead"], handle.cams["wrist"]
    assert tuple(overhead.res) == (640, 480) and overhead.fov == 55
    assert tuple(wrist.res) == (320, 240) and wrist.fov == 70
    assert wrist._attached_link is not None and wrist._attached_link.name == "hand"
    rgb, depth, seg, _ = overhead.render(rgb=True, depth=True, segmentation=True)
    assert rgb.shape == (480, 640, 3) and rgb.dtype == np.uint8
    assert depth.shape == (480, 640) and depth.dtype == np.float32
    assert seg.shape == (480, 640) and seg.dtype == np.int64
    hit = depth[depth > 0]
    assert hit.size > 0 and 0.3 < hit.min() < hit.max() < 2.0  # meters, camera at z=1.2
    ids = {int(i) for i in np.unique(seg)} - {-1}
    idx_dict = handle.scene.segmentation_idx_dict
    assert ids and ids <= set(idx_dict)
    # every box is visible from the overhead camera under its own id
    entity_idx = {name: entity.idx for name, entity in handle.boxes.items()}
    box_ids = {sid for sid, ref in idx_dict.items() if ref != -1 and ref[0] in entity_idx.values()}
    assert box_ids <= ids
    wrist_rgb = wrist.render()[0]
    assert wrist_rgb.shape == (240, 320, 3)


def test_realized_calibration_matches_nominal(handle):
    """BRG-8 / VER-8: the realized overhead transform is the Genesis look-at
    convention, so the published v1 block passes stage 0."""
    from aisle.nodes.dora_genesis import realized_calibration
    from aisle.verifier.calibration import build_calibration_v1

    physics = load_physics()
    realized = realized_calibration(handle, physics, is_store=False)
    from aisle.scenes.pharmacy import wrist_mount_transform

    wrist = wrist_mount_transform(physics["cameras"], physics["embodiment"]["franka"])
    nominal = build_calibration_v1(
        overhead_pos=physics["cameras"]["overhead_pos"],
        overhead_lookat=physics["cameras"]["overhead_lookat"],
        overhead_resolution=(640, 480),
        overhead_fov_deg=55,
        wrist_offset_m=wrist[:3, 3].tolist(),
        wrist_resolution=(320, 240),
        wrist_fov_deg=70,
        wrist_mount_rotation_gl=wrist[:3, :3],
    )
    assert realized["overhead"]["intrinsics"] == nominal["overhead"]["intrinsics"]
    assert np.allclose(
        realized["overhead"]["cam_to_base"]["pos"],
        nominal["overhead"]["cam_to_base"]["pos"],
        atol=1e-5,
    )
    assert np.allclose(
        realized["overhead"]["cam_to_base"]["quat_xyzw"],
        nominal["overhead"]["cam_to_base"]["quat_xyzw"],
        atol=1e-4,
    )


def test_step_teleport_and_control(handle):
    """BRG-2/BRG-4: stepping advances physics, a teleport reset lands a box
    exactly where asked with zero velocity, and position control holds the
    home pose under gravity."""
    robot = handle.robot
    home = to_numpy(robot.get_qpos()).reshape(-1).copy()
    robot.control_dofs_position(home)
    box = next(iter(handle.boxes.values()))
    robot_before = to_numpy(robot.get_link("hand").get_pos()).reshape(-1)
    for _ in range(50):
        handle.scene.step()
    after = to_numpy(robot.get_qpos()).reshape(-1)
    assert np.allclose(after[:7], home[:7], atol=0.02)
    assert np.allclose(
        to_numpy(robot.get_link("hand").get_pos()).reshape(-1), robot_before, atol=0.02
    )
    box.set_pos(np.array([0.3, -0.45, 0.3], dtype=np.float32))
    box.set_quat(np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32))
    box.zero_all_dofs_velocity()
    assert np.allclose(to_numpy(box.get_pos()).reshape(-1), [0.3, -0.45, 0.3], atol=1e-5)
    handle.scene.step()
    pos = to_numpy(box.get_pos()).reshape(-1)
    assert pos[2] < 0.3 and pos[2] > 0.29  # falling under gravity, one step later


def test_resting_boxes_hold_still(handle):
    """SCN-3 on Nexus: boxes at rest on the boards must not creep past the
    oracle's knock threshold over an episode. The substep count in
    nexus_physics.toml is chosen for this; one second of settling must stay
    well under a millimeter per box."""
    robot = handle.robot
    robot.control_dofs_position(to_numpy(robot.get_qpos()).reshape(-1))
    start = oracle_state(handle).reshape(-1, 7)[:, :3]
    for _ in range(100):
        handle.scene.step()
    drift = np.linalg.norm(oracle_state(handle).reshape(-1, 7)[:, :3] - start, axis=1)
    assert drift.max() < 1.0e-3, drift


def test_so101_urdf_matches_frozen_chain(so101_handle):
    """ADR-55: the imported SO-101 kinematics agree with the frozen URDF
    chain (`aisle.kinematics`) at several configurations, so grasp planning
    and the guard see the same arm the physics does."""
    from aisle.kinematics import so101_chain

    chain = so101_chain()
    robot = so101_handle.robot
    rng = np.random.default_rng(1)
    for _ in range(3):
        q = rng.uniform(-1.0, 1.0, size=5)
        expected, _ = chain.forward(q)
        pos, _ = robot.scene.state.robot_forward_kinematics(
            robot.robots[0], [float(v) for v in q] + [0.0], "gripper_frame_link"
        )
        assert np.allclose(pos, expected, atol=2e-4), (q, pos, expected)


def test_so101_home_and_gripper_gains(so101_handle):
    """SCN-1 / TC-5: the SO-101 rests at its profile home in wire order and
    carries the profile's gripper gains; batched envs share the layout."""
    from aisle.embodiment import profile_dof_indices

    profile = load_physics()["embodiment"]["so101"]
    robot = so101_handle.robot
    wire = profile_dof_indices(robot, profile)
    qpos = to_numpy(robot.get_qpos()).reshape(-1)
    assert np.allclose(qpos[list(wire)], profile["home_qpos"], atol=1e-4)
    gripper = robot.robots[0]
    assert np.isclose(gripper.kp[wire[-1]], profile["gripper_kp"][0])
    assert np.isclose(gripper.kv[wire[-1]], profile["gripper_kv"][0])
    assert so101_handle.reachability_errors == []


def test_batched_build_oracle_covers_all_envs():
    """SCN-1 on Nexus: a batched build reports every env and identical
    initial placements."""
    from aisle.sim import build_scene

    batched = build_scene("nexus", seed=7, embodiment="franka", n_envs=2, headless=True)
    state = oracle_state(batched)
    assert state.shape == (2, len(MED_NAMES) * 7)
    assert np.array_equal(state[0], state[1])
    qpos = to_numpy(batched.robot.get_qpos())
    assert qpos.shape == (2, batched.robot.n_dofs)
