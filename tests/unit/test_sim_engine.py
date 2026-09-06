"""Engine selection for the bridge and the harness (ADR-55).

Unit-marked: nothing here imports a simulator. The nexus backend module is
imported to prove it stays sim-free at import time (CON-12), like
`aisle.scenes.pharmacy`.
"""

import sys

import numpy as np
import pytest

from aisle.sim import (
    DEFAULT_ENGINE,
    ENGINE_BACKENDS,
    ENGINES,
    normalize_engine,
    select_engine,
    select_nexus_backend,
    select_sim_backend,
    validate_backend,
)

pytestmark = pytest.mark.unit


def test_default_engine_is_genesis():
    """ADR-55: the frozen Genesis path stays the default; the engine is an
    explicit, graph-declared choice."""
    assert DEFAULT_ENGINE == "genesis"
    assert select_engine({}) == "genesis"
    assert select_engine({"AISLE_SIM_ENGINE": "nexus"}) == "nexus"
    assert select_engine({"AISLE_SIM_ENGINE": " Nexus "}) == "nexus"


def test_unknown_engine_is_refused_not_defaulted():
    """ADR-55 (same rule as TC-9's rung): a typo must not silently attest
    another engine."""
    with pytest.raises(ValueError, match="unknown simulation engine"):
        normalize_engine("bullet")
    with pytest.raises(ValueError, match="unknown simulation engine"):
        select_engine({"AISLE_SIM_ENGINE": ""})


@pytest.mark.parametrize(
    ("sim_extra", "platform_name", "cuda_available", "expected"),
    [
        ("sim", "Darwin", False, "metal"),
        ("sim", "Darwin", True, "metal"),
        ("sim", "Linux", False, "webgpu"),
        ("sim", "Linux", True, "webgpu"),
        ("cuda", "Linux", True, "cuda"),
    ],
)
def test_select_nexus_backend(sim_extra, platform_name, cuda_available, expected):
    """ADR-55: the portable extra never changes physics because a GPU is
    visible; CUDA is the explicit Linux opt-in."""
    assert select_nexus_backend(sim_extra, platform_name, cuda_available) == expected


@pytest.mark.parametrize(
    ("sim_extra", "platform_name", "cuda_available", "message"),
    [
        ("cuda", "Darwin", True, "only on Linux"),
        ("cuda", "Linux", False, "requires an available CUDA device"),
        ("gpu", "Linux", True, "unknown simulation extra"),
    ],
)
def test_select_nexus_backend_fails_closed(sim_extra, platform_name, cuda_available, message):
    with pytest.raises(ValueError, match=message):
        select_nexus_backend(sim_extra, platform_name, cuda_available)


def test_select_sim_backend_dispatches_per_engine():
    """ADR-55: the Genesis table is the frozen module's own, unchanged."""
    assert select_sim_backend("genesis", "sim", "Darwin") == "metal"
    assert select_sim_backend("genesis", "sim", "Linux") == "cpu"
    assert select_sim_backend("nexus", "sim", "Linux") == "webgpu"


def test_validate_backend_per_engine():
    """BRG-6: AISLE_SIM_BACKEND must name a backend the engine can run."""
    for engine in ENGINES:
        for backend in ENGINE_BACKENDS[engine]:
            assert validate_backend(engine, backend) == backend
    assert validate_backend("nexus", None) is None
    with pytest.raises(ValueError, match="unknown simulation backend"):
        validate_backend("genesis", "webgpu")


def test_backend_modules_stay_sim_free():
    """CON-12: importing the selection package and the nexus backend must
    import neither genesis nor nexus3d."""
    import importlib

    importlib.import_module("aisle.sim")
    importlib.import_module("aisle.sim.nexus_backend")
    assert "genesis" not in sys.modules
    assert "nexus3d" not in sys.modules


def test_nexus_physics_constants_are_declared():
    """SCN-2 spirit: the Nexus realization constants live in a toml, outside
    the frozen scene directory (CON-7)."""
    from aisle.sim.nexus_backend import load_nexus_physics

    physics = load_nexus_physics()
    assert physics["robot"]["default_kp"] == 100.0  # Genesis's URDF default
    assert physics["robot"]["default_kv"] == 10.0
    assert len(physics["ground"]["size"]) == 3
    # the pinch-grasp solver settings (ADR-55): friction solved with the
    # normals, Genesis's max() friction rule
    assert physics["sim"]["friction_in_bias_pass"] is True
    assert physics["sim"]["friction_combine_rule"] == "max"
    assert physics["sim"]["internal_pgs_iterations"] >= 1
    assert isinstance(physics["sim"]["implicit_coriolis"], bool)
    # render settings the Nexus viewer applies to its sensor cameras
    assert physics["camera"]["msaa_samples"] in (1, 4)
    assert 0.0 <= physics["camera"]["shadow_softness"] <= 1.0
    assert len(physics["ground"]["color"]) == 4
    assert physics["camera"]["shadow_resolution"] >= 1024
    assert 1 <= physics["camera"]["shadow_atlas_layers"] <= 16


def test_checkerboard_texture_alternates():
    """The floor texture is a deterministic checkerboard of the two ground
    colors, `squares` cells a side at `px_per_square` pixels each."""
    from aisle.sim.nexus_backend import checkerboard_texture

    image = checkerboard_texture(4, [0.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0], px_per_square=2)
    assert image.shape == (8, 8, 3) and image.dtype == np.uint8
    assert image[0, 0].tolist() == [0, 0, 0] and image[0, 2].tolist() == [255, 255, 255]
    assert image[2, 0].tolist() == [255, 255, 255] and image[2, 2].tolist() == [0, 0, 0]
    assert image[1, 1].tolist() == [0, 0, 0]  # 2x2 pixel cells


def test_uv_cube_asset_parses():
    """T2 labels on Nexus reuse the committed UV cube: 12 faces, one vertex
    per (position, uv) pair, uvs inside the unit square."""
    from aisle.sim.nexus_backend import load_uv_cube

    positions, faces, uvs = load_uv_cube()
    assert len(faces) == 12
    assert len(positions) == len(uvs) == 24
    assert all(0.0 <= u <= 1.0 and 0.0 <= v <= 1.0 for u, v in uvs)
    assert all(abs(abs(c) - 0.5) < 1e-9 for p in positions for c in p)


def test_lookat_transform_matches_genesis_convention():
    """VER-8 / SCN-5: the Nexus camera pose is Genesis's look-at transform
    (OpenGL frame, up = +z): -Z looks at the target, +Y is up."""
    import numpy as np

    from aisle.sim.nexus_backend import pos_lookat_up_to_transform

    transform = pos_lookat_up_to_transform((0.55, 0.0, 1.2), (0.55, 0.0, 0.2))
    forward = -transform[:3, 2]
    assert np.allclose(forward, [0.0, 0.0, -1.0])
    assert np.allclose(transform[:3, 3], [0.55, 0.0, 1.2])
    assert np.allclose(transform[:3, :3] @ transform[:3, :3].T, np.eye(3), atol=1e-9)
    # a tilted look-at keeps world +z in the camera's up half-plane
    tilted = pos_lookat_up_to_transform((1.0, 0.0, 1.0), (0.0, 0.0, 0.0))
    assert tilted[2, 1] > 0.0
