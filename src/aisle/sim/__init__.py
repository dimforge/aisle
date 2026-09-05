"""Physics-engine selection for the bridge and the harness (ADR-55).

The scene contract (SPEC 020) and the bridge contract (SPEC 030) are engine
neutral at the object level: the bridge talks to `robot`, entity, link and
camera objects through the small duck-typed surface Genesis exposes
natively. This package names the supported engines, resolves the attested
backend for each, and dispatches scene construction:

- ``genesis``: the frozen `aisle.scenes` builders, unchanged (CON-7).
- ``nexus``: `aisle.sim.nexus_backend`, which rebuilds the same scenes from
  the frozen pure functions (layout, placements, textures) on the Nexus
  GPU engine behind the same object surface.

Nothing here imports a simulator at module level (CON-12): the engine name
alone is a plain string decision, and every simulator import happens inside
the builder that needs it.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from typing import Any

ENGINES: tuple[str, ...] = ("genesis", "nexus")
DEFAULT_ENGINE = "genesis"
ENGINE_ENV_VAR = "AISLE_SIM_ENGINE"

# Backend names each engine accepts through AISLE_SIM_BACKEND (BRG-6).
ENGINE_BACKENDS: dict[str, tuple[str, ...]] = {
    "genesis": ("cpu", "metal", "cuda"),
    "nexus": ("webgpu", "metal", "cuda", "cpu"),
}


def normalize_engine(name: str | None) -> str:
    """Validate an engine name; None means the default. Unknown names are
    refused rather than defaulted, for the same reason TC-9 refuses an
    unknown perception rung: a typo must not silently attest another engine."""
    if name is None:
        return DEFAULT_ENGINE
    engine = name.strip().lower()
    if engine not in ENGINES:
        raise ValueError(f"unknown simulation engine {name!r}; expected one of {ENGINES}")
    return engine


def select_engine(env: Mapping[str, str] | None = None) -> str:
    """The engine named by ``AISLE_SIM_ENGINE`` (default ``genesis``)."""
    env = os.environ if env is None else env
    return normalize_engine(env.get(ENGINE_ENV_VAR))


def select_nexus_backend(sim_extra: str, platform_name: str, cuda_available: bool = False) -> str:
    """Nexus counterpart of `select_genesis_backend`: the portable ``sim``
    selection maps to Metal on macOS and WebGPU elsewhere; ``cuda`` is the
    Linux-only explicit opt-in and fails closed without a device."""
    if sim_extra == "sim":
        return "metal" if platform_name == "Darwin" else "webgpu"
    if sim_extra != "cuda":
        raise ValueError(f"unknown simulation extra {sim_extra!r}; expected 'sim' or 'cuda'")
    if platform_name != "Linux":
        raise ValueError("the locked CUDA simulation extra is supported only on Linux")
    if not cuda_available:
        raise ValueError("the CUDA simulation extra requires an available CUDA device")
    return "cuda"


def select_sim_backend(
    engine: str, sim_extra: str, platform_name: str, cuda_available: bool = False
) -> str:
    """Resolve (engine, extra, platform) to one attested backend name."""
    engine = normalize_engine(engine)
    if engine == "genesis":
        from aisle.scenes.pharmacy import select_genesis_backend

        return select_genesis_backend(sim_extra, platform_name, cuda_available)
    return select_nexus_backend(sim_extra, platform_name, cuda_available)


def validate_backend(engine: str, backend: str | None) -> str | None:
    """Refuse a backend name the engine does not know (bridge config)."""
    engine = normalize_engine(engine)
    if backend is not None and backend not in ENGINE_BACKENDS[engine]:
        raise ValueError(
            f"unknown simulation backend {backend!r} for engine {engine!r}; "
            f"expected one of {ENGINE_BACKENDS[engine]}"
        )
    return backend


def engine_available(engine: str) -> bool:
    """Whether the engine's Python package is importable (collection-safe:
    uses find_spec, never imports the simulator)."""
    import importlib.util

    module = {"genesis": "genesis", "nexus": "nexus3d"}[normalize_engine(engine)]
    return importlib.util.find_spec(module) is not None


def engine_version(engine: str) -> str:
    """The installed simulator's version string, for `bridge_info`."""
    engine = normalize_engine(engine)
    if engine == "genesis":
        import genesis

        return str(genesis.__version__)
    from importlib.metadata import PackageNotFoundError, version

    try:
        return version("dimforge-nexus3d")
    except PackageNotFoundError:
        import nexus3d

        return str(getattr(nexus3d, "__version__", "unknown"))


def build_scene(engine: str, *args: Any, **kwargs: Any):
    """`aisle.scenes.pharmacy.build_scene` on the chosen engine."""
    engine = normalize_engine(engine)
    if engine == "genesis":
        from aisle.scenes.pharmacy import build_scene as genesis_build_scene

        return genesis_build_scene(*args, **kwargs)
    from aisle.sim.nexus_backend import build_scene as nexus_build_scene

    return nexus_build_scene(*args, **kwargs)


def build_store(engine: str, *args: Any, **kwargs: Any):
    """`aisle.scenes.store.build_store` on the chosen engine."""
    engine = normalize_engine(engine)
    if engine == "genesis":
        from aisle.scenes.store import build_store as genesis_build_store

        return genesis_build_store(*args, **kwargs)
    from aisle.sim.nexus_backend import build_store as nexus_build_store

    return nexus_build_store(*args, **kwargs)
