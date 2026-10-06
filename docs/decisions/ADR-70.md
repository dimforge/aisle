# ADR-70 — Engine wheels from PyPI, inside the lock

Status: PROPOSED — owner review required under CON-14, with ADR-67 and ADR-68.
Amends: ADR-67 (scope, attestation) and ADR-68 (installation, pinning).
Trigger: `dimforge-nexus3d` 0.2.0 and `rapier3d` 0.36.1 were published to PyPI
with every binding AISLE uses.

## Context

ADR-67 and ADR-68 built the Nexus and rapier wheels from source:
`engine-runtime.json` pinned a commit of each repository,
`tools/nexus_runtime.py` and `tools/rapier_runtime.py` built and installed the
wheels outside `uv.lock`, and a gitignored build receipt was the only provenance
a run could record. The trusted checker refused a Nexus or rapier run whose
receipt was missing or named a dirty checkout, and any `uv sync` removed the
wheels. ADR-67 named "the engines are installed from released versions inside
the lock" as the first precondition for engine results entering the measured
record.

Both engines now have releases on PyPI. The Nexus wheel is built with its
default `webgpu` feature only: it has no `with_metal` or `with_cuda`, and wheels
exist for macOS arm64, Linux x86_64 and Windows x64 (rapier3d also covers Linux
aarch64 and musl).

## Decision

1. **The wheels are locked dependencies of the `sim` extra.**
   `dimforge-nexus3d==0.2.0` and `rapier3d==0.36.1` join the `sim` extra in
   `pyproject.toml`, under a marker limited to the platforms Nexus ships wheels
   for (rapier renders through Nexus, so it follows the same set).
   `uv sync --extra sim` installs both; the CUDA extra does not, because the
   published Nexus wheel has no CUDA feature.
2. **The source-build machinery is retired.** `engine-runtime.json`,
   `tools/engine_sources.py`, `tools/nexus_runtime.py`, `tools/rapier_runtime.py`
   and the receipts they wrote are removed. An unreleased engine change is tried
   by installing a locally built wheel over the locked one; such a run is
   recorded as unattested and refused by trusted runs, like any other
   out-of-lock dist.
3. **Provenance comes from the lock.** `tools/env_hash.py --sim-engine` keeps
   the engine digest (`sim_engine_hash` over the engine name and
   `src/aisle/sim/**`) but no longer folds in or requires a build receipt. The
   engine's wheel is identified the way Genesis is: by `uv.lock`, the
   `uv sync --locked --check` selection check, and the PEP 610 record of every
   dist in the engine-aware attested set. The manifest keeps its
   `sim_engine_build` key, now `{"engine", "sim_engine_hash", "n_files"}`.
4. **Nexus resolves `webgpu` on every platform.** `select_nexus_backend("sim",
   ...)` returns `webgpu` on macOS too (it returned `metal`), since that is the
   only GPU feature of the locked wheel. `metal`, `cuda` and `cpu` stay
   accepted through `AISLE_SIM_BACKEND` for development builds.

## Consequences

- A Nexus or rapier run can now pass the dist attestation, so the first of
  ADR-67's three preconditions holds. The other two (the `spec-change` PR
  generalizing SPEC 020/030, and a re-established frozen baseline) still stand,
  so neither engine enters the measured record yet.
- `uv sync --extra sim` no longer removes the engines, and no install step
  remains beyond the sync.
- Nexus runs on macOS step through wgpu over Metal instead of Nexus's native
  Metal backend, and wgpu validates buffer usages where native Metal did not.
  The published 0.2.0 reads multibody joint velocities from a buffer without
  `COPY_SRC`, which panics on WebGPU. Only
  `test_zeroing_velocities_brings_the_arm_to_rest` reads them, and it is a
  strict expected failure on 0.2.0; the fix is on nexus branch
  `fix-webgpu-readback-usages` and needs a release.
- Determinism (`deterministic = true`) is per backend: two WebGPU runs replay
  bit for bit, but a WebGPU run is not expected to match a native Metal run.
- Platforms without a Nexus wheel (macOS x86_64, Linux aarch64) have Genesis
  only; `--sim-engine nexus` or `rapier` refuses there at the `sim_engine` gate.
- Editing `tools/env_hash.py` makes trusted runs refuse on this branch until it
  merges, so this lands as an env-change PR, not mid-campaign (ADR-67).
