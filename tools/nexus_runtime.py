#!/usr/bin/env python3
"""Build and install AISLE's Nexus Python module from sibling checkouts (ADR-55).

Nexus is an optional engine outside the uv lock: this builds the `nexus3d`
wheel with maturin from a nexus checkout (whose Cargo manifest resolves the
rapier and kiss3d crates it links), installs it into the project environment
with `uv pip`, and writes a receipt naming both source commits so a run's
`bridge_info` can be traced to the exact engine sources. `read_receipt()`
is the public reader for that provenance: the run manifest and
`env_hash.sim_engine_hash` record it, since the receipt is gitignored and
local to the machine that built the wheel. CON-8: JSON on stdout, logs on
stderr, exit 0 iff ok.
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RECEIPT = ROOT / ".nexus-runtime-receipt.json"
WHEEL_GLOB = "dimforge_nexus3d-*.whl"


def receipt_path(root: Path | None = None) -> Path:
    """Where the build receipt lives for a checkout (default: this one)."""
    return (ROOT if root is None else Path(root)) / RECEIPT.name


def read_receipt(root: Path | None = None) -> dict:
    """The ADR-55 engine build provenance, for a run manifest to carry.

    The receipt is gitignored and local to the machine that built the
    wheel, so recording it at rollout time is the only trace of which
    engine sources produced a run. Never raises: an absent or malformed
    receipt is a reported fact, not an exception.

    Returns {"installed": bool, "path": str, "receipt": dict | None,
    "problem": str | None}; `receipt` holds the wheel name, version,
    cargo feature, platform and the nexus/rapier/kiss3d commits.
    """
    path = receipt_path(root)
    result: dict = {"installed": False, "path": str(path), "receipt": None, "problem": None}
    if not path.is_file():
        result["problem"] = (
            f"no nexus build receipt at {path}; run `tools/nexus_runtime.py install`"
        )
        return result
    try:
        receipt = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        result["problem"] = f"unreadable nexus build receipt at {path}: {exc}"
        return result
    if not isinstance(receipt, dict):
        result["problem"] = f"malformed nexus build receipt at {path}: expected a JSON object"
        return result
    return {"installed": True, "path": str(path), "receipt": receipt, "problem": None}


def resolve_sources(
    nexus: Path | None, rapier: Path | None, kiss3d: Path | None
) -> tuple[Path, Path | None, Path | None]:
    """Local checkouts when given, otherwise the pinned GitHub source.

    Passing any path keeps the whole set local, so a developer working across
    sibling checkouts never gets a surprise mix of local and pinned sources
    (ADR-55). With none given, only nexus is fetched: its manifest takes rapier
    from crates.io and patches kiss3d from git by rev, and cargo resolves both."""
    if nexus is not None or rapier is not None or kiss3d is not None:
        return (nexus or ROOT.parent / "nexus", rapier, kiss3d)
    # pinned build: cargo fetches rapier from crates.io and kiss3d from git
    # by rev, so no local copy of either is needed
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from engine_sources import materialize

    report = materialize()
    paths = report["sources"]
    print(f"using the pinned engine sources in {report['sources_dir']}", file=sys.stderr)
    return (Path(paths["nexus"]["path"]), None, None)


def _run(command: list[str], cwd: Path) -> str:
    proc = subprocess.run(command, cwd=cwd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"{' '.join(command)} failed:\n{proc.stderr.strip()}")
    return proc.stdout


def _commit(repo: Path) -> dict:
    head = _run(["git", "rev-parse", "HEAD"], repo).strip()
    dirty = bool(_run(["git", "status", "--porcelain"], repo).strip())
    branch = _run(["git", "rev-parse", "--abbrev-ref", "HEAD"], repo).strip()
    return {"path": str(repo), "commit": head, "branch": branch, "dirty": dirty}


def gpu_feature(system: str) -> str:
    """The nexus3d cargo feature for this host's GPU API."""
    return "metal" if system == "Darwin" else "webgpu"


def install(
    nexus: Path, rapier: Path | None, kiss3d: Path | None, python: Path, feature: str
) -> dict:
    nexus = nexus.resolve()
    if not (nexus / "crates" / "nexus_python3d" / "Cargo.toml").is_file():
        raise FileNotFoundError(f"{nexus} is not a nexus checkout")
    sources = {"nexus": _commit(nexus)}
    if rapier is not None:
        rapier = rapier.resolve()
        if not (rapier / "crates" / "rapier3d-urdf").is_dir():
            raise FileNotFoundError(f"{rapier} is not a rapier checkout")
        sources["rapier"] = _commit(rapier)
    if kiss3d is not None:
        kiss3d = kiss3d.resolve()
        if not (kiss3d / "src" / "window").is_dir():
            raise FileNotFoundError(f"{kiss3d} is not a kiss3d checkout")
        sources["kiss3d"] = _commit(kiss3d)
    _run(
        [
            "maturin",
            "build",
            "--release",
            "-m",
            "crates/nexus_python3d/Cargo.toml",
            "--features",
            feature,
            "-i",
            str(python),
        ],
        nexus,
    )
    wheels = sorted((nexus / "target" / "wheels").glob(WHEEL_GLOB), key=lambda p: p.stat().st_mtime)
    if not wheels:
        raise FileNotFoundError("maturin produced no wheel")
    wheel = wheels[-1]
    _run(
        [
            "uv",
            "pip",
            "install",
            "--python",
            str(python),
            "--reinstall-package",
            "dimforge-nexus3d",
            str(wheel),
        ],
        ROOT,
    )
    version = _run(
        [str(python), "-c", "import importlib.metadata as m; print(m.version('dimforge-nexus3d'))"],
        ROOT,
    )
    receipt = {
        "schema_version": 1,
        "wheel": wheel.name,
        "feature": feature,
        "version": version.strip(),
        "platform": f"{platform.system().lower()}-{platform.machine()}",
        "sources": sources,
    }
    RECEIPT.write_text(json.dumps(receipt, indent=2) + "\n")
    return receipt


def verify(python: Path) -> dict:
    read = read_receipt()
    if not read["installed"]:
        raise FileNotFoundError(read["problem"])
    receipt = read["receipt"]
    version = _run(
        [str(python), "-c", "import importlib.metadata as m; print(m.version('dimforge-nexus3d'))"],
        ROOT,
    ).strip()
    if version != receipt["version"]:
        raise ValueError(
            f"installed nexus3d {version} differs from the receipt's {receipt['version']}"
        )
    for name, source in receipt["sources"].items():
        repo = Path(source["path"])
        if repo.is_dir() and _commit(repo)["commit"] != source["commit"]:
            raise ValueError(
                f"{name} checkout moved off the receipt's commit {source['commit'][:12]}"
            )
    return receipt


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    inst = sub.add_parser("install", help="build the wheel and install it into the project venv")
    inst.add_argument(
        "--nexus",
        type=Path,
        default=None,
        help="nexus checkout to build (default: the commit engine-runtime.json pins, "
        "fetched from GitHub)",
    )
    inst.add_argument(
        "--rapier", type=Path, default=None, help="rapier checkout the nexus manifest patches to"
    )
    inst.add_argument(
        "--kiss3d", type=Path, default=None, help="kiss3d checkout the nexus manifest patches to"
    )
    inst.add_argument("--python", type=Path, default=Path(sys.executable))
    inst.add_argument(
        "--feature", default=gpu_feature(platform.system()), help="nexus3d GPU feature"
    )
    ver = sub.add_parser("verify", help="check the installed wheel against the receipt")
    ver.add_argument("--python", type=Path, default=Path(sys.executable))
    sub.add_parser("receipt", help="print the build provenance a run manifest records")
    args = parser.parse_args(argv)
    if args.command == "receipt":
        read = read_receipt()
        print(json.dumps({"ok": read["installed"], **read}))
        if read["problem"]:
            print(read["problem"], file=sys.stderr)
        return 0 if read["installed"] else 1
    try:
        if args.command == "install":
            nexus, rapier, kiss3d = resolve_sources(args.nexus, args.rapier, args.kiss3d)
            receipt = install(nexus, rapier, kiss3d, args.python, args.feature)
        else:
            receipt = verify(args.python)
    except (RuntimeError, FileNotFoundError, ValueError) as exc:
        print(json.dumps({"ok": False, "error": str(exc)}))
        print(str(exc), file=sys.stderr)
        return 1
    print(json.dumps({"ok": True, **receipt}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
