#!/usr/bin/env python3
"""Build and install AISLE's Nexus Python module from sibling checkouts (ADR-55).

Nexus is an optional engine outside the uv lock: this builds the `nexus3d`
wheel with maturin from a nexus checkout (whose Cargo manifest patches the
rapier crates to a rapier checkout), installs it into the project environment
with `uv pip`, and writes a receipt naming both source commits so a run's
`bridge_info` can be traced to the exact engine sources. CON-8: JSON on
stdout, logs on stderr, exit 0 iff ok.
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
    if not RECEIPT.is_file():
        raise FileNotFoundError(f"no receipt at {RECEIPT}; run `install` first")
    receipt = json.loads(RECEIPT.read_text())
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
    inst.add_argument("--nexus", type=Path, default=ROOT.parent / "nexus")
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
    args = parser.parse_args(argv)
    try:
        if args.command == "install":
            receipt = install(args.nexus, args.rapier, args.kiss3d, args.python, args.feature)
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
