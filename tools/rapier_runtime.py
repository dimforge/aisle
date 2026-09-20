#!/usr/bin/env python3
"""Build and install AISLE's rapier Python module from a sibling checkout (ADR-56).

The rapier engine is optional and outside the uv lock, like Nexus: this builds
the `rapier3d` wheel with maturin from a rapier checkout, installs it into the
project environment with `uv pip`, and writes a receipt naming the source
commit so a run's engine build is traceable. It renders through the Nexus
viewer, so `tools/nexus_runtime.py install` has to have run too; `verify`
says so rather than letting the first render fail. `read_receipt()` is the
public reader the run manifest and `env_hash.sim_engine_hash` use. CON-8:
JSON on stdout, logs on stderr, exit 0 iff ok.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
from pathlib import Path

from nexus_runtime import ROOT, _commit, _run

RECEIPT = ROOT / ".rapier-runtime-receipt.json"
WHEEL_GLOB = "rapier3d-*.whl"
MANIFEST = "python/rapier-py-3d/Cargo.toml"


def receipt_path(root: Path | None = None) -> Path:
    """Where the build receipt lives for a checkout (default: this one)."""
    return (ROOT if root is None else Path(root)) / RECEIPT.name


def read_receipt(root: Path | None = None) -> dict:
    """The ADR-56 engine build provenance, for a run manifest to carry.

    Same contract as `nexus_runtime.read_receipt`: never raises, and an
    absent or malformed receipt is a reported fact. Returns {"installed",
    "path", "receipt", "problem"}.
    """
    path = receipt_path(root)
    result: dict = {"installed": False, "path": str(path), "receipt": None, "problem": None}
    if not path.is_file():
        result["problem"] = (
            f"no rapier build receipt at {path}; run `tools/rapier_runtime.py install`"
        )
        return result
    try:
        receipt = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        result["problem"] = f"unreadable rapier build receipt at {path}: {exc}"
        return result
    if not isinstance(receipt, dict):
        result["problem"] = f"malformed rapier build receipt at {path}: expected a JSON object"
        return result
    return {"installed": True, "path": str(path), "receipt": receipt, "problem": None}


def resolve_rapier(rapier: Path | None) -> Path:
    """The local checkout when given, otherwise the pinned GitHub source
    (ADR-56). The bindings live in the same repository the Nexus build
    patches, so both installers resolve to the same pinned commit."""
    if rapier is not None:
        return rapier
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from engine_sources import materialize

    report = materialize()
    print(f"using the pinned engine sources in {report['sources_dir']}", file=sys.stderr)
    return Path(report["sources"]["rapier"]["path"])


def _maturin_env() -> dict:
    """maturin refuses to run with both VIRTUAL_ENV and CONDA_PREFIX set
    (rapier's own python/dev.sh unsets them for the same reason)."""
    env = dict(os.environ)
    env.pop("CONDA_PREFIX", None)
    return env


def install(rapier: Path, python: Path, determinism: bool) -> dict:
    rapier = rapier.resolve()
    if not (rapier / MANIFEST).is_file():
        raise FileNotFoundError(f"{rapier} is not a rapier checkout with Python bindings")
    sources = {"rapier": _commit(rapier)}
    command = ["maturin", "build", "--release", "-m", MANIFEST, "-i", str(python)]
    if determinism:
        # libm transcendentals: cross-platform bit reproducibility (CON-5)
        command += ["-F", "determinism"]
    proc = subprocess.run(command, cwd=rapier, capture_output=True, text=True, env=_maturin_env())
    if proc.returncode != 0:
        raise RuntimeError(f"{' '.join(command)} failed:\n{proc.stderr.strip()}")
    wheels = sorted(
        (rapier / "target" / "wheels").glob(WHEEL_GLOB), key=lambda p: p.stat().st_mtime
    )
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
            "rapier3d",
            str(wheel),
        ],
        ROOT,
    )
    version = _run(
        [str(python), "-c", "import importlib.metadata as m; print(m.version('rapier3d'))"], ROOT
    )
    receipt = {
        "schema_version": 1,
        "wheel": wheel.name,
        "determinism": determinism,
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
        [str(python), "-c", "import importlib.metadata as m; print(m.version('rapier3d'))"], ROOT
    ).strip()
    if version != receipt["version"]:
        raise ValueError(
            f"installed rapier3d {version} differs from the receipt's {receipt['version']}"
        )
    for name, source in receipt["sources"].items():
        repo = Path(source["path"])
        if repo.is_dir() and _commit(repo)["commit"] != source["commit"]:
            raise ValueError(
                f"{name} checkout moved off the receipt's commit {source['commit'][:12]}"
            )
    import nexus_runtime

    renderer = nexus_runtime.read_receipt()
    if not renderer["installed"]:
        raise FileNotFoundError(
            "the rapier engine renders through the Nexus viewer (ADR-56) and "
            f"nexus3d is not installed: {renderer['problem']}"
        )
    return {**receipt, "renderer": renderer["receipt"]}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    inst = sub.add_parser("install", help="build the wheel and install it into the project venv")
    inst.add_argument(
        "--rapier",
        type=Path,
        default=None,
        help="rapier checkout to build (default: the commit engine-runtime.json pins, "
        "fetched from GitHub)",
    )
    inst.add_argument("--python", type=Path, default=Path(sys.executable))
    inst.add_argument(
        "--determinism",
        action="store_true",
        help="build with rapier's enhanced-determinism feature (libm transcendentals)",
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
            receipt = install(resolve_rapier(args.rapier), args.python, args.determinism)
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
