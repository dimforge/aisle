#!/usr/bin/env python3
"""Materialize the pinned engine sources from GitHub (ADR-55, ADR-56).

`engine-runtime.json` pins the two repositories AISLE builds wheels from,
nexus and rapier, the same way `dora-runtime.json` pins the Dora CLI. Each is
fetched by exact commit.

kiss3d is deliberately absent, and so are the rapier crates the engine links
against: nexus's own Cargo manifest patches kiss3d from git by rev and takes
the published rapier release from crates.io, so cargo resolves and caches
both. Pinning them here as well would be two sources of
truth for one dependency.

The engine installers use this when no local checkout is passed, so a machine
with no sibling working copies can still build the wheels, and the commit a
wheel came from is the pinned one rather than whatever happened to be checked
out. CON-8: JSON on stdout, logs on stderr, exit 0 iff ok.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PINS = ROOT / "engine-runtime.json"
DEFAULT_SOURCES = ROOT / ".engine-sources"
NAMES = ("nexus", "rapier")


def load_pins(path: Path | None = None) -> dict:
    """The pinned sources, refusing a file that does not name both."""
    pins = json.loads((path or PINS).read_text())
    missing = [n for n in NAMES if n not in pins.get("sources", {})]
    if missing:
        raise ValueError(f"engine pins miss {missing}; both of {NAMES} are built here")
    for name, source in pins["sources"].items():
        for field in ("repository", "branch", "commit"):
            if not source.get(field):
                raise ValueError(f"engine pin {name!r} has no {field}")
        if len(source["commit"]) != 40:
            raise ValueError(f"engine pin {name!r} must name a full commit, not a prefix")
    return pins


def _run(command: list[str], cwd: Path) -> str:
    proc = subprocess.run(command, cwd=cwd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"{' '.join(command)} failed:\n{proc.stderr.strip()}")
    return proc.stdout.strip()


def fetch(name: str, source: dict, into: Path) -> dict:
    """One pinned checkout at `into/<name>`, fetched by exact commit.

    Reuses an existing checkout when it already sits on the pinned commit, so
    a rebuild does not re-download or invalidate the cargo target directory."""
    repo = into / name
    if (repo / ".git").is_dir():
        # an interrupted fetch leaves an initialized repo with no HEAD, so ask
        # for the revision rather than assuming one exists
        head = subprocess.run(
            ["git", "rev-parse", "--verify", "-q", "HEAD"],
            cwd=repo,
            capture_output=True,
            text=True,
        ).stdout.strip()
        if head == source["commit"]:
            return {"name": name, "path": str(repo), "commit": head, "fetched": False}
    else:
        repo.mkdir(parents=True, exist_ok=True)
        _run(["git", "init", "-q"], repo)
        _run(["git", "remote", "add", "origin", source["repository"]], repo)
    _run(["git", "fetch", "--depth", "1", "origin", source["commit"]], repo)
    _run(["git", "checkout", "-q", "--detach", "FETCH_HEAD"], repo)
    head = _run(["git", "rev-parse", "HEAD"], repo)
    if head != source["commit"]:
        raise ValueError(f"{name}: fetched {head} but the pin names {source['commit']}")
    return {"name": name, "path": str(repo), "commit": head, "fetched": True}


def materialize(into: Path | None = None, pins_path: Path | None = None) -> dict:
    """The pinned checkouts. Returns {name: {path, commit, fetched}}."""
    into = Path(into) if into else DEFAULT_SOURCES
    pins = load_pins(pins_path)
    into.mkdir(parents=True, exist_ok=True)
    results = [fetch(name, pins["sources"][name], into) for name in NAMES]
    return {
        "ok": True,
        "sources_dir": str(into),
        "sources": {r["name"]: r for r in results},
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--into", type=Path, default=DEFAULT_SOURCES)
    parser.add_argument("--pins", type=Path, default=PINS)
    args = parser.parse_args(argv)
    try:
        report = materialize(args.into, args.pins)
    except (RuntimeError, ValueError, OSError, json.JSONDecodeError) as exc:
        print(json.dumps({"ok": False, "error": str(exc)}))
        print(str(exc), file=sys.stderr)
        return 1
    print(json.dumps(report))
    return 0


if __name__ == "__main__":
    sys.exit(main())
