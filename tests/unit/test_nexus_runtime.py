"""Unit tests for tools/nexus_runtime.py's build-provenance reader (ADR-55).

The Nexus wheel is built from sibling checkouts and installed outside the uv
lock, so `uv.lock` says nothing about it and the receipt is gitignored: the
only way a run stays traceable to the engine sources that produced it is for
the rollout manifest to record the receipt. These tests pin the reader the
manifest calls, and its CON-8 CLI contract.
"""

import json
import sys
from pathlib import Path

import pytest
from cli_helpers import REPO_ROOT, run_tool

pytestmark = pytest.mark.unit

RECEIPT_NAME = ".nexus-runtime-receipt.json"


def runtime_module():
    sys.path.insert(0, str(REPO_ROOT / "tools"))
    import nexus_runtime

    return nexus_runtime


def write_receipt(root: Path, payload: dict) -> Path:
    path = root / RECEIPT_NAME
    path.write_text(json.dumps(payload))
    return path


def test_read_receipt_returns_the_engine_build_provenance(tmp_path):
    """CON-5, ADR-55: a run manifest can carry the wheel, feature, platform
    and the nexus/rapier/kiss3d commits behind the physics it measured."""
    payload = {
        "schema_version": 1,
        "wheel": "dimforge_nexus3d-0.1.0-cp39-abi3-macosx_11_0_arm64.whl",
        "feature": "metal",
        "version": "0.1.0",
        "platform": "darwin-arm64",
        "sources": {"nexus": {"commit": "a" * 40, "branch": "aisle-backend", "dirty": True}},
    }
    path = write_receipt(tmp_path, payload)

    result = runtime_module().read_receipt(tmp_path)

    assert result == {
        "installed": True,
        "path": str(path),
        "receipt": payload,
        "problem": None,
    }


def test_read_receipt_reports_absence_and_corruption_as_facts(tmp_path):
    """CON-8, ADR-55: an engine that was never built, or a receipt that no
    longer parses, is a reported problem the manifest can record — never an
    exception escaping into the rollout."""
    module = runtime_module()

    missing = module.read_receipt(tmp_path)
    assert missing["installed"] is False and missing["receipt"] is None
    assert "no nexus build receipt" in missing["problem"]

    (tmp_path / RECEIPT_NAME).write_text("{not json")
    broken = module.read_receipt(tmp_path)
    assert broken["installed"] is False and broken["receipt"] is None
    assert "unreadable" in broken["problem"]

    (tmp_path / RECEIPT_NAME).write_text('["a list"]')
    wrong_shape = module.read_receipt(tmp_path)
    assert wrong_shape["installed"] is False and "malformed" in wrong_shape["problem"]


def test_receipt_cli_emits_json_and_exits_on_the_verdict():
    """CON-8: JSON on stdout, logs on stderr, exit 0 iff a usable receipt
    exists — the same contract every AISLE CLI holds to."""
    proc = run_tool("nexus_runtime.py", "receipt")
    report = json.loads(proc.stdout)
    assert report["ok"] is report["installed"]
    assert proc.returncode == (0 if report["installed"] else 1)
    assert report["path"].endswith(RECEIPT_NAME)
