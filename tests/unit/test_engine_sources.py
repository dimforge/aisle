"""The engine sources are pinned by commit, not tracked by branch (ADR-55,
ADR-56, CON-5).

A wheel built from "whatever is checked out" cannot be traced back to a
revision, which is the same reason `dora-runtime.json` pins the Dora CLI. The
pin file is validated rather than trusted: a prefix instead of a full commit,
or a missing repository, would silently widen what a run could have been
built from.
"""

from __future__ import annotations

import json
import sys

import pytest
from cli_helpers import REPO_ROOT

sys.path.insert(0, str(REPO_ROOT / "tools"))

from engine_sources import NAMES, load_pins  # noqa: E402

pytestmark = pytest.mark.unit


def test_committed_pins_name_a_full_commit_for_every_built_engine():
    """CON-5: the pins AISLE ships resolve to exact revisions."""
    pins = load_pins()
    assert set(pins["sources"]) == set(NAMES)
    for name in NAMES:
        source = pins["sources"][name]
        assert source["repository"].startswith("https://"), name
        assert len(source["commit"]) == 40, name
        assert int(source["commit"], 16) >= 0, name
        assert source["branch"], name


def test_the_linked_crates_are_pinned_once_in_the_engine_manifest():
    """ADR-56: kiss3d and the rapier crates the engine links against are
    resolved by cargo from nexus's own manifest (rapier from its crates.io
    release, kiss3d by git revision). Pinning them here too would
    be a second source of truth that drifts silently."""
    pins = load_pins()
    assert "kiss3d" not in pins["sources"]


@pytest.mark.parametrize(
    "mangle,expected",
    [
        (lambda p: p["sources"].pop("rapier"), "miss"),
        (lambda p: p["sources"]["nexus"].__setitem__("commit", "abc1234"), "full commit"),
        (lambda p: p["sources"]["nexus"].__setitem__("repository", ""), "no repository"),
    ],
)
def test_a_loose_pin_is_refused(tmp_path, mangle, expected):
    """CON-8: the loader refuses rather than defaulting, so a half-specified
    pin cannot reach a build."""
    pins = json.loads((REPO_ROOT / "engine-runtime.json").read_text())
    mangle(pins)
    path = tmp_path / "engine-runtime.json"
    path.write_text(json.dumps(pins))
    with pytest.raises(ValueError, match=expected):
        load_pins(path)
