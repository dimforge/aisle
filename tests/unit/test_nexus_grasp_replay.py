"""The grasp replay probe keeps the CLI contract on a run with nothing to
replay (CON-8): JSON on stdout and a nonzero exit, not a traceback."""

from __future__ import annotations

import json
import sys

import pyarrow as pa
import pytest
from cli_helpers import REPO_ROOT

sys.path.insert(0, str(REPO_ROOT / "tools"))

import nexus_grasp_replay  # noqa: E402

pytestmark = pytest.mark.unit


def _empty_trace(path):
    schema = pa.schema([("sim_time_ns", pa.int64()), ("data", pa.list_(pa.float32()))])
    with pa.OSFile(str(path), "wb") as sink, pa.ipc.new_stream(sink, schema):
        pass


def test_an_empty_trace_is_a_json_refusal(tmp_path, capsys):
    """CON-8: an empty command trace refuses with JSON before any engine loads."""
    traces = tmp_path / "traces"
    traces.mkdir()
    for name in ("budget-guard__joint_cmd_safe", "budget-guard__gripper_cmd_safe"):
        _empty_trace(traces / f"{name}.arrow")
    assert nexus_grasp_replay.main(["--run", str(tmp_path)]) == 1
    report = json.loads(capsys.readouterr().out)
    assert report["ok"] is False and "no joint or gripper commands" in report["error"]
