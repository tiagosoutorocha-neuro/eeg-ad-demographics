import json
import shutil
from pathlib import Path

import pytest

from eegad.config import load_context, plan_hash
from eegad.reproduction import compare

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture
def repo_copy(tmp_path, monkeypatch):
    """A minimal copy of the repository configuration with data paths given by environment variables."""
    (tmp_path / "config").mkdir()
    for f in ("analysis_plan.json", "ds004504_v1.0.9_manifest.json", "paths.example.yaml"):
        shutil.copy(REPO / "config" / f, tmp_path / "config" / f)
    monkeypatch.setenv("EEGAD_REPO_ROOT", str(tmp_path))
    monkeypatch.setenv("EEGAD_BRAINLAT_DIR", str(tmp_path / "brainlat"))
    monkeypatch.setenv("EEGAD_DS004504_DIR", str(tmp_path / "ds004504"))
    monkeypatch.setenv("EEGAD_OUTPUTS_DIR", str(tmp_path / "outputs"))
    return tmp_path


def test_plan_hash_is_stable():
    plan = json.loads((REPO / "config" / "analysis_plan.json").read_text(encoding="utf-8"))
    assert plan_hash(plan) == plan_hash(json.loads(json.dumps(plan)))
    assert len(plan_hash(plan)) == 16


def test_frozen_plan_blocks_changes(repo_copy):
    ctx = load_context("00", require_frozen=False)
    ctx.freeze_plan()
    load_context("02")                                      # unchanged plan: fine
    plan_file = repo_copy / "config" / "analysis_plan.json"
    plan = json.loads(plan_file.read_text(encoding="utf-8"))
    plan["seed"] = 7
    plan_file.write_text(json.dumps(plan), encoding="utf-8")
    with pytest.raises(RuntimeError, match="differs from the plan frozen"):
        load_context("02")


def test_result_log_replaces_entries_of_a_rerun(repo_copy):
    ctx = load_context("00", require_frozen=False)
    ctx.log.record("A", "x", "1")
    ctx.log.key("k", 1.0)
    ctx2 = load_context("00", require_frozen=False)         # same notebook again
    assert [e["item"] for e in ctx2.log.entries] == [] and "k" not in ctx2.log.keys


def test_compare_statuses():
    original = {"a": {"value": 0.68, "decimals": 2}, "b": {"value": 0.001, "kind": "lt"},
                "c": {"value": 0.123456789, "decimals": None}, "d": {"value": 1.0, "decimals": 3},
                "e": {"value": None, "decimals": None}}
    new = {"a": 0.6849, "b": 0.0004, "c": 0.123456789, "d": 1.01, "e": None, "f": 3.0}
    t = compare(new, original).set_index("key")["status"]
    assert t["a"] == "match" and t["b"] == "match" and t["c"] == "match"
    assert t["d"] == "differs" and t["e"] == "match" and t["f"] == "new only"
