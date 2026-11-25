"""Post hoc amendments (notebooks 09-11) and the article figures and tables (notebook 12)."""
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from eegad.article_tables import build_blocks, feature_definitions, write_docx, write_html
from eegad.config import load_context
from eegad.posthoc import (POST_HOC_PLAN_2, calibration_bins, calibration_metrics, freeze_post_hoc_plan, recalibrate_loo,
                           recalibration_analysis, recalibration_table, wilson_interval)

REPO = Path(__file__).resolve().parents[1]


def miscalibrated(n1=150, n0=150, seed=0):
    """Discriminating but timid predictions (true calibration slope about 2.5, risk underestimated)."""
    rng = np.random.default_rng(seed)
    y = np.r_[np.ones(n1), np.zeros(n0)].astype(int)
    z = np.r_[rng.normal(1.5, 1, n1), rng.normal(0, 1, n0)] - 0.75
    p = 1 / (1 + np.exp(-(0.4 * z - 0.3)))
    return pd.DataFrame({"subject_id": [f"s{i:03d}" for i in range(len(y))], "y": y, "p": p})


def test_recalibration_corrects_calibration():
    d = miscalibrated()
    res, subj = recalibration_analysis(d, "test", 0.5, n_boot=50, seed=1)
    fz, ab = res["versions"]["frozen"], res["versions"]["intercept_and_slope"]
    assert fz["calibration_slope"] > 1.8 and fz["citl"] > 0.2
    assert abs(ab["calibration_slope"] - 1) < 0.2 and abs(ab["citl"]) < 0.1
    assert ab["brier"] < fz["brier"]
    assert res["equations"]["intercept_and_slope"]["b"] > 1.8
    assert list(subj.columns) == ["subject_id", "y", "p_frozen", "p_intercept_only", "p_intercept_and_slope"]
    tab = recalibration_table({"RQ3a": res})
    assert len(tab) == 3 and tab.iloc[0]["AUC"] != "as frozen (see note)"


def test_leave_one_out_excludes_the_participant():
    d = miscalibrated(20, 20, seed=3)
    y, p = d["y"].to_numpy(), d["p"].to_numpy()
    loo = recalibrate_loo(y, p, "intercept_only")
    flipped = y.copy()
    flipped[0] = 1 - flipped[0]
    assert np.isclose(recalibrate_loo(flipped, p, "intercept_only")[0], loo[0])   # own label is never used


def test_calibration_helpers():
    assert np.isclose(wilson_interval(0, 10)[0], 0.0) and np.isclose(wilson_interval(10, 10)[1], 1.0)
    d = miscalibrated(40, 40)
    b = calibration_bins(d["y"], d["p"], 5)
    assert len(b) == 5 and b["n"].sum() == 80 and (b["lo"] <= b["observed"]).all()
    m = calibration_metrics(d["y"], d["p"])
    assert 0 < m["brier"] < 0.25 and m["n"] == 80


@pytest.fixture
def repo_with_posthoc(tmp_path, monkeypatch):
    (tmp_path / "config").mkdir()
    for f in ("analysis_plan.json", "post_hoc_plan.json", "post_hoc_plan_2.json", "ds004504_v1.0.9_manifest.json"):
        shutil.copy(REPO / "config" / f, tmp_path / "config" / f)
    for k, v in (("EEGAD_REPO_ROOT", tmp_path), ("EEGAD_BRAINLAT_DIR", tmp_path / "b"),
                 ("EEGAD_DS004504_DIR", tmp_path / "d"), ("EEGAD_OUTPUTS_DIR", tmp_path / "outputs")):
        monkeypatch.setenv(k, str(v))
    return tmp_path


def test_post_hoc_plan_amends_the_frozen_plan_and_is_frozen(repo_with_posthoc):
    ctx = load_context("00", require_frozen=False)
    ph, h, t1 = freeze_post_hoc_plan(ctx)
    assert ph["amends"]["plan_hash"] == ctx.plan_hash
    assert freeze_post_hoc_plan(ctx)[2] == t1                         # second run: same freeze
    f = repo_with_posthoc / "config" / "post_hoc_plan.json"
    plan = json.loads(f.read_text(encoding="utf-8"))
    plan["metrics"]["threshold"] = 0.4
    f.write_text(json.dumps(plan), encoding="utf-8")
    with pytest.raises(RuntimeError, match="changed after it was frozen"):
        freeze_post_hoc_plan(ctx)
    ph2, h2, _ = freeze_post_hoc_plan(ctx, POST_HOC_PLAN_2)            # amendment 2 is frozen in its own file
    assert ph2["amendment"] == 2 and h2 != h
    assert (repo_with_posthoc / "outputs" / "post_hoc" / "post_hoc_plan_2_frozen.json").exists()


def test_article_tables_without_outputs(tmp_path):
    blocks = build_blocks(tmp_path, {"plan_hash": "x", "mode": "quick"})
    kinds = [k for k, _ in blocks]
    assert "warning" in kinds and kinds.count("legend") == 12
    labels = [v["label"] for k, v in blocks if k == "table"]
    assert labels == ["Table 1", "Table S12"]          # related work and validity criteria need no outputs
    html = write_html(blocks, tmp_path / "t.html").read_text(encoding="utf-8")
    assert "Table S12" in html and "C7" in html and "This study" in html
    pytest.importorskip("docx")
    assert write_docx(blocks, tmp_path / "t.docx").stat().st_size > 5000


def test_feature_definitions_follow_the_plan():
    plan = json.loads((REPO / "config" / "analysis_plan.json").read_text(encoding="utf-8"))
    t = feature_definitions(plan)
    assert len(t) == len(plan["features"]["all"]) == 11
    assert t["band"].iloc[0] == "4-8 Hz" and "171 channel pairs" in t["definition"].iloc[-1]


def test_hyperparameter_grid_changes_one_setting_at_a_time():
    from types import SimpleNamespace

    from eegad.hyperparams import configurations
    plan = json.loads((REPO / "config" / "analysis_plan.json").read_text(encoding="utf-8"))
    grid = json.loads((REPO / "config" / "post_hoc_plan_2.json").read_text(encoding="utf-8"))
    grid = grid["A_hyperparameter_sensitivity"]["grid"]
    for trees in (300, 40):                                            # full and quick mode
        ctx = SimpleNamespace(plan=plan, rt={"rf_trees": trees})
        cfg = configurations(ctx, grid)
        frozen = cfg[0]["params"]
        assert cfg[0]["frozen"] and sum(c["frozen"] for c in cfg) == 1
        assert frozen == {**{k: plan["classifiers"]["RandomForest"][k] for k in frozen}, "n_estimators": trees}
        for c in cfg[1:]:
            changed = [k for k in frozen if c["params"][k] != frozen[k]]
            assert len(changed) == 1, c["label"]
        assert cfg[-1]["params"]["n_estimators"] == round(1000 * trees / 300)


def test_relative_spectrum_and_group_bands():
    from eegad.spectra import group_spectra, relative_spectrum
    sf, rng = 250.0, np.random.default_rng(0)
    t = np.arange(int(4 * sf)) / sf
    ep = np.sin(2 * np.pi * 10 * t)[None, None, :] + 0.1 * rng.standard_normal((5, 19, len(t)))
    f, spec, n = relative_spectrum(ep, sf, 1.0)
    assert n == 5 and f[np.argmax(spec)] == 10.0
    integrate = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
    assert np.isclose(integrate(spec, f), 1.0)
    df = pd.DataFrame([{"subject_id": f"s{i}", "n_epochs": 5, **{f"{x:g}": v * (1 + 0.1 * i) for x, v in zip(f, spec)}}
                       for i in range(6)])
    subj = pd.DataFrame({"subject_id": [f"s{i}" for i in range(6)], "y": [1, 1, 1, 0, 0, 0]})
    g = group_spectra(df, subj, "test", n_boot=100, seed=1)
    assert set(g.group) == {"AD", "HC"} and (g["lo"] <= g["mean"]).all() and (g["mean"] <= g["hi"]).all()
    assert "subject_id" not in g.columns                                 # group level only


def test_new_figures_draw(tmp_path):
    import matplotlib
    matplotlib.use("Agg")
    from eegad.article_figures import fig_hyperparameters, fig_montage_error, fig_spectra
    rng = np.random.default_rng(0)
    ch = ["Fp1", "Fp2", "Cz", "O1", "O2"]
    U = rng.normal(size=(30, 3))
    U[:, 2] = np.abs(U[:, 2])
    U /= np.linalg.norm(U, axis=1, keepdims=True)
    elec = pd.DataFrame({"electrode": [f"A{i}" for i in range(30)], "x": U[:, 0], "y": U[:, 1], "z": U[:, 2],
                         "assigned_to": ch + [""] * 25})
    targ = pd.DataFrame({"channel_1020": ch, "x": U[:5, 0], "y": U[:5, 1], "z": U[:5, 2], "angle_deg": [1, 2, 3, 4, 5.5]})
    fig_montage_error(elec, targ, {"median_neighbour_angle_deg": 36.0}).savefig(tmp_path / "s1.png")
    f = np.arange(1, 46, dtype=float)
    g = pd.concat([pd.DataFrame({"cohort": c, "group": gr, "n": 10, "frequency_hz": f, "mean": 1 / f, "lo": 0.8 / f,
                                 "hi": 1.2 / f}) for c in ("BrainLat", "Greek") for gr in ("AD", "HC")])
    fig_spectra(g).savefig(tmp_path / "s2.png")
    w = {"auc": [0.7, 0.6, 0.8], "OR": 2.0, "OR_lo": 1.0, "OR_hi": np.inf, "LR": 3.0}
    res = [{"label": f"cfg {i}", "frozen": i == 0, "BrainLat": w, "Greek": w,
            "RQ3a": {"auc": [0.8, 0.7, 0.9], "calibration_slope": 1.0},
            "RQ3b": {"auc": [0.7, 0.6, 0.8], "calibration_slope": 1.0}} for i in range(4)]
    fig_hyperparameters(res).savefig(tmp_path / "s5.png")
    assert all((tmp_path / f).stat().st_size > 5000 for f in ("s1.png", "s2.png", "s5.png"))
