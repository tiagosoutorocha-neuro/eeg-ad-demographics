"""End-to-end run of notebooks 00-12 on small synthetic data, in quick test mode (a few minutes)."""
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]


@pytest.mark.slow
def test_whole_pipeline_on_synthetic_data(tmp_path):
    data = tmp_path / "synthetic"
    subprocess.run([sys.executable, str(REPO / "scripts" / "make_synthetic.py"), str(data), "--small", "--fdt"],
                   check=True)
    env = {**os.environ, "EEGAD_BRAINLAT_DIR": str(data / "brainlat"), "EEGAD_DS004504_DIR": str(data / "ds004504"),
           "EEGAD_DS004504_MANIFEST": str(data / "ds004504_manifest.json"), "EEGAD_DOWNLOAD": "0",
           "EEGAD_OUTPUTS_DIR": str(tmp_path / "outputs"), "EEGAD_QUICK": "1", "MPLBACKEND": "Agg"}
    subprocess.run([sys.executable, str(REPO / "scripts" / "run_all.py"), "--quick"], check=True, env=env, cwd=REPO)
    out = tmp_path / "outputs"
    for f in ("plan_frozen.json", "run_log.json", "results_summary.md", "share.zip", "preprint_package.zip",
              "reproduction_check.csv", "tables/T2_main_results.csv", "tables/F0_participant_flow.csv",
              "figures/fig1_roc_both_cohorts.pdf", "figures/fig0_participant_flow.pdf",
              "post_hoc/post_hoc_plan_frozen.json", "post_hoc/recalibration_table.csv",
              "post_hoc/post_hoc_plan_2_frozen.json", "post_hoc/hyperparameter_table.csv",
              "descriptive/spectra_by_group.csv", "descriptive/montage_targets.csv",
              "article/Fig1_study_design.pdf", "article/Fig1_study_design.svg", "article/Fig2_late_fusion.pdf",
              "article/Fig2_late_fusion.svg", "article/Fig5_RQ1_incremental_value.pdf",
              "article/Fig6_RQ2_confounding_by_design.pdf", "article/Fig7_RQ3_transportability.pdf",
              "article/Fig8_feature_consistency.pdf", "article/FigS1_montage_error.pdf",
              "article/FigS2_group_spectra.pdf", "article/FigS3_e_value.pdf", "article/FigS4_domain_shift.pdf",
              "article/FigS5_hyperparameters.pdf", "article/supplementary_files/post_hoc_plan_2.json"):
        assert (out / f).exists(), f
    assert (out / "article" / "article_tables.docx").exists() or (out / "article" / "article_tables.html").exists()
    # the post hoc amendment leaves the pre-specified results untouched
    keys = json.loads((out / "results" / "key_results.json").read_text(encoding="utf-8"))
    assert not any(v["notebook"] in ("09", "10", "11", "12") for v in keys.values())
    log = json.loads((out / "run_log.json").read_text(encoding="utf-8"))
    assert log["greek_labels_first_read"] >= log["plan_first_frozen_at"]
    verdicts = json.loads((out / "results" / "verdicts.json").read_text(encoding="utf-8"))
    assert {"RQ1a", "RQ1b", "RQ1c", "RQ2a", "RQ2b", "RQ3a", "RQ3b"} <= set(verdicts)
