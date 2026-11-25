"""Paths, analysis plan, run settings and the frozen-plan check shared by all notebooks."""
from __future__ import annotations

import datetime
import hashlib
import json
import os
import platform
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from .reporting import ResultLog

PLAN_FILE = Path("config") / "analysis_plan.json"
LOCAL_FILE = Path("config") / "paths.yaml"
EXAMPLE_FILE = Path("config") / "paths.example.yaml"
MANIFEST_FILE = Path("config") / "ds004504_v1.0.9_manifest.json"
OUTPUT_SUBFOLDERS = ("tables", "figures", "cache", "features", "models", "derived", "predictions", "results")

# Settings used only to test the code quickly (EEGAD_QUICK=1 or quick_test_mode: true). Never for results.
QUICK_RUNTIME = {"rf_trees": 40, "xgb_trees": 40, "n_boot": 200, "n_tiebreaks": 20, "max_epochs": 20, "null_B": 9}


def find_repo_root(start: Path | None = None) -> Path:
    """Repository root: EEGAD_REPO_ROOT, else the first parent of the working directory that has the plan file."""
    if os.environ.get("EEGAD_REPO_ROOT"):
        return Path(os.environ["EEGAD_REPO_ROOT"]).resolve()
    for base in [Path(start or Path.cwd()).resolve(), Path(__file__).resolve()]:
        for p in [base, *base.parents]:
            if (p / PLAN_FILE).exists():
                return p
    raise FileNotFoundError("Repository root not found (no config/analysis_plan.json above the working directory). "
                            "Open the notebooks from the repository, or set EEGAD_REPO_ROOT.")


def plan_hash(plan: dict) -> str:
    return hashlib.sha256(json.dumps(plan, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()[:16]


def _env_flag(name: str, default: bool) -> bool:
    v = os.environ.get(name)
    return default if v is None else v.strip().lower() in ("1", "true", "yes", "on")


def load_local_settings(repo: Path) -> dict:
    """config/paths.yaml (machine-specific, not in git); environment variables override every entry."""
    f = repo / LOCAL_FILE
    cfg = {}
    if f.exists():
        cfg = yaml.safe_load(f.read_text(encoding="utf-8")) or {}
    elif not (os.environ.get("EEGAD_BRAINLAT_DIR") and os.environ.get("EEGAD_DS004504_DIR")):
        raise FileNotFoundError(
            f"{f} not found. Copy {repo / EXAMPLE_FILE} to {f} and set brainlat_dir and ds004504_dir "
            "(use forward slashes, also on Windows).")

    def path(key, env, default=None):
        v = os.environ.get(env, cfg.get(key, default))
        if v in (None, ""):
            raise KeyError(f"'{key}' is missing in {f} (or set {env}).")
        p = Path(str(v)).expanduser()
        return (repo / p).resolve() if not p.is_absolute() else p

    return {
        "brainlat_dir": path("brainlat_dir", "EEGAD_BRAINLAT_DIR"),
        "ds004504_dir": path("ds004504_dir", "EEGAD_DS004504_DIR"),
        "outputs_dir": path("outputs_dir", "EEGAD_OUTPUTS_DIR", "outputs"),
        "n_jobs": int(os.environ.get("EEGAD_N_JOBS", cfg.get("n_jobs", -1))),
        "download_ds004504": _env_flag("EEGAD_DOWNLOAD", bool(cfg.get("download_ds004504", True))),
        "quick": _env_flag("EEGAD_QUICK", bool(cfg.get("quick_test_mode", False))),
        # test hooks (synthetic data and a local mirror); not needed for real runs
        "ds004504_manifest": Path(os.environ.get("EEGAD_DS004504_MANIFEST", repo / MANIFEST_FILE)),
        "ds004504_base_url": os.environ.get("EEGAD_DS004504_BASE_URL", "https://s3.amazonaws.com/openneuro.org/ds004504"),
    }


@dataclass
class Context:
    """Everything a notebook needs: paths, the analysis plan, run settings and the result log."""
    repo: Path
    plan: dict
    plan_hash: str
    brainlat_dir: Path
    ds004504_dir: Path
    out: Path
    n_jobs: int
    quick: bool
    download_ds004504: bool
    ds004504_manifest: Path
    ds004504_base_url: str
    notebook: str
    log: ResultLog
    rt: dict = field(default_factory=dict)

    # shortcuts used throughout the notebooks
    @property
    def seed(self) -> int:
        return int(self.plan["seed"])

    @property
    def tag(self) -> str:
        return "quick" if self.quick else "full"

    @property
    def features(self) -> list:
        return list(self.plan["features"]["all"])

    @property
    def spectral(self) -> list:
        return list(self.plan["features"]["spectral"])

    @property
    def channels(self) -> list:
        return list(self.plan["channels_1020"])

    @property
    def cov_brainlat(self) -> list:
        return list(self.plan["cohorts"]["brainlat"]["covariates"])

    @property
    def cov_greek(self) -> list:
        return list(self.plan["cohorts"]["greek"]["covariates"])

    @property
    def threshold(self) -> float:
        return float(self.plan["rq3"]["threshold"])

    @property
    def n_boot(self) -> int:
        return int(self.rt["n_boot"])

    def path(self, *parts) -> Path:
        return self.out.joinpath(*parts)

    # ---- frozen plan and the moment the Greek labels were first read ---------------------------------------
    @property
    def run_log_file(self) -> Path:
        return self.out / "run_log.json"

    def read_run_log(self) -> dict:
        f = self.run_log_file
        return json.loads(f.read_text(encoding="utf-8")) if f.exists() else {}

    def write_run_log(self, log: dict) -> None:
        self.run_log_file.write_text(json.dumps(log, indent=2, ensure_ascii=False), encoding="utf-8")

    def freeze_plan(self) -> dict:
        """First run: store the plan and its hash. Later runs: refuse to continue if the plan changed."""
        f = self.out / "plan_frozen.json"
        if f.exists():
            old = json.loads(f.read_text(encoding="utf-8"))
            if old["hash"] != self.plan_hash:
                changed = sorted(k for k in set(self.plan) | set(old["plan"])
                                 if json.dumps(self.plan.get(k), sort_keys=True, ensure_ascii=False)
                                 != json.dumps(old["plan"].get(k), sort_keys=True, ensure_ascii=False))
                raise RuntimeError(
                    f"The analysis plan changed after it was frozen on {old['frozen_at']} (changed keys: {changed}).\n"
                    "If the change was decided without looking at results, record date, what and why in "
                    f"'change_history' of config/analysis_plan.json, delete {f} and run notebook 00 again.")
            frozen_at = old["frozen_at"]
            print(f"Analysis plan matches the version frozen on {frozen_at} (hash {self.plan_hash}).")
        else:
            frozen_at = datetime.datetime.now().isoformat(timespec="seconds")
            f.write_text(json.dumps({"hash": self.plan_hash, "frozen_at": frozen_at, "plan": self.plan},
                                    indent=2, ensure_ascii=False), encoding="utf-8")
            print(f"Analysis plan frozen now ({frozen_at}), hash {self.plan_hash}: {f}")
        log = self.read_run_log()
        log.setdefault("plan_first_frozen_at", frozen_at)
        freezes = log.setdefault("freezes", [])
        if not any(c["hash"] == self.plan_hash for c in freezes):
            freezes.append({"hash": self.plan_hash, "frozen_at": frozen_at})
        log.setdefault("runs", []).append({"started": datetime.datetime.now().isoformat(timespec="seconds"),
                                           "quick": self.quick, "plan_hash": self.plan_hash})
        self.write_run_log(log)
        return log

    def check_frozen(self) -> None:
        f = self.out / "plan_frozen.json"
        if not f.exists():
            raise RuntimeError(f"{f} not found: run notebooks/00_setup_and_data.ipynb first.")
        old = json.loads(f.read_text(encoding="utf-8"))
        if old["hash"] != self.plan_hash:
            raise RuntimeError("config/analysis_plan.json differs from the plan frozen by notebook 00 "
                               f"({old['frozen_at']}). Undo the change, or record it in 'change_history' and rerun "
                               "everything from notebook 00.")

    def mark_greek_labels_read(self) -> dict:
        log = self.read_run_log()
        if "greek_labels_first_read" not in log:
            log["greek_labels_first_read"] = datetime.datetime.now().isoformat(timespec="seconds")
            self.write_run_log(log)
        if log["greek_labels_first_read"] < log.get("plan_first_frozen_at", ""):
            self.log.alert("The Greek labels were read BEFORE the plan freeze recorded in run_log.json.")
        return log


def load_context(notebook: str, require_frozen: bool = True) -> Context:
    """Load paths, plan and settings; make the output folders; check the frozen plan (except in notebook 00)."""
    repo = find_repo_root()
    local = load_local_settings(repo)
    plan = json.loads((repo / PLAN_FILE).read_text(encoding="utf-8"))
    out = local["outputs_dir"]
    for sub in OUTPUT_SUBFOLDERS:
        (out / sub).mkdir(parents=True, exist_ok=True)
    rt = {"rf_trees": plan["classifiers"]["RandomForest"]["n_estimators"],
          "xgb_trees": plan["classifiers"]["XGBoost"]["n_estimators"],
          "n_boot": plan["validation"]["bootstrap_n"],
          "n_tiebreaks": plan["rq2"]["brainlat_matching"]["n_random_tiebreaks"],
          "max_epochs": plan["preprocessing"]["max_epochs_per_subject"],
          "null_B": plan["rq1"]["null_bootstrap"]["B_max"]}
    if local["quick"]:
        rt.update(QUICK_RUNTIME)
    ctx = Context(repo=repo, plan=plan, plan_hash=plan_hash(plan), brainlat_dir=local["brainlat_dir"],
                  ds004504_dir=local["ds004504_dir"], out=out, n_jobs=local["n_jobs"], quick=local["quick"],
                  download_ds004504=local["download_ds004504"], ds004504_manifest=local["ds004504_manifest"],
                  ds004504_base_url=local["ds004504_base_url"], notebook=notebook,
                  log=ResultLog(out / "results"), rt=rt)
    ctx.log.start(notebook)
    if require_frozen:
        ctx.check_frozen()
    print(f"Repository: {repo}\nOutputs:    {out}\nPlan hash:  {ctx.plan_hash} | mode: "
          f"{'QUICK TEST (do not use these numbers)' if ctx.quick else 'full'} | RF trees {rt['rf_trees']} | "
          f"bootstrap {rt['n_boot']} | null-bootstrap B {rt['null_B']}")
    if ctx.quick:
        print("!" * 78 + "\nQUICK TEST MODE: only to check that the code runs. Do NOT use these numbers.\n" + "!" * 78)
    return ctx


def library_versions() -> dict:
    import matplotlib
    import numpy
    import pandas
    import scipy
    import sklearn
    v = {"python": platform.python_version(), "system": platform.platform(), "numpy": numpy.__version__,
         "pandas": pandas.__version__, "scipy": scipy.__version__, "scikit-learn": sklearn.__version__,
         "matplotlib": matplotlib.__version__}
    try:
        import mne
        v["mne"] = mne.__version__
    except ImportError:
        v["mne"] = "MISSING"
    try:
        import xgboost
        v["xgboost"] = xgboost.__version__
    except ImportError:
        v["xgboost"] = "MISSING"
    return v
