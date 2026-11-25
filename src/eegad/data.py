"""Data access: ds004504 download and SHA-256 verification, BrainLat folder checks and the per-subject tables of
both cohorts. Nothing here reads EEG signals (see features.py)."""
from __future__ import annotations

import datetime
import hashlib
import json
import os
import re
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

# ================================================================================================================
# ds004504 (OpenNeuro, version 1.0.9)
# ================================================================================================================


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def fetch_ds004504(ds_dir, record_file, manifest_file, download=True, base_url=None, install_openneuro_py=True):
    """Make sure ds004504 v1.0.9 is in ds_dir: check every file of the manifest by size and SHA-256 and download only
    what is missing or different (OpenNeuro public bucket over HTTPS; openneuro-py as fallback).

    Returns a dict with the outcome (also written to record_file). Raises if participants.tsv cannot be obtained.
    """
    ds_dir = Path(ds_dir)
    manifest = json.loads(Path(manifest_file).read_text(encoding="utf-8"))
    version, doi = manifest.get("version", "1.0.9"), manifest.get("doi", "")
    files = [tuple(f) for f in manifest["files"]]
    essential = set(manifest.get("essential", ["participants.tsv"]))
    base_url = base_url or "https://s3.amazonaws.com/openneuro.org/ds004504"
    record_file = Path(record_file)
    rec = json.loads(record_file.read_text(encoding="utf-8")) if record_file.exists() else {}
    rec.setdefault("verified", {})
    warnings = []

    def ok(rel, size, sha):
        p = ds_dir / rel
        if not p.exists() or p.stat().st_size != size:
            return False
        mark = [p.stat().st_size, p.stat().st_mtime_ns, sha]
        if rec["verified"].get(rel) == mark:          # already hashed and unchanged since
            return True
        good = sha256_file(p) == sha
        if good:
            rec["verified"][rel] = mark
        return good

    def download_direct(rel, size, sha):
        url = f"{base_url}/{urllib.parse.quote(rel)}"
        dest = ds_dir / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_name(dest.name + ".part")
        last = None
        for attempt in range(4):
            try:
                h = hashlib.sha256()
                req = urllib.request.Request(url, headers={"User-Agent": "eeg-ad-demographics"})
                with urllib.request.urlopen(req, timeout=120) as r, open(tmp, "wb") as fh:
                    for block in iter(lambda: r.read(1 << 20), b""):
                        fh.write(block)
                        h.update(block)
                if tmp.stat().st_size != size:
                    raise IOError(f"size {tmp.stat().st_size} instead of {size}")
                if h.hexdigest() != sha:
                    raise IOError(f"SHA-256 differs from version {version}")
                os.replace(tmp, dest)
                return
            except urllib.error.HTTPError as exc:
                last = exc
                if exc.code in (403, 404):
                    break
            except Exception as exc:
                last = exc
            time.sleep(2 ** attempt)
        if tmp.exists():
            tmp.unlink()
        raise RuntimeError(f"{rel}: {last}")

    def download_openneuro_py(rels):
        try:
            import openneuro
        except ImportError:
            if not install_openneuro_py:
                raise
            print("Installing openneuro-py (pip) ...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "openneuro-py"])
            import openneuro
        openneuro.download(dataset="ds004504", tag=version, target_dir=str(ds_dir), include=list(rels))

    ds_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    missing = [(r, s, h) for r, s, h in files if not ok(r, s, h)]
    print(f"ds004504 v{version}: {len(files) - len(missing)} of {len(files)} files already present and verified "
          f"({time.time() - t0:.0f} s)")
    method = "none (already complete)" if not missing else "none (download disabled)"
    if missing and download:
        gb = sum(s for _, s, _ in missing) / 1e9
        print(f"Downloading {len(missing)} file(s), {gb:.2f} GB, from {base_url} ...")
        method, failed, done, t0 = "OpenNeuro public bucket (HTTPS)", [], 0, time.time()
        for k, (r, s, h) in enumerate(missing):
            try:
                download_direct(r, s, h)
                done += s
            except Exception as exc:
                failed.append((r, s, h))
                print(f"  failed: {exc}")
                if len(failed) >= 3 and done == 0:      # nothing works: do not insist file by file
                    failed += [x for x in missing[k + 1:]]
                    break
            if (k + 1) % 10 == 0 or k + 1 == len(missing):
                dt = time.time() - t0
                print(f"  {k + 1}/{len(missing)} | {done / 1e9:.2f} GB | {done / 1e6 / max(dt, 1e-9):.1f} MB/s")
        if failed:
            print(f"{len(failed)} file(s) did not come from the bucket; trying openneuro-py (version {version}) ...")
            try:
                download_openneuro_py([r for r, _, _ in failed])
                method += " + openneuro-py"
            except Exception as exc:
                print(f"  openneuro-py failed too: {str(exc)[:300]}")
        missing = [(r, s, h) for r, s, h in files if not ok(r, s, h)]
    elif missing:
        print("Download disabled (download_ds004504: false): only checked what is already there.")

    bad_data = [r for r, _, _ in missing if r in essential or r.endswith(".set")]
    bad_meta = [r for r, _, _ in missing if r not in bad_data]
    if bad_meta:
        warnings.append(f"ds004504: {len(bad_meta)} metadata file(s) missing or from another version: {bad_meta}")
    if bad_data:
        warnings.append(f"ds004504: {len(bad_data)} data file(s) missing or different from version {version} "
                        f"(first: {bad_data[:5]}). The affected participants may be left out.")
    if not (ds_dir / "participants.tsv").exists():
        raise FileNotFoundError(
            f"Could not obtain {ds_dir / 'participants.tsv'}. Download ds004504 v{version} manually "
            "(https://openneuro.org/datasets/ds004504/versions/1.0.9 -> Download, or "
            "'aws s3 sync --no-sign-request s3://openneuro.org/ds004504 <folder>') and set ds004504_dir.")
    doi_local = ""
    if (ds_dir / "dataset_description.json").exists():
        try:
            doi_local = json.loads((ds_dir / "dataset_description.json").read_text(encoding="utf-8")).get("DatasetDOI", "")
        except Exception:
            pass
    rec.update({"version": version, "expected_doi": doi, "doi_in_folder": doi_local, "method": method,
                "checked_at": datetime.datetime.now().isoformat(timespec="seconds"),
                "files_ok": len(files) - len(missing), "files_in_manifest": len(files), "warnings": warnings})
    record_file.write_text(json.dumps(rec, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"DOI in folder: {doi_local or 'n/a'} | method: {method}")
    return rec


# ================================================================================================================
# BrainLat
# ================================================================================================================


def find_brainlat_root(given, plan):
    """The folder that contains 1_AD and 5_HC. If ``given`` is a parent of it (e.g. the root of the downloaded
    repository), the single matching subfolder is used."""
    given = Path(given)
    ad, hc = plan["cohorts"]["brainlat"]["folders"]["AD"], plan["cohorts"]["brainlat"]["folders"]["HC"]
    if (given / ad).is_dir() and (given / hc).is_dir():
        return given
    if not given.exists():
        raise FileNotFoundError(f"BrainLat folder not found: {given}. Set brainlat_dir in config/paths.yaml "
                                f"(the folder that contains {ad} and {hc}). See README, 'Data access'.")
    cands = sorted({p.parent for p in given.rglob(ad) if p.is_dir() and (p.parent / hc).is_dir()})
    if len(cands) == 1:
        print(f"Using {cands[0]} (the folder inside brainlat_dir that contains {ad} and {hc}). "
              "You can set brainlat_dir to it in config/paths.yaml.")
        return cands[0]
    raise FileNotFoundError(f"{given} does not contain {ad} and {hc}" +
                            (f"; candidates found: {[str(c) for c in cands]}" if cands else "") +
                            ". Set brainlat_dir to the folder that contains them.")


def extract_subject_id(fpath):
    fpath = Path(fpath)
    m = re.search(r"(sub-\d+)", fpath.stem, flags=re.IGNORECASE)
    if m:
        return m.group(1)
    m = re.search(r"sub[-_]?(\d+)", fpath.stem, flags=re.IGNORECASE)
    return f"sub-{m.group(1)}" if m else fpath.stem


def infer_country(fpath):
    parts = {p.upper() for p in Path(fpath).parts}
    return "Argentina" if "AR" in parts else "Chile" if "CL" in parts else "Other"


def find_csv(folder, name):
    folder = Path(folder)
    if (folder / name).exists():
        return folder / name
    cands = sorted(folder.rglob(name)) if folder.exists() else []
    return cands[0] if cands else None


def find_cognitive_csv(folder):
    if folder is None or not Path(folder).exists():
        return None
    for f in sorted(Path(folder).rglob("*.csv")):
        try:
            cols = [c.strip().lower() for c in pd.read_csv(f, nrows=0).columns]
        except Exception:
            continue
        if "moca_total" in cols or "ifs_total_score" in cols:
            return f
    return None


def read_cognitive(f):
    d = pd.read_csv(f)
    d.columns = d.columns.str.strip().str.lower()
    for id_col in ("id eeg", "subject_id", "id"):
        if id_col in d.columns:
            d = d.rename(columns={id_col: "subject_id"})
            break
    d["subject_id"] = d["subject_id"].astype(str).str.strip()
    return d[["subject_id"] + [c for c in ("moca_total", "ifs_total_score") if c in d.columns]]


def check_brainlat(root, plan):
    """Counts of .set files (and of .set files whose .fdt is missing) per group and site; demographic files found.
    Reads no EEG data and no labels beyond the folder names."""
    rows = []
    for grp, folder in plan["cohorts"]["brainlat"]["folders"].items():
        for f in sorted((Path(root) / folder).rglob("*.set")):
            rows.append({"group": grp, "site": infer_country(f), "fdt_missing": not f.with_suffix(".fdt").exists()})
    if not rows:
        raise FileNotFoundError(f"No .set files under {root}. Check brainlat_dir in config/paths.yaml.")
    tab = (pd.DataFrame(rows).groupby(["group", "site"])
           .agg(set_files=("fdt_missing", "size"), fdt_missing=("fdt_missing", "sum")).reset_index())
    demo = {g: find_csv(Path(root) / plan["cohorts"]["brainlat"]["folders"][g], f"demographics_{g.lower()}_eeg_data.csv")
            for g in ("AD", "HC")}
    if any(v is None for v in demo.values()):
        raise FileNotFoundError(f"BrainLat demographic files not found: {demo}.")
    return tab, demo


def brainlat_files(root, plan, log=None):
    """One row per .set file: subject_id, y, site, file. Subjects in both folders raise; duplicates keep the first."""
    rows = []
    for grp, folder in plan["cohorts"]["brainlat"]["folders"].items():
        for f in sorted((Path(root) / folder).rglob("*.set")):
            rows.append({"subject_id": extract_subject_id(f), "y": plan["labels"][grp], "site": infer_country(f),
                         "file": str(f)})
    files = pd.DataFrame(rows)
    inconsistent = files.groupby("subject_id")["y"].nunique()
    if (inconsistent > 1).any():
        raise ValueError(f"Subjects in both folders: {inconsistent[inconsistent > 1].index.tolist()}")
    dup = files[files.subject_id.duplicated(keep="first")]
    if len(dup) and log is not None:
        log.alert(f"{len(dup)} repeated .set file(s) for the same subject; the first was used: {dup.subject_id.tolist()[:5]}")
    return files.drop_duplicates("subject_id", keep="first") if len(dup) else files


def brainlat_metadata(root, plan):
    """Demographics (both folders) and cognition (MoCA, IFS)."""
    folders = plan["cohorts"]["brainlat"]["folders"]
    dir_ad, dir_hc = Path(root) / folders["AD"], Path(root) / folders["HC"]
    f_demo = {"AD": find_csv(dir_ad, "demographics_ad_eeg_data.csv"), "HC": find_csv(dir_hc, "demographics_hc_eeg_data.csv")}
    if any(v is None for v in f_demo.values()):
        raise FileNotFoundError(f"Demographics not found: {f_demo}. Check brainlat_dir.")
    demo_all = pd.concat([pd.read_csv(f).assign(folder_group=g) for g, f in f_demo.items()], ignore_index=True)
    demo_all = demo_all.rename(columns={"id EEG": "subject_id"})
    demo_all["subject_id"] = demo_all["subject_id"].astype(str).str.strip()
    cog_files = [f for f in (find_cognitive_csv(dir_ad), find_cognitive_csv(dir_hc)) if f is not None]
    df_cog = (pd.concat([read_cognitive(f) for f in cog_files], ignore_index=True).drop_duplicates("subject_id")
              if cog_files else pd.DataFrame(columns=["subject_id", "moca_total", "ifs_total_score"]))
    for c in ("moca_total", "ifs_total_score"):
        if c not in df_cog.columns:
            df_cog[c] = np.nan
    return demo_all, df_cog, cog_files


def brainlat_subjects(files, demo_all, df_cog, plan):
    """Per-subject BrainLat table (all .set files, before checking whether the EEG is analysable)."""
    s = (files.merge(demo_all[["subject_id", "sex", "Age", "years_education"]].drop_duplicates("subject_id"),
                     on="subject_id", how="left")
              .merge(df_cog[["subject_id", "moca_total", "ifs_total_score"]], on="subject_id", how="left"))
    for c in ("sex", "Age", "years_education", "moca_total", "ifs_total_score"):
        s[c] = pd.to_numeric(s[c], errors="coerce")
    fem = plan["cohorts"]["brainlat"]["female_sex_value"]
    s["sex_fem"] = np.where(s["sex"].isna(), np.nan, (s["sex"] == fem).astype(float))
    s["site_cl"] = (s["site"] == "Chile").astype(float)
    s["cohort"] = "BrainLat"
    return s.sort_values("subject_id").reset_index(drop=True)


# ================================================================================================================
# Greek cohort (ds004504)
# ================================================================================================================


def greek_participants_raw(ds_dir):
    """participants.tsv as distributed (all groups, including frontotemporal dementia)."""
    part = pd.read_csv(Path(ds_dir) / "participants.tsv", sep="\t")
    part.columns = [c.strip().lower() for c in part.columns]
    missing = {"participant_id", "group", "age", "gender", "mmse"} - set(part.columns)
    if missing:
        raise ValueError(f"participants.tsv lacks the columns {missing}: is this ds004504?")
    part["group"] = part["group"].astype(str).str.strip().str.upper()
    return part


def greek_participants(part_raw, ds_dir, plan):
    """AD (A) and controls (C) with their .set file (derivatives first). Returns (participants, has_education)."""
    groups = plan["cohorts"]["greek"]["groups"]
    part = part_raw[part_raw["group"].isin(groups)].copy()
    part["y"] = part["group"].map(groups).astype(int)
    part["sex_fem"] = part["gender"].astype(str).str.strip().str.upper().str[0].map({"F": 1.0, "M": 0.0})
    part["Age"] = pd.to_numeric(part["age"], errors="coerce")
    part["mmse"] = pd.to_numeric(part["mmse"], errors="coerce")
    has_education = any("educ" in c for c in part.columns)

    def find_set(pid):
        bases = ([Path(ds_dir) / "derivatives" / pid / "eeg"] if plan["cohorts"]["greek"]["use_derivatives"] else []) \
            + [Path(ds_dir) / pid / "eeg"]
        for b in bases:
            cands = sorted(b.glob("*.set")) if b.exists() else []
            if cands:
                return str(cands[0])
        return None

    part["file"] = [find_set(p) for p in part["participant_id"]]
    part["origin"] = ["missing" if not isinstance(f, str) else "derivatives" if "derivatives" in Path(f).parts
                      else "raw" for f in part["file"]]
    return part, has_education


def greek_subjects(part):
    s = (part.rename(columns={"participant_id": "subject_id"})
             [["subject_id", "y", "Age", "sex_fem", "mmse", "file", "origin"]]
             .assign(cohort="Greek").sort_values("subject_id").reset_index(drop=True))
    s["subject_id"] = s["subject_id"].astype(str).str.strip()
    return s


# ================================================================================================================
# Derived tables shared between notebooks (exact float round-trip)
# ================================================================================================================


def save_table(df, path):
    df.to_csv(path, index=False)


def load_table(path, **kw):
    """Read a derived CSV with exact float round-trip and subject ids as strings."""
    dtypes = {c: str for c in ("subject_id", "ad_id", "hc_id")}
    return pd.read_csv(path, dtype=dtypes, float_precision="round_trip", **kw)
