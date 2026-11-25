"""Synthetic data with the folder structure of BrainLat and ds004504, to test the code without the real datasets.

The signals are simulated (1/f background plus theta, alpha and beta sources mixed over the channels); "AD" subjects
have more theta and slower, weaker alpha. Nothing here comes from real participants.

    python scripts/make_synthetic.py <destination> [--small] [--fdt]

--small fewer and shorter recordings (used by the automated tests)
--fdt   store the BrainLat files as .set + .fdt (as in the real data) and delete one .fdt (an unreadable control,
         like the 14 Chilean controls without .fdt in the real data)

Writes <destination>/brainlat, <destination>/ds004504 and <destination>/ds004504_manifest.json (sizes and SHA-256 of
the synthetic ds004504, so notebook 00 can verify it). Use them with the environment variables printed at the end.
"""
import argparse
import hashlib
import json
from pathlib import Path

import mne
import numpy as np
import pandas as pd
import scipy.io as sio

mne.set_log_level("ERROR")
ap = argparse.ArgumentParser()
ap.add_argument("dest")
ap.add_argument("--small", "-small", action="store_true")
ap.add_argument("--fdt", "-fdt", action="store_true")
args = ap.parse_args()
dest = Path(args.dest).resolve()
rng = np.random.default_rng(7)
if args.small:
    N_AD_BL, N_HC_BL, N_A_GR, N_C_GR, N_F_GR, DUR_BL, DUR_GR = 10, 9, 9, 8, 3, 20, 24
else:
    N_AD_BL, N_HC_BL, N_A_GR, N_C_GR, N_F_GR, DUR_BL, DUR_GR = 16, 14, 14, 12, 4, 36, 40


def pink(n_ch, n, sf, expo=1.0):
    X = rng.normal(size=(n_ch, n))
    F = np.fft.rfft(X, axis=1)
    f = np.fft.rfftfreq(n, 1 / sf)
    f[0] = f[1]
    F /= f[None, :] ** (expo / 2)
    return np.fft.irfft(F, n=n, axis=1)


def signal(n_ch, sf, dur, sev, age, scale, expo, mix_seed):
    """6 latent sources (theta, alpha, beta + 1/f) mixed over the channels + channel noise."""
    n = int(sf * dur)
    t = np.arange(n) / sf
    fa = 10.5 - 0.03 * (age - 70) - 0.8 * sev
    S = []
    for _ in range(6):
        ph = rng.uniform(0, 2 * np.pi)
        drift = np.cumsum(rng.normal(0, 0.02, n))
        theta = (1.0 + 0.7 * sev) * np.sin(2 * np.pi * 6.0 * t + ph + drift)
        alpha = (1.6 - 0.6 * sev) * np.sin(2 * np.pi * fa * t + ph + 0.5 * drift)
        beta = 0.4 * np.sin(2 * np.pi * 20.0 * t + ph + 2 * drift)
        S.append(theta + alpha + beta)
    S = np.array(S) + 3 * pink(6, n, sf, expo)
    M = np.random.default_rng(mix_seed).normal(size=(n_ch, 6))
    X = M @ S + (1.5 + 0.3 * sev) * pink(n_ch, n, sf, expo) + 0.3 * rng.normal(size=(n_ch, n))
    return scale * 1e-6 * X


def to_set_fdt(f):
    """Move the data of an EEGLAB .set into a separate .fdt file (format of the BrainLat files)."""
    m = sio.loadmat(str(f), appendmat=False)
    if "EEG" in m:
        eeg = m["EEG"][0, 0]
        data = np.asarray(eeg.data, dtype="<f4")
        eeg.data = np.array([f.with_suffix(".fdt").name])
        m["EEG"][0, 0] = eeg
    else:
        data = np.asarray(m["data"], dtype="<f4")
        m["data"] = f.with_suffix(".fdt").name
    data.T.tofile(str(f.with_suffix(".fdt")))   # channel varies fastest (EEGLAB order)
    m = {k: v for k, v in m.items() if not k.startswith("__")}
    sio.savemat(str(f), m, appendmat=False, format="5")


# ---------------------------------------------------------------- BrainLat (BioSemi 128, 512 Hz)
bl = dest / "brainlat"
mt128 = mne.channels.make_standard_montage("biosemi128")
rows = []
for grp, folder, base_id, n in (("AD", "1_AD", 30001, N_AD_BL), ("HC", "5_HC", 10001, N_HC_BL)):
    for k in range(n):
        sid = f"sub-{base_id + k}"
        site = "AR" if k % 2 == 0 else "CL"
        ad = grp == "AD"
        age = float(np.clip(rng.normal(74 if ad else 68, 7), 52, 90).round())
        edu = int(np.clip(rng.normal(9 if ad else 13, 4), 1, 22))
        sev = float(np.clip(rng.normal(1.0 if ad else 0.0, 0.6), -0.5, 2.0)) + 0.03 * (12 - edu)
        moca = float(np.clip(round(26 - 6 * sev + rng.normal(0, 2)), 5, 30))
        X = signal(128, 512.0, DUR_BL, sev, age, 8.0, 1.0, 11)
        raw = mne.io.RawArray(X, mne.create_info(mt128.ch_names, 512.0, "eeg"))
        raw.set_montage(mt128)
        d = bl / folder / site
        d.mkdir(parents=True, exist_ok=True)
        f = d / f"{sid}_task-rest_eeg.set"
        mne.export.export_raw(str(f), raw, fmt="eeglab", overwrite=True)
        if args.fdt:
            to_set_fdt(f)
        rows.append({"path": str(f), "id EEG": sid, "diagnosis": grp, "sex": int(rng.random() < (0.35 if ad else 0.45)),
                     "Age": age, "years_education": edu, "moca_total": moca,
                     "ifs_total_score": float(np.clip(round(22 - 5 * sev + rng.normal(0, 2)), 2, 30)), "grp": grp})
# an unreadable control with a row in the demographics
bad = bl / "5_HC" / "CL" / "sub-19999_task-rest_eeg.set"
if args.fdt:      # like the real data: a .set whose .fdt is missing
    raw = mne.io.RawArray(signal(128, 512.0, 8, 0.0, 70, 8.0, 1.0, 11), mne.create_info(mt128.ch_names, 512.0, "eeg"))
    mne.export.export_raw(str(bad), raw, fmt="eeglab", overwrite=True)
    to_set_fdt(bad)
    bad.with_suffix(".fdt").unlink()
else:
    bad.write_bytes(b"not a set file")
rows.append({"path": str(bad), "id EEG": "sub-19999", "diagnosis": "HC", "sex": 1, "Age": 70.0,
             "years_education": 12, "moca_total": 27.0, "ifs_total_score": 24.0, "grp": "HC"})
D = pd.DataFrame(rows)
D.loc[D.index[2], "years_education"] = np.nan        # one missing education
D.loc[D.index[3], "moca_total"] = np.nan             # one missing MoCA
for grp, folder, tag in (("AD", "1_AD", "ad"), ("HC", "5_HC", "hc")):
    s = D[D.grp == grp]
    s[["path", "id EEG", "diagnosis", "sex", "Age", "years_education"]].assign(laterality="R").to_csv(
        bl / folder / f"demographics_{tag}_eeg_data.csv", index=False)
    s[["id EEG", "moca_total", "ifs_total_score"]].to_csv(bl / folder / f"cognition_{tag}_eeg_data.csv", index=False)

# ---------------------------------------------------------------- ds004504 (19 channels, 500 Hz)
ds = dest / "ds004504"
CH_DS = ["Fp1", "Fp2", "F3", "F4", "C3", "C4", "P3", "P4", "O1", "O2", "F7", "F8", "T3", "T4", "T5", "T6", "Fz", "Cz", "Pz"]
groups = ["A"] * N_A_GR + ["C"] * N_C_GR + ["F"] * N_F_GR
rng.shuffle(groups)
part = []
for i, g in enumerate(groups, start=1):
    pid = f"sub-{i:03d}"
    ad = g == "A"
    age = float(np.clip(rng.normal(66 if ad else 67 if g == "C" else 63, 7), 45, 85).round())
    fem = rng.random() < (0.66 if ad else 0.38 if g == "C" else 0.4)
    sev = float(np.clip(rng.normal(1.0 if ad else 0.0 if g == "C" else 0.6, 0.6), -0.5, 2.0))
    mmse = 30 if g == "C" else int(np.clip(round(26 - 5 * sev + rng.normal(0, 2)), 4, 29))
    part.append({"participant_id": pid, "Gender": "F" if fem else "M", "Age": int(age), "Group": g, "MMSE": mmse})
    names = list(CH_DS)
    if i == 2:   # one file with the 10-10 names (T7/T8/P7/P8) to test the aliases
        names = [{"T3": "T7", "T4": "T8", "T5": "P7", "T6": "P8"}.get(c, c) for c in names]
    X = signal(19, 500.0, DUR_GR, sev, age, 20.0, 1.3, 23)
    raw = mne.io.RawArray(X, mne.create_info(names, 500.0, "eeg"))
    if i == 5:   # 'boundary' marks, as in files with segments cut in EEGLAB
        raw.set_annotations(mne.Annotations(onset=[5.0, 17.5], duration=[0.0, 0.0], description=["boundary"] * 2))
    targets = [ds / "derivatives" / pid / "eeg"]
    if i <= 3:
        targets.append(ds / pid / "eeg")       # some also with the raw file (derivatives must take precedence)
    if i == 4:
        targets = [ds / pid / "eeg"]           # one only with the raw file
    for d in targets:
        d.mkdir(parents=True, exist_ok=True)
        mne.export.export_raw(str(d / f"{pid}_task-eyesclosed_eeg.set"), raw, fmt="eeglab", overwrite=True)
pd.DataFrame(part).to_csv(ds / "participants.tsv", sep="\t", index=False)
(ds / "dataset_description.json").write_text('{"Name": "synthetic ds004504-like"}', encoding="utf-8")


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


files = [["participants.tsv", (ds / "participants.tsv").stat().st_size, sha(ds / "participants.tsv")]]
files += [[str(p.relative_to(ds)).replace("\\", "/"), p.stat().st_size, sha(p)]
          for p in sorted((ds / "derivatives").rglob("*.set"))]
(dest / "ds004504_manifest.json").write_text(json.dumps(
    {"dataset": "ds004504 (synthetic)", "version": "1.0.9", "doi": "synthetic", "essential": ["participants.tsv"],
     "files": files}, indent=1), encoding="utf-8")
print(f"Synthetic BrainLat: {N_AD_BL} AD + {N_HC_BL} HC (+1 unreadable) in {bl}")
print(f"Synthetic ds004504: {pd.DataFrame(part).Group.value_counts().to_dict()} in {ds}")
print("\nTo use it (bash):")
print(f"  export EEGAD_BRAINLAT_DIR={bl} EEGAD_DS004504_DIR={ds} "
      f"EEGAD_DS004504_MANIFEST={dest / 'ds004504_manifest.json'} EEGAD_DOWNLOAD=0 EEGAD_QUICK=1")
