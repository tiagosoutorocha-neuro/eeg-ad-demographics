"""EEG reading, harmonised preprocessing (19 channels of the 10-20 system), the 11 epoch-level features and the
BioSemi-128 -> 10-20 channel correspondence for BrainLat."""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.signal import butter, hilbert, sosfiltfilt

try:
    import mne
    from mne.time_frequency import psd_array_welch
    mne.set_log_level("ERROR")
except ImportError as exc:  # pragma: no cover
    raise ImportError("MNE is required to read the EEGLAB .set files: pip install mne") from exc

BANDS = {"Delta": (0.5, 4.0), "Theta": (4.0, 8.0), "Alpha": (8.0, 13.0), "Beta": (13.0, 30.0), "Gamma": (30.0, 45.0)}
FMIN, FMAX = 0.5, 45.0
CONN_BANDS = {"Theta": (4.0, 8.0), "Alpha": (8.0, 13.0), "Beta": (13.0, 30.0)}
ALIASES = {"T3": ["T3", "T7"], "T4": ["T4", "T8"], "T5": ["T5", "P7"], "T6": ["T6", "P8"]}


def load_raw_safe(fpath):
    """Read an EEGLAB .set; if the paired .fdt fails, retry from a temporary copy."""
    fpath = Path(fpath)
    try:
        return mne.io.read_raw_eeglab(str(fpath), preload=True, verbose=False)
    except Exception as exc:
        msg = str(exc).lower()
        fdt = fpath.with_suffix(".fdt")
        if ("fdt" in msg or "not found" in msg) and fdt.exists():
            from tempfile import TemporaryDirectory
            with TemporaryDirectory() as tmp:
                tmp_p = Path(tmp)
                (tmp_p / fpath.name).write_bytes(fpath.read_bytes())
                (tmp_p / fdt.name).write_bytes(fdt.read_bytes())
                return mne.io.read_raw_eeglab(str(tmp_p / fpath.name), preload=True, verbose=False)
        raise


def extract_spectral_features(data, sfreq, n_fft=None):
    """Relative band power (Welch, mean over channels), theta/alpha and theta/beta ratios, spectral entropy."""
    data = np.asarray(data, dtype=float)
    n_fft = min(512, data.shape[-1]) if n_fft is None else min(int(n_fft), data.shape[-1])
    psds, freqs = psd_array_welch(data, sfreq=sfreq, fmin=FMIN, fmax=FMAX, n_fft=n_fft, verbose=False)
    psd_mean = np.mean(psds, axis=0)
    integrate = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
    band_power = {}
    for band, (flo, fhi) in BANDS.items():
        idx = (freqs >= flo) & (freqs < fhi)
        band_power[band] = float(integrate(psd_mean[idx], freqs[idx])) if np.sum(idx) >= 2 else 0.0
    total = sum(band_power.values()) + 1e-12
    feats = {f"Rel_{b}_mean": band_power[b] / total for b in BANDS}
    feats["Ratio_Theta_Alpha"] = feats["Rel_Theta_mean"] / (feats["Rel_Alpha_mean"] + 1e-12)
    feats["Ratio_Theta_Beta"] = feats["Rel_Theta_mean"] / (feats["Rel_Beta_mean"] + 1e-12)
    psd_norm = psd_mean / (np.sum(psd_mean) + 1e-12)
    feats["Spectral_Entropy"] = float(-np.sum(psd_norm * np.log2(psd_norm + 1e-12)))
    return feats


def extract_connectivity_features(data, sfreq):
    """Mean phase-locking value (theta, alpha, beta) and mean absolute broadband correlation over channel pairs."""
    data = np.asarray(data, dtype=float)
    n_ch, n_samp = data.shape
    out = {"PLV_Theta_mean": np.nan, "PLV_Alpha_mean": np.nan, "PLV_Beta_mean": np.nan, "Corr_Broadband_mean": np.nan}
    if n_ch < 2:
        return out
    iu = np.triu_indices(n_ch, k=1)
    out["Corr_Broadband_mean"] = float(np.nanmean(np.abs(np.corrcoef(data)[iu])))
    nyq, order = sfreq / 2.0, 4
    for band, (flo, fhi) in CONN_BANDS.items():
        fhi_safe = min(fhi, nyq * 0.99)
        if flo >= fhi_safe or n_samp < 3 * order * 6:
            continue
        sos = butter(order, [flo / nyq, fhi_safe / nyq], btype="band", output="sos")
        phase_exp = np.exp(1j * np.angle(hilbert(sosfiltfilt(sos, data, axis=1), axis=1)))
        plv = np.abs((phase_exp @ phase_exp.conj().T) / n_samp)
        out[f"PLV_{band}_mean"] = float(np.nanmean(plv[iu]))
    return out


def locate_channels(ch_names, mapping):
    """mapping = {name in the file: 10-20 position}; accepts T7/T8/P7/P8 in place of T3/T4/T5/T6."""
    low = {c.lower(): c for c in ch_names}
    pairs = []
    for source, target in mapping.items():
        cands = [source] + (ALIASES.get(target, []) if source == target else [])
        found = next((low[c.lower()] for c in cands if c.lower() in low), None)
        if found is None:
            raise KeyError(f"channel {source} (position {target}) not found in the file")
        pairs.append((found, target))
    return pairs


def preprocessed_epochs(fpath, mapping, plan, max_epochs=None, harmonize_band=False):
    """The harmonised preprocessing shared by the features (notebook 01) and the group spectra (notebook 11):
    19 channels, average reference, (optional) 40 Hz low-pass harmonisation, 0.5-45 Hz FIR, 4-s epochs, RMS
    normalisation. Returns (epochs array [n_epochs, 19, n_samples], sampling rate, recording info)."""
    channels = list(plan["channels_1020"])
    prep = plan["preprocessing"]
    try:
        raw = mne.io.read_raw_eeglab(str(fpath), preload=False, verbose=False)
        pairs = locate_channels(raw.ch_names, mapping)
        raw.pick([o for o, _ in pairs]).load_data()
    except KeyError:
        raise
    except Exception:
        raw = load_raw_safe(fpath)
        pairs = locate_channels(raw.ch_names, mapping)
        raw.pick([o for o, _ in pairs])
    rename = {o: a for o, a in pairs if o != a}
    if rename:
        raw.rename_channels(rename)
    raw.set_channel_types({c: "eeg" for c in raw.ch_names})
    raw.reorder_channels(channels)
    raw.set_eeg_reference("average", projection=False)
    if harmonize_band:   # same pass band as the distributed BrainLat (Butterworth 0.5-40 Hz, order 8, zero phase)
        hb = prep["greek_band_harmonization"]
        raw.filter(None, hb["h_freq"], method="iir", phase="zero", verbose=False,
                   iir_params=dict(order=hb["order"], ftype=hb["type"], output="sos"))
    raw.filter(FMIN, FMAX, fir_design="firwin", verbose=False)
    ep = mne.make_fixed_length_epochs(raw, duration=float(prep["epoch_s"]), preload=True,
                                      verbose=False).get_data().astype(np.float64)
    n_total = len(ep)
    if max_epochs:
        ep = ep[:int(max_epochs)]
    if len(ep) == 0:
        raise RuntimeError("no 4-s epoch")
    sf = float(raw.info["sfreq"])
    rms = np.sqrt(np.mean(ep ** 2))
    ep = ep / (rms if np.isfinite(rms) and rms > 1e-12 else 1.0)
    n_boundary = int(sum(str(d).strip().lower() == "boundary" for d in raw.annotations.description))
    info = {"epochs_total": n_total, "sfreq": sf, "duration_min": float(raw.times[-1]) / 60,
            "boundary_marks": n_boundary}
    return ep, sf, info


def features_19ch(fpath, mapping, sid, plan, max_epochs=None, harmonize_band=False):
    """Epoch-level features of one recording (preprocessing in preprocessed_epochs); epochs with non-finite features
    are dropped."""
    features = list(plan["features"]["all"])
    prep = plan["preprocessing"]
    ep, sf, rec = preprocessed_epochs(fpath, mapping, plan, max_epochs, harmonize_band)
    n_fft = int(round(sf / float(prep["welch_resolution_hz"])))
    rows, bad = [], 0
    for i, e in enumerate(ep):
        f = extract_spectral_features(e, sf, n_fft=n_fft)
        f.update(extract_connectivity_features(e, sf))
        if not all(np.isfinite(f[k]) for k in features):
            bad += 1
            continue
        f.update({"subject_id": sid, "epoch_id": i, "sfreq": sf})
        rows.append(f)
    info = {"subject_id": sid, "epochs_total": rec["epochs_total"], "epochs_used": len(rows), "epochs_invalid": bad,
            "sfreq": sf, "duration_min": rec["duration_min"], "boundary_marks": rec["boundary_marks"]}
    return rows, info


def extract_cohort(items, name, params, ctx, harmonize_band=False):
    """Extract features for all (subject_id, file, mapping) items; cached in outputs/features/ and always read back
    from the CSV (exact round-trip), so the first and later runs use exactly the same numbers."""
    folder = ctx.out / "features"
    f_csv, f_info = folder / f"{name}.csv", folder / f"{name}_per_subject.csv"
    f_fail, f_par = folder / f"{name}_failures.csv", folder / f"{name}.json"

    def read():
        fail = (pd.read_csv(f_fail, dtype={"subject_id": str}) if f_fail.exists() and f_fail.stat().st_size > 25
                else pd.DataFrame(columns=["subject_id", "file", "error"]))
        return (pd.read_csv(f_csv, dtype={"subject_id": str}, float_precision="round_trip"),
                pd.read_csv(f_info, dtype={"subject_id": str}, float_precision="round_trip"), fail)

    if f_csv.exists() and f_info.exists() and f_par.exists() and json.loads(f_par.read_text(encoding="utf-8")) == params:
        print(f"[cache] {name}")
        return read()
    rows, infos, failures, t0 = [], [], [], time.time()
    for k, (sid, fpath, mapping) in enumerate(items):
        try:
            r, info = features_19ch(fpath, mapping, sid, ctx.plan, ctx.rt["max_epochs"], harmonize_band)
            rows += r
            infos.append(info)
        except Exception as exc:
            failures.append({"subject_id": sid, "file": str(fpath), "error": str(exc)[:150]})
        if (k + 1) % 10 == 0 or k + 1 == len(items):
            dt = time.time() - t0
            print(f"  {name}: {k + 1}/{len(items)} files | {dt:.0f} s | ~{dt / (k + 1) * (len(items) - k - 1):.0f} s left")
    if not rows:
        raise RuntimeError(f"{name}: no file produced features. First errors: {failures[:3]}")
    pd.DataFrame(rows).to_csv(f_csv, index=False)
    pd.DataFrame(infos).to_csv(f_info, index=False)
    pd.DataFrame(failures, columns=["subject_id", "file", "error"]).to_csv(f_fail, index=False)
    f_par.write_text(json.dumps(params, ensure_ascii=False), encoding="utf-8")
    return read()


def failure_reason(error):
    """Short, reportable reason for a recording without features."""
    e = str(error).lower()
    if ".fdt" in e or "no such file" in e:
        return "EEG data file (.fdt) missing"
    if "channel" in e:
        return "required channels missing"
    if "not found" in e:
        return "EEG data file (.fdt) missing"
    if "epoch" in e:
        return "no valid 4-s epoch"
    return "file could not be read"


# ---- BioSemi-128 -> 10-20 correspondence ---------------------------------------------------------------------------
def name_in_montage(c, pos):
    """Accept T7/T8/P7/P8 when the montage has no T3/T4/T5/T6."""
    return next((x for x in [c] + ALIASES.get(c, []) if x in pos), None)


def standard_1020_positions(channels):
    """'standard_1020' became 'colin27_1020' in recent MNE versions; try both (and the 10-05 versions)."""
    for name in ("colin27_1020", "standard_1020", "colin27_1005", "standard_1005"):
        try:
            pos = mne.channels.make_standard_montage(name).get_positions()["ch_pos"]
        except Exception:
            continue
        if all(name_in_montage(c, pos) for c in channels):
            return pos, name
    raise RuntimeError("No standard MNE montage has the 19 positions of the 10-20 system.")


def unit_vectors(pos_dict):
    """Electrode positions projected on the unit sphere fitted to them (least-squares sphere centre)."""
    names = [k for k, v in pos_dict.items() if v is not None and np.all(np.isfinite(v))]
    P = np.array([pos_dict[k] for k in names], float)
    A = np.column_stack([2 * P, np.ones(len(P))])          # |p - c|^2 = r^2  ->  2p.c + k = |p|^2
    sol = np.linalg.lstsq(A, (P ** 2).sum(axis=1), rcond=None)[0]
    U = P - sol[:3]
    return names, U / np.linalg.norm(U, axis=1, keepdims=True)


def map_to_1020(pos_src, U_st, n_st, target_names, channels):
    """One-to-one assignment of source electrodes to the 19 positions minimising the angle (degrees)."""
    n_o, U_o = unit_vectors(pos_src)
    U_target = np.array([U_st[n_st.index(target_names[c])] for c in channels])
    ang = np.degrees(np.arccos(np.clip(U_target @ U_o.T, -1, 1)))      # 19 x n
    r_idx, c_idx = linear_sum_assignment(ang)
    return pd.DataFrame({"channel_1020": [channels[i] for i in r_idx], "channel_brainlat": [n_o[j] for j in c_idx],
                         "angle_deg": ang[r_idx, c_idx]})


def brainlat_channel_map(files, channels):
    """Correspondence BioSemi-128 -> 10-20 from the first readable file: the positions stored in the file and the
    MNE biosemi128 template (by channel name); the source with the smaller maximum angle is used.

    Returns (mapping table, chosen source, montage name, file used, {source: table})."""
    raw0, file0 = None, None
    for f in files:
        try:
            raw0 = mne.io.read_raw_eeglab(str(f), preload=False, verbose=False)
            file0 = f
            break
        except Exception:
            continue
    if raw0 is None:
        raise RuntimeError("No BrainLat .set file could be read.")
    pos_1020, montage_name = standard_1020_positions(channels)
    n_st, U_st = unit_vectors(pos_1020)
    target_names = {c: name_in_montage(c, pos_1020) for c in channels}
    candidates = {}
    mont0 = raw0.get_montage()
    pos_file = {k: v for k, v in (mont0.get_positions()["ch_pos"].items() if mont0 is not None else [])
                if k in raw0.ch_names and v is not None and np.all(np.isfinite(v)) and np.linalg.norm(v) > 0}
    if len(pos_file) >= 100:
        candidates["positions stored in the file"] = map_to_1020(pos_file, U_st, n_st, target_names, channels)
    tmpl = mne.channels.make_standard_montage("biosemi128").get_positions()["ch_pos"]
    pos_tmpl = {k: tmpl[k] for k in raw0.ch_names if k in tmpl}
    if len(pos_tmpl) >= 100:
        candidates["MNE biosemi128 template (by channel name)"] = map_to_1020(pos_tmpl, U_st, n_st, target_names,
                                                                             channels)
    if not candidates:
        raise RuntimeError("Could not obtain the BrainLat electrode positions (neither from the file nor by name).")
    source = min(candidates, key=lambda k: candidates[k].angle_deg.max())
    table = (candidates[source].sort_values("channel_1020", key=lambda s: s.map(channels.index))
             .reset_index(drop=True))
    return table, source, montage_name, file0, candidates
