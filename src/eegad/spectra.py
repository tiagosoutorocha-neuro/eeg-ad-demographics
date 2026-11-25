"""Post hoc amendment 2, parts B and C (notebook 11): group-average power spectra of the harmonised EEG and the
geometry of the BrainLat 10-20 correspondence. Descriptive only.

The spectra use exactly the preprocessing of the features (features.preprocessed_epochs) and the same Welch settings
as the spectral features; each epoch's spectrum is averaged over the 19 channels and divided by its power in
0.5-45 Hz, so the curves are on the same relative scale as the relative band powers."""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from mne.time_frequency import psd_array_welch

from .features import (FMAX, FMIN, map_to_1020, name_in_montage, preprocessed_epochs, standard_1020_positions,
                       unit_vectors)


# ---- B. spectra --------------------------------------------------------------------------------------------------
def relative_spectrum(ep, sf, resolution_hz):
    """Mean over epochs of the channel-averaged Welch spectrum divided by its power in 0.5-45 Hz (unit: 1/Hz)."""
    n_fft = min(int(round(sf / float(resolution_hz))), ep.shape[-1])
    psds, freqs = psd_array_welch(ep, sfreq=sf, fmin=FMIN, fmax=FMAX, n_fft=n_fft, verbose=False)
    psd = psds.mean(axis=1)                                     # epochs x frequencies (mean of the 19 channels)
    integrate = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
    total = integrate(psd, freqs, axis=1)
    ok = np.isfinite(total) & (total > 0)
    rel = psd[ok] / total[ok, None]
    return freqs, rel.mean(axis=0), int(ok.sum())


def cohort_spectra(items, name, params, ctx, harmonize_band=False):
    """Relative spectrum of every participant (items: subject_id, file, mapping), cached in outputs/descriptive/."""
    folder = ctx.out / "descriptive"
    folder.mkdir(parents=True, exist_ok=True)
    f_csv, f_par = folder / f"spectra_{name}.csv", folder / f"spectra_{name}.json"
    if f_csv.exists() and f_par.exists() and json.loads(f_par.read_text(encoding="utf-8")) == params:
        print(f"[cache] spectra {name}")
        return pd.read_csv(f_csv, dtype={"subject_id": str}, float_precision="round_trip")
    rows, t0 = [], time.time()
    res = ctx.plan["preprocessing"]["welch_resolution_hz"]
    for k, (sid, fpath, mapping) in enumerate(items):
        try:
            ep, sf, _ = preprocessed_epochs(fpath, mapping, ctx.plan, ctx.rt["max_epochs"], harmonize_band)
            freqs, spec, n_ep = relative_spectrum(ep, sf, res)
            rows.append({"subject_id": sid, "n_epochs": n_ep, **{f"{f:g}": v for f, v in zip(freqs, spec)}})
        except Exception as exc:                                # same failures as notebook 01 (e.g. missing .fdt)
            print(f"  {sid}: no spectrum ({str(exc)[:80]})")
        if (k + 1) % 10 == 0 or k + 1 == len(items):
            dt = time.time() - t0
            print(f"  spectra {name}: {k + 1}/{len(items)} files | {dt:.0f} s | ~{dt / (k + 1) * (len(items) - k - 1):.0f} s left")
    df = pd.DataFrame(rows)
    df.to_csv(f_csv, index=False)
    f_par.write_text(json.dumps(params, ensure_ascii=False), encoding="utf-8")
    return pd.read_csv(f_csv, dtype={"subject_id": str}, float_precision="round_trip")


def frequency_columns(df):
    return [c for c in df.columns if c not in ("subject_id", "n_epochs", "y", "cohort")]


def group_spectra(spec, subjects, cohort, n_boot=2000, seed=42):
    """Mean relative spectrum of each diagnostic group with a bootstrap 95% interval (participants resampled).
    Returns a long table: cohort, group, n, frequency, mean, lo, hi (group-level only: shareable)."""
    d = spec.merge(subjects[["subject_id", "y"]], on="subject_id", how="inner")
    fcols = frequency_columns(spec)
    freqs = np.array([float(c) for c in fcols])
    rng = np.random.default_rng(seed)
    rows = []
    for y, group in ((1, "AD"), (0, "HC")):
        M = d.loc[d.y == y, fcols].to_numpy(float)
        if len(M) == 0:
            continue
        idx = rng.integers(0, len(M), size=(int(n_boot), len(M)))
        boot = M[idx].mean(axis=1)                              # n_boot x frequencies
        lo, hi = np.percentile(boot, [2.5, 97.5], axis=0)
        for f, m, l_, h_ in zip(freqs, M.mean(axis=0), lo, hi):
            rows.append({"cohort": cohort, "group": group, "n": len(M), "frequency_hz": f, "mean": m, "lo": l_,
                         "hi": h_})
    return pd.DataFrame(rows)


# ---- C. montage geometry -----------------------------------------------------------------------------------------
def montage_geometry(file0, channels, s1, source):
    """Unit-sphere positions of the BrainLat electrodes and of the 19 standard positions, as used by notebook 01
    (brainlat_channel_map), with the assignment of Table S1. Returns (electrodes, targets, spacing)."""
    import mne

    raw0 = mne.io.read_raw_eeglab(str(file0), preload=False, verbose=False)
    pos_1020, montage_name = standard_1020_positions(channels)
    n_st, U_st = unit_vectors(pos_1020)
    target_names = {c: name_in_montage(c, pos_1020) for c in channels}
    if source.startswith("positions stored in the file"):
        mont0 = raw0.get_montage()
        pos_src = {k: v for k, v in mont0.get_positions()["ch_pos"].items()
                   if k in raw0.ch_names and v is not None and np.all(np.isfinite(v)) and np.linalg.norm(v) > 0}
    else:
        tmpl = mne.channels.make_standard_montage("biosemi128").get_positions()["ch_pos"]
        pos_src = {k: tmpl[k] for k in raw0.ch_names if k in tmpl}
    check = map_to_1020(pos_src, U_st, n_st, target_names, channels).set_index("channel_1020")
    s1i = s1.set_index("channel_1020")
    same = all(check.loc[c, "channel_brainlat"] == s1i.loc[c, "channel_brainlat"] for c in channels)
    if not same:
        raise RuntimeError("The electrode positions do not reproduce Table S1 (outputs/tables/S1_...): was notebook 01 "
                           "run on other files?")
    n_o, U_o = unit_vectors(pos_src)
    chosen = dict(zip(s1i["channel_brainlat"], s1i.index))
    electrodes = pd.DataFrame({"electrode": n_o, "x": U_o[:, 0], "y": U_o[:, 1], "z": U_o[:, 2],
                               "assigned_to": [chosen.get(n, "") for n in n_o]})
    U_t = np.array([U_st[n_st.index(target_names[c])] for c in channels])
    ang = np.degrees(np.arccos(np.clip(U_t @ U_t.T, -1, 1)))
    np.fill_diagonal(ang, np.inf)
    targets = pd.DataFrame({"channel_1020": channels, "x": U_t[:, 0], "y": U_t[:, 1], "z": U_t[:, 2],
                            "angle_deg": [float(s1i.loc[c, "angle_deg"]) for c in channels],
                            "nearest_1020_neighbour": [channels[j] for j in ang.argmin(axis=1)],
                            "neighbour_angle_deg": ang.min(axis=1)})
    spacing = {"montage_1020": montage_name, "source": source, "file": Path(file0).name,
               "median_neighbour_angle_deg": float(np.median(targets.neighbour_angle_deg)),
               "min_neighbour_angle_deg": float(targets.neighbour_angle_deg.min())}
    return electrodes, targets, spacing
