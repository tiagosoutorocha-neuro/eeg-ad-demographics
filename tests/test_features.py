import numpy as np

from eegad.features import (extract_connectivity_features, extract_spectral_features, failure_reason,
                            locate_channels)


def _sine(freq, sf=500.0, seconds=4.0, n_ch=19, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(int(sf * seconds)) / sf
    return np.sin(2 * np.pi * freq * t)[None, :] + 0.01 * rng.normal(size=(n_ch, len(t)))


def test_theta_signal_has_high_relative_theta():
    f = extract_spectral_features(_sine(6.0), 500.0, n_fft=500)
    assert f["Rel_Theta_mean"] > 0.9
    assert f["Ratio_Theta_Alpha"] > 10 and f["Ratio_Theta_Beta"] > 10
    assert np.isclose(sum(f[f"Rel_{b}_mean"] for b in ("Delta", "Theta", "Alpha", "Beta", "Gamma")), 1.0, atol=1e-6)


def test_alpha_signal_has_low_theta_alpha_ratio():
    f = extract_spectral_features(_sine(10.0), 500.0, n_fft=500)
    assert f["Rel_Alpha_mean"] > 0.9 and f["Ratio_Theta_Alpha"] < 0.1


def test_identical_channels_are_fully_synchronised():
    c = extract_connectivity_features(_sine(6.0), 500.0)
    assert c["PLV_Theta_mean"] > 0.99 and c["Corr_Broadband_mean"] > 0.99


def test_channel_aliases():
    pairs = locate_channels(["Fp1", "T7", "P8"], {"Fp1": "Fp1", "T3": "T3", "T6": "T6"})
    assert pairs == [("Fp1", "Fp1"), ("T7", "T3"), ("P8", "T6")]


def test_failure_reasons():
    assert failure_reason("File /x/sub-1.fdt not found") == "EEG data file (.fdt) missing"
    assert failure_reason("channel T3 (position T3) not found in the file") == "required channels missing"
    assert failure_reason("no 4-s epoch") == "no valid 4-s epoch"
    assert failure_reason("Mat file appears to be truncated") == "file could not be read"
