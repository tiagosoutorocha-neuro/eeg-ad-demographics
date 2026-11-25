"""Compare the key results of this run with those of the original run (provenance/original_run/key_results.json).

Original values come from the reported tables, so most have limited precision: a new value "matches" when it rounds
to the reported value (|new - original| <= half a unit of the last reported decimal). Values reported as "< 0.001"
match when the new value is below 0.001. Values stored at full precision must agree to 1e-9.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


def load_original(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))["keys"]


def load_new(path: Path) -> dict:
    return {k: v["value"] for k, v in json.loads(Path(path).read_text(encoding="utf-8")).items()}


def compare(new: dict, original: dict) -> pd.DataFrame:
    rows = []
    for k in sorted(set(original) | set(new)):
        o, n = original.get(k), new.get(k)
        if o is None:
            rows.append({"key": k, "original": None, "new": n, "reported_decimals": None, "status": "new only"})
            continue
        ov, kind, dec = o["value"], o.get("kind", "number"), o.get("decimals")
        if n is None or (isinstance(n, float) and not np.isfinite(n)):
            status = "match" if ov is None else "missing in this run"      # both not available counts as a match
        elif ov is None:
            status = "differs"
        elif kind == "lt":
            status = "match" if float(n) < float(ov) else "differs"
        elif dec is None:
            status = "match" if abs(float(n) - float(ov)) <= 1e-9 * max(1.0, abs(float(ov))) else "differs"
        else:
            status = "match" if abs(float(n) - float(ov)) <= 0.5 * 10 ** (-dec) + 1e-9 else "differs"
        rows.append({"key": k, "original": ("< " + str(ov)) if kind == "lt" else ov, "new": n,
                     "reported_decimals": "full" if dec is None and kind != "lt" else dec, "status": status,
                     "source": o.get("source", "")})
    return pd.DataFrame(rows)


def summary(tab: pd.DataFrame) -> str:
    counts = tab["status"].value_counts().to_dict()
    compared = tab[tab.status.isin(["match", "differs"])]
    return (f"{counts.get('match', 0)} of {len(compared)} compared values match the original run; "
            f"{counts.get('differs', 0)} differ; {counts.get('missing in this run', 0)} missing in this run; "
            f"{counts.get('new only', 0)} only in this run (not reported in the original).")
