"""Result log (persisted across notebooks), number formatting and figure saving."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


class ResultLog:
    """Results, readings (verdicts), alerts and key numbers, stored as JSON in outputs/results/.

    Each notebook calls ``start(notebook)`` (done by ``load_context``): its previous entries are removed, so
    re-running a notebook replaces its results instead of duplicating them.
    """

    def __init__(self, folder: Path):
        self.folder = Path(folder)
        self.folder.mkdir(parents=True, exist_ok=True)
        self.notebook = "??"
        self.entries = self._load("results_log.json", [])
        self.verdicts = self._load("verdicts.json", {})
        self.alerts = self._load("alerts.json", [])
        self.keys = self._load("key_results.json", {})

    def _load(self, name, default):
        f = self.folder / name
        return json.loads(f.read_text(encoding="utf-8")) if f.exists() else default

    def _save(self):
        for name, obj in (("results_log.json", self.entries), ("verdicts.json", self.verdicts),
                          ("alerts.json", self.alerts), ("key_results.json", self.keys)):
            (self.folder / name).write_text(json.dumps(obj, indent=1, ensure_ascii=False, default=_json_default),
                                            encoding="utf-8")

    def start(self, notebook: str) -> None:
        self.notebook = notebook
        self.entries = [e for e in self.entries if e["notebook"] != notebook]
        self.alerts = [a for a in self.alerts if a["notebook"] != notebook]
        self.verdicts = {k: v for k, v in self.verdicts.items() if v["notebook"] != notebook}
        self.keys = {k: v for k, v in self.keys.items() if v["notebook"] != notebook}
        self._save()

    def record(self, section: str, item: str, value: str, where: str = "", note: str = "") -> None:
        self.entries.append({"notebook": self.notebook, "section": section, "item": item, "value": value,
                             "where": where, "note": note})
        self._save()

    def alert(self, msg: str) -> None:
        self.alerts.append({"notebook": self.notebook, "message": msg})
        self._save()
        print("WARNING:", msg)

    def verdict(self, rq: str, reading: str) -> str:
        self.verdicts[rq] = {"notebook": self.notebook, "reading": reading}
        self._save()
        return reading

    def key(self, name: str, value) -> None:
        """Numeric result compared with the original run by notebook 08."""
        self.keys[name] = {"notebook": self.notebook, "value": _json_default(value)}
        self._save()

    def keys_from(self, prefix: str, values: dict) -> None:
        for k, v in values.items():
            self.keys[f"{prefix}.{k}"] = {"notebook": self.notebook, "value": _json_default(v)}
        self._save()

    def get_verdict(self, rq: str) -> str:
        return self.verdicts[rq]["reading"]


def _json_default(v):
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return None if not np.isfinite(v) else float(v)
    if isinstance(v, float) and not np.isfinite(v):
        return None
    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, (np.bool_,)):
        return bool(v)
    if isinstance(v, Path):
        return str(v)
    return v


def write_json(path: Path, obj) -> None:
    Path(path).write_text(json.dumps(obj, indent=1, ensure_ascii=False, default=_json_default), encoding="utf-8")


def read_json(path: Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


# ---- formatting ------------------------------------------------------------------------------------------------
def fmt(a, nd=3):
    try:
        return "n/a" if a is None or not np.isfinite(float(a)) else f"{float(a):.{nd}f}"
    except (TypeError, ValueError):
        return str(a)


def fmt_ci(a, lo, hi, nd=3):
    return f"{fmt(a, nd)} [{fmt(lo, nd)}, {fmt(hi, nd)}]"


def fmt_p(p):
    return "n/a" if p is None or not np.isfinite(p) else ("< 0.001" if p < 0.001 else f"{p:.3f}")


def fmt_metrics(m):
    return (f"sens {m['sens']:.3f}, spec {m['spec']:.3f}, balanced accuracy {m['bal_acc']:.3f}, "
            f"accuracy {m['acc']:.3f}, F1 {m['F1']:.3f}, MCC {m['MCC']:.3f}")


def save_figure(fig, name: str, out: Path, log: ResultLog | None = None) -> Path:
    """PDF (TrueType fonts embedded) + PNG preview; warns if the PDF contains Type 3 fonts."""
    pdf = Path(out) / "figures" / f"{name}.pdf"
    fig.savefig(pdf)
    fig.savefig(Path(out) / "figures" / f"{name}.png", dpi=200)
    if b"/Type3" in pdf.read_bytes() and log is not None:
        log.alert(f"{pdf.name} contains Type 3 fonts; check rcParams['pdf.fonttype'].")
    return pdf
