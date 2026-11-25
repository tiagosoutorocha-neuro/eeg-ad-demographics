# Provenance of the original analysis

The analysis was first run on 2026-11-20 with a single Portuguese-language notebook. This folder keeps what is needed
to show what was run, when, and what it produced, without any subject-level data.

| File | Content |
|---|---|
| `original_run/SPEC_journal.json` | The original analysis plan, its hash (`fcdd59491d8d1aa8`) and freeze time (2026-11-20T10:33:16) |
| `original_run/log_execucao.json` | Run log: plan freezes, runs, and when the Greek labels were first read (after the freeze) |
| `original_run/ambiente.txt` | Library versions of the original run (Python 3.11.9, Windows) |
| `original_run/download_ds004504.json` | Verification of ds004504 v1.0.9 (93 of 93 files by SHA-256) |
| `original_run/RESUMO_JOURNAL_redacted.md` | Original results summary; the 14 identifiers of excluded BrainLat participants were redacted |
| `original_run/tables/`, `original_run/figures/` | Aggregated tables and figures of the original run (no per-subject predictions or pairs) |
| `original_run/original_notebook_pt.ipynb` | The original notebook (code only); its two default local data paths were replaced by placeholders |
| `original_run/key_results.json` | 546 numbers extracted from the original outputs, with their reported precision |
| `original_run/MANIFEST.json` | SHA-256 of every file here and of the unredacted originals kept by the authors |
| `extract_original_keys.py` | Script that produced `key_results.json` from the original outputs |

The files in `original_run/` are kept byte for byte (`.gitattributes` turns off line-ending conversion for them), so
their integrity can be checked on any system:

```bash
python -c "import json,hashlib,pathlib; d=pathlib.Path('provenance/original_run'); m=json.loads((d/'MANIFEST.json').read_text(encoding='utf-8')); print(all(hashlib.sha256((d/f).read_bytes()).hexdigest()==h for f,h in m['files_sha256'].items()))"
```

`key_results.json` can be regenerated with

```bash
python provenance/extract_original_keys.py provenance/original_run provenance/original_run/key_results.json
```

and `notebooks/08_reproduction_check.ipynb` compares it with a new run. The original run was later re-run once from its
caches to add the submission section (flow diagram, severity tables, TRIPOD+AI); all numbers in common with the first
complete run were identical.

The analysis plan in `config/analysis_plan.json` is a one-to-one translation of `SPEC_journal.json`; its hash differs
only because keys and descriptions were translated.
