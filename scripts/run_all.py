"""Run the notebooks in order, without opening Jupyter.

    python scripts/run_all.py                 # 00 -> 12
    python scripts/run_all.py --from 03       # 03 -> 12 (earlier notebooks must have run once)
    python scripts/run_all.py --from 09       # only the post hoc amendments (09-11) and the article (12)
    python scripts/run_all.py --to 01         # 00 -> 01
    python scripts/run_all.py --quick         # quick test mode (never for results)

The notebooks run in this same Python interpreter (the one with the pinned packages), never in another interpreter
that happens to own a Jupyter kernel called "python3". Executed copies (with outputs) are saved to
<outputs>/executed_notebooks/; the notebooks in notebooks/ stay clean.
Settings come from config/paths.yaml (or the EEGAD_* environment variables).
"""
import argparse
import asyncio
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

import nbformat
from nbclient import NotebookClient
from nbclient.exceptions import CellExecutionError

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))


def kernel_for_this_interpreter():
    """Write a temporary kernel spec that starts this same interpreter and put it first on JUPYTER_PATH."""
    from ipykernel.kernelspec import write_kernel_spec
    root = Path(tempfile.mkdtemp(prefix="eegad_kernel_"))
    write_kernel_spec(path=root / "kernels" / "eegad-run", overrides={"display_name": "eegad (run_all)"})
    os.environ["JUPYTER_PATH"] = os.pathsep.join(p for p in (str(root), os.environ.get("JUPYTER_PATH", "")) if p)
    return "eegad-run", root


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--from", dest="start", default="00")
    ap.add_argument("--to", dest="end", default="99")
    ap.add_argument("--quick", action="store_true", help="quick test mode (sets EEGAD_QUICK=1)")
    ap.add_argument("--kernel", default=None, help="name of an installed Jupyter kernel (default: this interpreter)")
    ap.add_argument("--timeout", type=int, default=0, help="seconds per cell (0 = no limit; the null bootstrap can take many hours)")
    a = ap.parse_args()
    if a.quick:
        os.environ["EEGAD_QUICK"] = "1"
    os.environ["EEGAD_REPO_ROOT"] = str(REPO)
    if sys.platform == "win32" and sys.version_info < (3, 14):
        # pyzmq (used by Jupyter) needs a selector event loop on Windows; ipykernel and jupyter_server do the same
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    from eegad.config import load_local_settings
    out = load_local_settings(REPO)["outputs_dir"] / "executed_notebooks"
    out.mkdir(parents=True, exist_ok=True)
    notebooks = sorted(p for p in (REPO / "notebooks").glob("[0-9][0-9]_*.ipynb") if a.start <= p.name[:2] <= a.end)
    kernel, tmp_kernel = (a.kernel, None) if a.kernel else kernel_for_this_interpreter()
    print(f"Python: {sys.executable} ({sys.version.split()[0]}); kernel: {kernel}", flush=True)
    t_all = time.time()
    try:
        for nb_path in notebooks:
            nb = nbformat.read(nb_path, as_version=4)
            client = NotebookClient(nb, timeout=a.timeout if a.timeout > 0 else None, kernel_name=kernel,
                                    resources={"metadata": {"path": str(REPO / "notebooks")}})
            t0 = time.time()
            print(f"{nb_path.name} ...", flush=True)
            try:
                client.execute()
            except CellExecutionError as exc:
                nbformat.write(nb, out / nb_path.name)
                print(f"FAILED after {time.time() - t0:.0f} s; executed copy: {out / nb_path.name}\n{str(exc)[-3000:]}")
                sys.exit(1)
            nbformat.write(nb, out / nb_path.name)
            print(f"  done in {time.time() - t0:.0f} s", flush=True)
    finally:
        if tmp_kernel is not None:
            shutil.rmtree(tmp_kernel, ignore_errors=True)
    print(f"All {len(notebooks)} notebook(s) ran in {(time.time() - t_all) / 60:.1f} min. Executed copies: {out}")


if __name__ == "__main__":
    main()
