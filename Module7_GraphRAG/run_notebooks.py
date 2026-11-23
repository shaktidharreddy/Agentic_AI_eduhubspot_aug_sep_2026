"""Execute the lesson notebooks top to bottom and report the first failing cell in each.

Usage:
    uv run python run_notebooks.py              # all notebooks
    uv run python run_notebooks.py L2 L3        # only notebooks whose name starts with these prefixes
    uv run python run_notebooks.py --save       # also write executed outputs back into the notebooks
"""
import glob
import sys
import time

import nbformat
from nbclient import NotebookClient
from nbclient.exceptions import CellExecutionError

args = [a for a in sys.argv[1:] if not a.startswith("--")]
save = "--save" in sys.argv

notebooks = sorted(glob.glob("L*.ipynb"))
if args:
    notebooks = [n for n in notebooks if any(n.startswith(p) for p in args)]

results = []
for path in notebooks:
    nb = nbformat.read(path, as_version=4)
    client = NotebookClient(nb, timeout=600, kernel_name="python3", resources={"metadata": {"path": "."}})
    start = time.time()
    try:
        client.execute()
        status = "PASS"
        detail = ""
    except CellExecutionError as e:
        status = "FAIL"
        failed = next((i for i, c in enumerate(nb.cells) if c.cell_type == "code"
                       and any(o.get("output_type") == "error" for o in c.get("outputs", []))), None)
        err = next((o for o in nb.cells[failed].outputs if o.get("output_type") == "error"), {}) if failed is not None else {}
        detail = f"cell {failed}: {err.get('ename', '')}: {err.get('evalue', str(e))[:1500]}"
    if save:
        nbformat.write(nb, path)
    results.append((path, status, time.time() - start, detail))
    print(f"{status}  {path}  ({time.time() - start:.0f}s)  {detail}", flush=True)

print("\nSummary:")
for path, status, secs, detail in results:
    print(f"  {status}  {path}")
sys.exit(0 if all(r[1] == "PASS" for r in results) else 1)
