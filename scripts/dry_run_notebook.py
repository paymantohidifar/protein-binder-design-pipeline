#!/usr/bin/env python
"""Execute the prototype notebook's code cells without a GPU or any containers.

The notebook sets DRY_RUN from whether nvidia-smi exists, so on a GPU-less host
every pipeline.docker call prints its command instead of running it. That makes
the whole notebook executable as plain Python, which is the strongest check
available before an instance is reserved: it exercises the real imports, the
real config wiring and the real parser entry points. Worth running after any 
change to pipeline/ or the notebook.

Usage:
    pixi run notebook-dry-run
    python scripts/dry_run_notebook.py [path/to/notebook.ipynb]

Exits non-zero if any cell raises.
"""

from __future__ import annotations

import builtins
import json
import os
import sys
import traceback
from pathlib import Path

DEFAULT_NOTEBOOK = "notebooks/01_binder_design_prototype.ipynb"


def main(argv: list[str]) -> int:
    repo = Path(__file__).resolve().parents[1]
    os.chdir(repo)
    sys.path.insert(0, str(repo))

    notebook = Path(argv[1]) if len(argv) > 1 else repo / DEFAULT_NOTEBOOK
    if not notebook.exists():
        print(f"notebook not found: {notebook}", file=sys.stderr)
        return 2

    # IPython builtins the cells rely on. display() is stubbed rather than
    # removed so the viz cells still exercise their code paths.
    builtins.display = lambda *args, **kwargs: print(
        f"   [display: {type(args[0]).__name__}]" if args else "   [display]"
    )

    cells = [
        "".join(cell["source"])
        for cell in json.loads(notebook.read_text())["cells"]
        if cell["cell_type"] == "code"
    ]
    print(f"{notebook.relative_to(repo)}: {len(cells)} code cells\n")

    # One shared namespace, so later cells see earlier state exactly as the
    # notebook would.
    namespace: dict = {"__name__": "__main__"}
    failures: list[tuple[int, str, str]] = []

    for index, source in enumerate(cells):
        print(f"--- cell {index} " + "-" * 48)
        try:
            exec(compile(source, f"<cell {index}>", "exec"), namespace)
        except Exception as exc:  # noqa: BLE001 - report every cell, keep going
            failures.append((index, type(exc).__name__, str(exc)[:200]))
            print(f"!! {type(exc).__name__}: {str(exc)[:300]}", file=sys.stderr)
            traceback.print_exc(limit=2)

    print("\n" + "=" * 62)
    if failures:
        print(f"FAIL: {len(failures)} of {len(cells)} cells raised")
        for index, kind, message in failures:
            print(f"  cell {index}: {kind}: {message}")
        return 1

    print(f"OK: all {len(cells)} cells executed without raising")
    print("\nNote: this proves the plumbing, not the science. Containers were")
    print("never invoked, so no metric was actually computed.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
