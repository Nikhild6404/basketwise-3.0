"""Small helpers shared by every step: saving figures, tables and results."""
from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")  # no display needed; works on any laptop or server
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config as C

PALETTE = ["#2a6f97", "#e07a5f", "#3d9970", "#9b5de5", "#f2a541",
           "#577590", "#c44569", "#43aa8b", "#7f7f7f", "#f28482"]

plt.rcParams.update({
    "figure.dpi": 110, "savefig.dpi": 150, "savefig.bbox": "tight",
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.titleweight": "bold", "axes.titlesize": 12, "font.size": 10,
    "axes.prop_cycle": matplotlib.cycler(color=PALETTE),
})


def ensure_dirs() -> None:
    for d in (C.DATA_PROCESSED, C.FIG_DIR, C.TAB_DIR):
        d.mkdir(parents=True, exist_ok=True)


def savefig(fig, name: str) -> None:
    ensure_dirs()
    fig.savefig(C.FIG_DIR / f"{name}.png")
    plt.close(fig)


def save_table(df: pd.DataFrame, name: str, index: bool = False) -> None:
    ensure_dirs()
    df.to_csv(C.TAB_DIR / f"{name}.csv", index=index)


def _to_jsonable(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return round(float(o), 4)
    if isinstance(o, dict):
        return {str(k): _to_jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_to_jsonable(v) for v in o]
    if isinstance(o, float):
        return round(o, 4)
    return o


def save_results(section: str, payload: dict) -> None:
    """Merge one step's headline numbers into outputs/results_summary.json."""
    ensure_dirs()
    data = {}
    if C.RESULTS_FILE.exists():
        data = json.loads(C.RESULTS_FILE.read_text())
    data[section] = _to_jsonable(payload)
    C.RESULTS_FILE.write_text(json.dumps(data, indent=2))


def banner(title: str) -> None:
    print("\n" + "=" * 70 + f"\n{title}\n" + "=" * 70)
