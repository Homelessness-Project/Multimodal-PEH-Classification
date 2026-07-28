#!/usr/bin/env python3
"""
Plot a calibration / reliability diagram for NIMBY using stored
gold + vote-fraction consensus items produced by:
scripts/zeroshot_calibration_audit.py

This uses vote_fraction as the "model confidence" proxy:
  p = mean{ six prompt-model predictions } for each item.
So the curve is directly aligned with how prevalence-gap auditing
is done in the paper.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]


def compute_reliability_bins(
    *,
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int,
) -> pd.DataFrame:
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    rows = []
    for i in range(n_bins):
        lo, hi = float(bins[i]), float(bins[i + 1])
        if i < n_bins - 1:
            mask = (y_prob >= lo) & (y_prob < hi)
        else:
            mask = (y_prob >= lo) & (y_prob <= hi)
        count = int(mask.sum())
        if count == 0:
            continue
        rows.append(
            {
                "bin_lo": lo,
                "bin_hi": hi,
                "count": count,
                "mean_prob": float(np.mean(y_prob[mask])),
                "emp_pos_rate": float(np.mean(y_true[mask])),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--shot_type",
        choices=["zero_shot", "few_shot"],
        default="zero_shot",
        help="Choose which audit artifact to use.",
    )
    ap.add_argument("--n_bins", type=int, default=10)
    ap.add_argument(
        "--out_dir",
        type=Path,
        default=REPO_ROOT / "output" / "openreview_artifacts" / "calibration",
    )
    args = ap.parse_args()

    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    item_path = (
        REPO_ROOT / "output" / "f1" / "zeroshot_calibration_audit" / f"item_vote_fraction_{args.shot_type}.csv"
    )
    df = pd.read_csv(item_path)

    # NIMBY
    sub = df[df["category"] == "not in my backyard"].copy()
    if sub.empty:
        raise ValueError(f"No rows found for NIMBY in {item_path}")

    y_true = sub["gold"].astype(int).values
    y_prob = sub["vote_fraction"].astype(float).values

    bins = compute_reliability_bins(
        y_true=y_true,
        y_prob=y_prob,
        n_bins=args.n_bins,
    )

    # Plot
    plt.figure(figsize=(5.2, 4.2))
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1)
    plt.scatter(bins["mean_prob"], bins["emp_pos_rate"], s=50 + 10 * bins["count"], color="tab:blue")

    plt.xlabel("Predicted NIMBY probability (vote fraction)")
    plt.ylabel("Empirical NIMBY positive rate (gold)")
    plt.title(f"NIMBY reliability diagram ({args.shot_type})")
    plt.xlim(0, 1.0)
    plt.ylim(0, 1.0)
    plt.grid(alpha=0.2)

    fig_base = out_dir / f"nimby_reliability_{args.shot_type}"
    plt.tight_layout()
    fig_path_png = fig_base.with_suffix(".png")
    fig_path_pdf = fig_base.with_suffix(".pdf")
    plt.savefig(fig_path_png, dpi=200)
    plt.savefig(fig_path_pdf)
    plt.close()

    bins_out = out_dir / f"nimby_reliability_bins_{args.shot_type}.csv"
    bins.to_csv(bins_out, index=False)

    print(f"Wrote: {fig_path_png}")
    print(f"Wrote: {fig_path_pdf}")
    print(f"Wrote: {bins_out}")


if __name__ == "__main__":
    main()

