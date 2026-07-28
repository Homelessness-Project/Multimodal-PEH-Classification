#!/usr/bin/env python3
"""
Sample de-identified gold texts where annotators disagree on a given category.

This is meant to address reviewer requests like:
  "Provide examples of annotator disagreements for category X".

We use:
- gold texts from gold_standard/sampled_*.csv (de-identified)
- soft labels from output/annotation/soft_labels/*_soft_labels.csv

We select rows where soft score is 1/3 or 2/3 (non-unanimous).
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]

SOURCES = ["reddit", "x", "news", "meeting_minutes"]

SOFT_COLS = [
    "ask a genuine question",
    "ask a rhetorical question",
    "provide a fact or claim",
    "provide an observation",
    "express their opinion",
    "express others opinions",
    "money aid allocation",
    "government critique",
    "societal critique",
    "solutions/interventions",
    "personal interaction",
    "media portrayal",
    "not in my backyard",
    "harmful generalization",
    "deserving/undeserving",
    "racist",
]

SOURCE_TEXT_SPECS: Dict[str, Tuple[str, str]] = {
    "reddit": ("gold_standard/sampled_reddit_comments.csv", "Comment"),
    "x": ("gold_standard/sampled_twitter_posts.csv", "Deidentified_text"),
    "news": ("gold_standard/sampled_lexisnexis_news.csv", "Deidentified_paragraph_text"),
    "meeting_minutes": ("gold_standard/sampled_meeting_minutes.csv", "Deidentified_paragraph"),
}


def _truncate(s: str, max_chars: int) -> str:
    t = (s or "").strip().replace("\n", " ")
    if len(t) <= max_chars:
        return t
    return t[: max_chars - 1].rsplit(" ", 1)[0] + "…"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--category", required=True, choices=SOFT_COLS)
    ap.add_argument("--n_total", type=int, default=12)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--out_dir",
        type=Path,
        default=REPO_ROOT / "output" / "audits" / "disagreement_examples",
    )
    ap.add_argument("--max_chars", type=int, default=280)
    args = ap.parse_args()

    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    rows: List[dict] = []

    for src in SOURCES:
        text_path_rel, text_col = SOURCE_TEXT_SPECS[src]
        text_path = REPO_ROOT / text_path_rel
        soft_path = REPO_ROOT / "output" / "annotation" / "soft_labels" / f"{src}_soft_labels.csv"

        if not text_path.exists():
            raise FileNotFoundError(f"Missing gold texts file: {text_path}")
        if not soft_path.exists():
            raise FileNotFoundError(f"Missing soft labels file: {soft_path}")

        texts = pd.read_csv(text_path, low_memory=False)
        soft = pd.read_csv(soft_path, low_memory=False)

        if args.category not in soft.columns:
            raise KeyError(f"Category {args.category} not found in {soft_path.name}")
        # Align by row index (these are produced from the same sampled files).
        m = min(len(texts), len(soft))
        if m == 0:
            continue
        texts = texts.iloc[:m].reset_index(drop=True)
        soft = soft.iloc[:m].reset_index(drop=True)

        y = pd.to_numeric(soft[args.category], errors="coerce").fillna(0).astype(float).values
        disagree_mask = np.isin(y, [1 / 3, 2 / 3])
        idxs = np.where(disagree_mask)[0]
        if len(idxs) == 0:
            continue

        k = max(1, int(round(args.n_total / len(SOURCES))))
        k = min(k, len(idxs))
        chosen = rng.choice(idxs, size=k, replace=False)
        for i in chosen:
            rows.append(
                {
                    "source": src,
                    "item_idx": int(i),
                    "soft_score": float(y[i]),
                    "text_excerpt": _truncate(str(texts.loc[i, text_col]), args.max_chars),
                }
            )

    if not rows:
        raise RuntimeError(f"No disagreement rows found for category={args.category}")

    df = pd.DataFrame(rows).sort_values(["source", "soft_score"])
    out_csv = out_dir / f"disagreement_{args.category.replace(' ','_').replace('/','_')}.csv"
    df.to_csv(out_csv, index=False)

    print(f"Wrote: {out_csv}")
    print(df.head(10).to_string(index=False))


if __name__ == "__main__":
    main()

