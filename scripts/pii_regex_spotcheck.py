#!/usr/bin/env python3
"""
Regex-based PII spot-check on de-identified texts already stored in the repo.

This is intended as a lightweight evidence artifact for the Ethics statement:
we sample rows from the GPT-4.1 labeled corpus CSVs and look for common direct-P​​II patterns
(emails, phone numbers, and street-address-like patterns).

Important: this is a heuristic regex check; it is not a substitute for full manual review.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCES = ["reddit", "news", "meeting_minutes", "x"]


PII_REGEXES: Dict[str, re.Pattern[str]] = {
    # Basic email pattern
    "email": re.compile(r"\b[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}\b"),
    # US-ish phone number patterns (very heuristic)
    "phone": re.compile(
        r"\b(?:\+?1[-.\s]?)?(?:\(\s*\d{3}\s*\)|\d{3})[-.\s]?\d{3}[-.\s]?\d{4}\b"
    ),
    # Street-address-like: leading house number + street type
    "street_address": re.compile(
        r"\b\d{1,6}\s+(?:[A-Za-z0-9'\-]+\.?\s+)?(?:street|st|avenue|ave|road|rd|boulevard|blvd|lane|ln|drive|dr)\b",
        re.IGNORECASE,
    ),
    # Raw URLs (should generally be masked into [URL])
    "url": re.compile(r"\bhttps?://\S+\b", re.IGNORECASE),
}


def _gpt_flags_path(source: str) -> Path:
    return (
        REPO_ROOT
        / "output"
        / source
        / "gpt4"
        / f"classified_comments_{source}_all_gpt4_{source}_flags.csv"
    )


def _get_text_column(df: pd.DataFrame) -> str:
    # For all sources, the de-identified text is stored in `Comment` (for meeting minutes,
    # the flags file uses the same schema).
    if "Comment" in df.columns:
        return "Comment"
    # Fallbacks (defensive)
    for cand in ["Deidentified text", "Deidentified_paragraph_text", "Deidentified_article_title"]:
        if cand in df.columns:
            return cand
    raise KeyError("Could not find a de-identified text column.")


def scan_sample(texts: List[str]) -> Tuple[int, Dict[str, int]]:
    n = len(texts)
    example_matches: Dict[str, int] = {k: 0 for k in PII_REGEXES.keys()}
    for t in texts:
        tt = str(t)
        for name, rx in PII_REGEXES.items():
            if rx.search(tt):
                example_matches[name] += 1
    return n, example_matches


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_per_source", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--out_dir",
        type=Path,
        default=REPO_ROOT / "output" / "audits" / "pii_spotcheck",
    )
    args = ap.parse_args()
    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    rows = []

    for src in SOURCES:
        path = _gpt_flags_path(src)
        if not path.exists():
            raise FileNotFoundError(f"Missing GPT flags file: {path}")
        df = pd.read_csv(path, low_memory=False)
        text_col = _get_text_column(df)

        if len(df) == 0:
            continue
        n = min(args.n_per_source, len(df))
        # Stable sampling by RNG indices
        idx = rng.choice(len(df), size=n, replace=False)
        sample = df.iloc[idx][text_col].astype(str).tolist()

        _, matches = scan_sample(sample)
        row = {"source": src, "n_sampled": n}
        row.update(matches)
        rows.append(row)

    out = pd.DataFrame(rows).sort_values("source")
    out_csv = out_dir / "pii_regex_spotcheck_summary.csv"
    out.to_csv(out_csv, index=False)

    print(f"Wrote: {out_csv}")
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()

