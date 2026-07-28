"""
Canonical gold-standard audit set (n=1,698).

The stratified sample CSVs sum to 1,702 rows, but four items lack matching
human annotation (X posts with no raw-score match). All gold-standard metrics
use the 1,698 annotated items only.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]

GOLD_STANDARD_N = 1698

# gold_csv, gold_text_col, raw_scores_csv, raw_text_col
SOURCE_SPECS: Dict[str, Tuple[str, str, str, str]] = {
    "reddit": (
        "gold_standard/sampled_reddit_comments.csv",
        "Comment",
        "annotation/reddit_raw_scores.csv",
        "Deidentified_Comment",
    ),
    "x": (
        "gold_standard/sampled_twitter_posts.csv",
        "Deidentified_text",
        "annotation/x_raw_scores.csv",
        "Deidentified text",
    ),
    "news": (
        "gold_standard/sampled_lexisnexis_news.csv",
        "Deidentified_paragraph_text",
        "annotation/news_raw_scores.csv",
        "Deidentified_paragraph",
    ),
    "meeting_minutes": (
        "gold_standard/sampled_meeting_minutes.csv",
        "Deidentified_paragraph",
        "annotation/meeting_minutes_raw_scores.csv",
        "Deidentified_paragraph",
    ),
}

SOURCES = list(SOURCE_SPECS.keys())


def normalize_text(text: str) -> str:
    return " ".join(str(text).split()).strip().lower()


def annotated_text_set(source: str) -> set[str]:
    """Normalized texts with human raw scores in the stratified gold sample."""
    gold_path, gold_col, raw_path, raw_col = SOURCE_SPECS[source]
    gold = pd.read_csv(REPO_ROOT / gold_path, low_memory=False)
    raw = pd.read_csv(REPO_ROOT / raw_path)
    gold_set = set(gold[gold_col].map(normalize_text))
    raw_norm = raw[raw_col].map(normalize_text)
    return set(raw_norm[raw_norm.isin(gold_set)])


def is_annotated_gold_text(source: str, text: str) -> bool:
    return normalize_text(text) in annotated_text_set(source)


def load_annotated_soft_labels(source: str) -> pd.DataFrame:
    """Soft-label rows aligned to human-annotated gold items (same order as raw scores)."""
    _, _, raw_path, raw_col = SOURCE_SPECS[source]
    raw = pd.read_csv(REPO_ROOT / raw_path)
    soft = pd.read_csv(
        REPO_ROOT / "output" / "annotation" / "soft_labels" / f"{source}_soft_labels.csv"
    )
    ann = annotated_text_set(source)
    mask = raw[raw_col].map(normalize_text).isin(ann)
    return soft.iloc[mask.values].reset_index(drop=True)


def gold_standard_item_count() -> int:
    return sum(len(load_annotated_soft_labels(src)) for src in SOURCES)
