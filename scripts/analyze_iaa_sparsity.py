#!/usr/bin/env python3
"""
Analyze annotator agreement vs soft-score rarity to explain low-κ categories.

Addresses reviewer concern that low IAA on critical categories undermines the gold set:
we report (a) unanimity by category, (b) positive prevalence, (c) share of items with
split votes (s in {1/3,2/3}), which is the mechanical driver of low κ for sparse labels.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCES = ["reddit", "news", "meeting_minutes", "x"]
GOLD_TEXT = {
    "reddit": "gold_standard/sampled_reddit_comments.csv",
    "x": "gold_standard/sampled_twitter_posts.csv",
    "news": "gold_standard/sampled_lexisnexis_news.csv",
    "meeting_minutes": "gold_standard/sampled_meeting_minutes.csv",
}
CATEGORIES = [
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=REPO_ROOT / "output" / "audits" / "iaa_analysis",
    )
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for cat in CATEGORIES:
        softs = []
        for src in SOURCES:
            path = REPO_ROOT / "output" / "annotation" / "soft_labels" / f"{src}_soft_labels.csv"
            df = pd.read_csv(path)
            if cat not in df.columns:
                continue
            n_gold = len(pd.read_csv(REPO_ROOT / GOLD_TEXT[src], low_memory=False))
            vals = pd.to_numeric(df[cat], errors="coerce").fillna(0).astype(float).values[:n_gold]
            softs.append(vals)
        s = np.concatenate(softs)
        n = len(s)
        # soft score s = k/3 for k annotators positive
        maj_pos = int((s >= 2 / 3 - 1e-9).sum())
        any_pos = int((s >= 1 / 3 - 1e-9).sum())
        split = int(np.isin(s, [1 / 3, 2 / 3]).sum())
        unanimous = int(np.isin(s, [0.0, 1.0]).sum())
        rows.append(
            {
                "category": cat,
                "n": n,
                "gold_pos_2of3": maj_pos,
                "prevalence_2of3_pct": 100.0 * maj_pos / n,
                "any_pos_1of3": any_pos,
                "split_vote_items": split,
                "split_vote_pct": 100.0 * split / n,
                "unanimous_pct": 100.0 * unanimous / n,
            }
        )
    out = pd.DataFrame(rows).sort_values("unanimous_pct")
    out.to_csv(args.out_dir / "iaa_soft_score_summary.csv", index=False)

    display = {
        "ask a genuine question": "Genuine Q",
        "ask a rhetorical question": "Rhetorical Q",
        "provide a fact or claim": "Fact/claim",
        "provide an observation": "Observation",
        "express their opinion": "Own opinion",
        "express others opinions": "Others' op.",
        "money aid allocation": "Money/aid",
        "government critique": "Gov. critique",
        "societal critique": "Soc. critique",
        "solutions/interventions": "Solutions",
        "personal interaction": "Personal",
        "media portrayal": "Media",
        "not in my backyard": "NIMBY",
        "harmful generalization": "Harm. gen.",
        "deserving/undeserving": "Deservingness",
        "racist": "Racist",
    }
    lines = [
        "% Auto-generated IAA soft-score summary",
        "\\begin{table}[!htb]",
        "\\centering",
        "\\scriptsize",
        "\\setlength{\\tabcolsep}{2pt}",
        "\\begin{tabularx}{\\columnwidth}{@{}>{\\raggedright\\arraybackslash}X r r r r@{}}",
        "\\toprule",
        "\\textbf{Cat.} & $n_{+}$ & Split & Unan. & Prev. \\\\",
        "\\midrule",
    ]
    for _, r in out.iterrows():
        cat = display.get(str(r["category"]), str(r["category"]))
        lines.append(
            f"{cat} & {int(r['gold_pos_2of3'])} & {r['split_vote_pct']:.1f} & "
            f"{r['unanimous_pct']:.1f} & {r['prevalence_2of3_pct']:.1f} \\\\"
        )
    lines += [
        "\\bottomrule",
        "\\end{tabularx}",
        "\\caption{Soft-label structure ($n{=}1{,}698$; 2-of-3 gold). Split = share with $s_{ij}\\in\\{{1/3,2/3\\}}$; Unan.\\ = $s_{ij}\\in\\{{0,1\\}}$; Prev.\\ = 2-of-3 positive rate.}",
        "\\label{tab:iaa_soft_structure}",
        "\\end{table}",
    ]
    (args.out_dir / "iaa_soft_structure.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(out.head(8).to_string(index=False))
    print(f"Wrote {args.out_dir}")


if __name__ == "__main__":
    main()
