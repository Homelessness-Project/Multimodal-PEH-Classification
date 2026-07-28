#!/usr/bin/env python3
"""
Generate a compact label-prevalence table for:
1) the gold-standard set (from soft labels, majority vote tau=2/3)
2) the GPT-4.1 labeled corpus (from stored GPT-4 flags on all texts)

Outputs a LaTeX table fragment.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from gold_standard_utils import load_annotated_soft_labels

REPO_ROOT = Path(__file__).resolve().parents[1]

SOURCES = ["reddit", "news", "meeting_minutes", "x"]

# Gold/16-category taxonomy (must match paper)
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

# Matches scripts/zeroshot_calibration_audit.py
MODEL_COL_TO_SOFT = {
    "Comment_ask a genuine question": "ask a genuine question",
    "Comment_ask a rhetorical question": "ask a rhetorical question",
    "Comment_provide a fact or claim": "provide a fact or claim",
    "Comment_provide an observation": "provide an observation",
    "Comment_express their opinion": "express their opinion",
    "Comment_express others opinions": "express others opinions",
    "Critique_money aid allocation": "money aid allocation",
    "Critique_government critique": "government critique",
    "Critique_societal critique": "societal critique",
    "Response_solutions/interventions": "solutions/interventions",
    "Perception_personal interaction": "personal interaction",
    "Perception_media portrayal": "media portrayal",
    "Perception_not in my backyard": "not in my backyard",
    "Perception_harmful generalization": "harmful generalization",
    "Perception_deserving/undeserving": "deserving/undeserving",
    "Racist_Flag": "racist",
}

CAT_TO_GPT_COL = {v: k for k, v in MODEL_COL_TO_SOFT.items()}


def _load_gpt_flags(source: str) -> pd.DataFrame:
    # Stored as: output/{source}/gpt4/classified_comments_{source}_all_gpt4_{source}_flags.csv
    path = (
        REPO_ROOT
        / "output"
        / source
        / "gpt4"
        / f"classified_comments_{source}_all_gpt4_{source}_flags.csv"
    )
    return pd.read_csv(path, low_memory=False)


def _binarize_soft(s: pd.Series, *, tau: float) -> np.ndarray:
    x = pd.to_numeric(s, errors="coerce").fillna(0).astype(float).values
    return (x >= tau - 1e-9).astype(int)


def _format_cat(cat: str, *, compact: bool = False) -> str:
    if compact:
        mapping = {
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
        return mapping.get(cat, cat)
    # Keep it readable in tables.
    return cat.replace("express others opinions", "Express others opinions").replace(
        "express their opinion", "Express their opinion"
    ).replace(
        "ask a genuine question", "Ask genuine question"
    ).replace("ask a rhetorical question", "Ask rhetorical question").replace(
        "provide a fact or claim", "Provide fact/claim"
    ).replace(
        "provide an observation", "Provide observation"
    ).replace(
        "money aid allocation", "Money/aid allocation"
    ).replace(
        "government critique", "Government critique"
    ).replace(
        "societal critique", "Societal critique"
    ).replace(
        "solutions/interventions", "Solutions/interventions"
    ).replace(
        "personal interaction", "Personal interaction"
    ).replace(
        "media portrayal", "Media portrayal"
    ).replace(
        "not in my backyard", "Not in my backyard"
    ).replace(
        "harmful generalization", "Harmful generalization"
    ).replace(
        "deserving/undeserving", "Deserving/undeserving"
    ).replace(
        "racist", "Racist"
    )


def compute_gold_prevalence(*, tau: float) -> pd.DataFrame:
    rows = []
    for cat in CATEGORIES:
        total_pos = 0
        total_n = 0
        for src in SOURCES:
            soft = load_annotated_soft_labels(src)
            if cat not in soft.columns:
                continue
            y = _binarize_soft(soft[cat], tau=tau)
            total_pos += int(y.sum())
            total_n += int(len(y))
        prev = float(total_pos) / float(total_n) if total_n else float("nan")
        rows.append(
            {
                "label": cat,
                "n_pos": total_pos,
                "n_total": total_n,
                "prevalence": prev,
                "source": "gold",
            }
        )
    return pd.DataFrame(rows)


def compute_gpt_prevalence() -> pd.DataFrame:
    rows = []
    for cat in CATEGORIES:
        col = CAT_TO_GPT_COL.get(cat)
        if not col:
            raise KeyError(f"Missing mapping for category {cat}")
        total_pos = 0
        total_n = 0
        for src in SOURCES:
            df = _load_gpt_flags(src)
            if col not in df.columns:
                raise KeyError(f"Missing column {col} in GPT flags for source {src}")
            y = (
                pd.to_numeric(df[col], errors="coerce")
                .fillna(0)
                .astype(int)
                .clip(0, 1)
                .values
            )
            total_pos += int(y.sum())
            total_n += int(len(y))
        prev = float(total_pos) / float(total_n) if total_n else float("nan")
        rows.append(
            {
                "label": cat,
                "n_pos": total_pos,
                "n_total": total_n,
                "prevalence": prev,
                "source": "gpt4.1_corpus",
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tau", type=float, default=2 / 3, help="Gold majority threshold (tau=2/3).")
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=REPO_ROOT / "output" / "openreview_artifacts" / "label_prevalence",
    )
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    gold = compute_gold_prevalence(tau=args.tau)
    gpt = compute_gpt_prevalence()

    merged = gold.merge(gpt, on="label", suffixes=("_gold", "_gpt"))
    # Keep paper taxonomy order (not alphabetical).
    merged["ord"] = merged["label"].map({c: i for i, c in enumerate(CATEGORIES)})
    merged = merged.sort_values("ord")

    gold_n = int(merged["n_total_gold"].iloc[0])
    gpt_n = int(merged["n_total_gpt"].iloc[0])
    gold_n_tex = f"{gold_n:,}".replace(",", "{,}")
    gpt_n_tex = f"{gpt_n:,}".replace(",", "{,}")

    # LaTeX table (compact: prevalence + positive counts).
    lines = []
    lines.append("% Auto-generated label prevalence table (gold vs GPT flags)")
    lines.append("\\begin{table}[t]")
    lines.append("\\centering")
    lines.append("\\scriptsize")
    lines.append("\\setlength{\\tabcolsep}{2pt}")
    lines.append(
        "\\begin{tabularx}{\\columnwidth}{@{}>{\\raggedright\\arraybackslash}X r r r r@{}}"
    )
    lines.append("\\toprule")
    lines.append(
        "\\textbf{Label} & \\multicolumn{2}{c}{\\textbf{Gold}} & "
        "\\multicolumn{2}{c}{\\textbf{GPT}} \\\\"
    )
    lines.append(" & \\% & $n_{+}$ & \\% & $n_{+}$ \\\\")
    lines.append("\\midrule")
    for _, r in merged.iterrows():
        gold_prev_pct = 100.0 * float(r["prevalence_gold"])
        gpt_prev_pct = 100.0 * float(r["prevalence_gpt"])
        gold_n_pos = int(r["n_pos_gold"])
        gpt_n_pos = int(r["n_pos_gpt"])
        lines.append(
            f"{_format_cat(str(r['label']), compact=True)} & {gold_prev_pct:.1f} & {gold_n_pos} & "
            f"{gpt_prev_pct:.1f} & {gpt_n_pos} \\\\"
        )
    lines.append("\\bottomrule")
    lines.append("\\end{tabularx}")
    lines.append(
        "\\caption{Label prevalence (\\%) and $n_{+}$ for 16 categories. "
        f"Gold: $n{{=}}{gold_n_tex}$ ($\\tau{{=}}2/3$); GPT corpus: $n{{=}}{gpt_n_tex}$ (exploratory).}"
    )
    lines.append("\\label{tab:label_prevalence_gold_vs_gpt}")
    lines.append("\\end{table}")

    tex_path = args.out_dir / "label_prevalence_gold_vs_gpt.tex"
    tex_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    # Also write CSVs so it's easy to revise columns without re-running.
    gold_path = args.out_dir / "gold_prevalence.csv"
    gpt_path = args.out_dir / "gpt4_1_corpus_prevalence.csv"
    gold.to_csv(gold_path, index=False)
    gpt.to_csv(gpt_path, index=False)
    merged.to_csv(args.out_dir / "label_prevalence_merged.csv", index=False)

    print(f"Wrote LaTeX: {tex_path}")
    print(f"Wrote CSVs: {gold_path}, {gpt_path}")


if __name__ == "__main__":
    main()

