#!/usr/bin/env python3
"""Export annotator-disagreement excerpts to a LaTeX appendix fragment."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]


def latex_escape(s: str) -> str:
    return (
        s.replace("\\", "\\textbackslash{}")
        .replace("&", "\\&")
        .replace("%", "\\%")
        .replace("$", "\\$")
        .replace("#", "\\#")
        .replace("_", "\\_")
        .replace("{", "\\{")
        .replace("}", "\\}")
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--csv",
        type=Path,
        default=REPO_ROOT
        / "output"
        / "openreview_artifacts"
        / "disagreement_examples"
        / "disagreement_provide_a_fact_or_claim.csv",
    )
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument(
        "--out",
        type=Path,
        default=REPO_ROOT
        / "output"
        / "openreview_artifacts"
        / "disagreement_examples"
        / "disagreement_fact_claim.tex",
    )
    args = ap.parse_args()

    df = pd.read_csv(args.csv).head(args.n)
    lines = [
        "% Auto-generated disagreement examples",
        "\\begin{table}[t]",
        "\\centering",
        "\\footnotesize",
        "\\begin{tabular}{@{}l c p{0.72\\linewidth}@{}}",
        "\\toprule",
        "\\textbf{Source} & $s_{ij}$ & \\textbf{De-identified excerpt} \\\\",
        "\\midrule",
    ]
    for _, r in df.iterrows():
        soft = float(r["soft_score"])
        soft_s = (
            "1/3"
            if abs(soft - 1 / 3) < 1e-6
            else ("2/3" if abs(soft - 2 / 3) < 1e-6 else f"{soft:.2f}")
        )
        excerpt = latex_escape(str(r["text_excerpt"]))
        src = str(r["source"]).replace("_", " ")
        lines.append(f"{src} & ${soft_s}$ & ``{excerpt}'' \\\\")
    lines += [
        "\\bottomrule",
        "\\end{tabular}",
        "\\caption{Illustrative annotator disagreements on \\textit{provide a fact or claim} "
        "(soft score $s_{ij}\\in\\{1/3,2/3\\}$). Council and news items often mix procedural "
        "speech with asserted claims, so raters split on whether a sentence is a verifiable claim "
        "vs.\\ opinionated or meta discourse.}",
        "\\label{tab:disagreement_fact_claim}",
        "\\end{table}",
    ]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
