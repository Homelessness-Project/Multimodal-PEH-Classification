#!/usr/bin/env python3
"""
Summarize mitigation strategies that reduce NIMBY prevalence gaps from
already-computed audit outputs (vote thresholds + fine-tuned tables).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
AUDIT = REPO_ROOT / "output" / "f1" / "zeroshot_calibration_audit"


def _load_threshold(shot: str) -> pd.DataFrame:
    path = AUDIT / f"vote_threshold_sweep_{shot}.csv"
    df = pd.read_csv(path)
    return df[df["category"] == "not in my backyard"].copy()


def _format_vote_threshold(t: float) -> str:
    mapping = {
        1 / 6: r"$v\geq\frac{1}{6}$",
        1 / 3: r"$v\geq\frac{1}{3}$",
        0.5: r"$v\geq\frac{1}{2}$",
        2 / 3: r"$v\geq\frac{2}{3}$",
        5 / 6: r"$v\geq\frac{5}{6}$",
        1.0: r"$v=1$",
    }
    for key, label in mapping.items():
        if abs(t - key) < 1e-3:
            return label
    return f"$v\\geq {t:.4g}$"


def build_table() -> pd.DataFrame:
    rows = []
    rows.append(
        {
            "block": "baseline",
            "setting": "Pooled per-model (Tab.~\\ref{tab:calibration_by_model})",
            "gap_pp": 11.5,
            "f1": float("nan"),
        }
    )
    for shot, label in [("zero_shot", "Zero-shot"), ("few_shot", "Few-shot")]:
        thr = _load_threshold(shot).sort_values("vote_threshold")
        for _, r in thr.iterrows():
            rows.append(
                {
                    "block": label,
                    "setting": _format_vote_threshold(float(r["vote_threshold"])),
                    "gap_pp": float(r["gap_pp"]),
                    "f1": float(r["f1"]),
                }
            )
    for model, gap in [
        ("BERT (val-opt)", -1.1),
        ("RoBERTa (val-opt)", 0.0),
        ("ModernBERT (val-opt)", 0.0),
        ("LLaMA LoRA (val-opt)", 3.0),
        ("Phi-4 LoRA (val-opt)", 2.4),
        ("Qwen LoRA (val-opt)", -1.9),
    ]:
        rows.append({"block": "Fine-tuned", "setting": model, "gap_pp": gap, "f1": float("nan")})
    return pd.DataFrame(rows)


def write_latex(df: pd.DataFrame, out_dir: Path) -> None:
    vote = df[df["block"].isin(["baseline", "Zero-shot", "Few-shot"])]
    pseudo = df[df["block"] == "Fine-tuned"]

    def _vote_table(sub: pd.DataFrame, caption: str, label: str) -> list[str]:
        lines = [
            "% Auto-generated NIMBY mitigation summary",
            "\\begin{table}[t]",
            "\\centering",
            "\\footnotesize",
            "\\setlength{\\tabcolsep}{3pt}",
            "\\begin{tabularx}{\\columnwidth}{@{}>{\\raggedright\\arraybackslash}X rr@{}}",
            "\\toprule",
            "\\textbf{Setting} & \\textbf{Gap} & \\textbf{F1} \\\\",
            "\\midrule",
        ]
        for block, grp in sub.groupby("block", sort=False):
            block_title = block.capitalize() if block == "baseline" else block
            lines.append(f"\\multicolumn{{3}}{{@{{}}l@{{}}}}{{\\textit{{{block_title}}}}} \\\\")
            for _, r in grp.iterrows():
                f1 = "---" if pd.isna(r["f1"]) else f"{r['f1']:.3f}"
                lines.append(f"{r['setting']} & ${r['gap_pp']:+.1f}$ & {f1} \\\\")
        lines += [
            "\\bottomrule",
            "\\end{tabularx}",
            f"\\caption{{{caption}}}",
            f"\\label{{{label}}}",
            "\\end{table}",
        ]
        return lines

    def _table(sub: pd.DataFrame, caption: str, label: str, *, include_f1: bool = True) -> list[str]:
        ncol = 4 if include_f1 else 3
        tab_spec = "@{}llrr@{}" if include_f1 else "@{}llr@{}"
        lines = [
            "% Auto-generated NIMBY mitigation summary",
            "\\begin{table}[t]",
            "\\centering",
            "\\small",
            f"\\begin{{tabular}}{{{tab_spec}}}",
            "\\toprule",
        ]
        if include_f1:
            lines.append("\\textbf{Family} & \\textbf{Setting} & \\textbf{Gap (pp)} & F1 \\\\")
        else:
            lines.append("\\textbf{Family} & \\textbf{Setting} & \\textbf{Gap (pp)} \\\\")
        lines.append("\\midrule")
        for block, grp in sub.groupby("block", sort=False):
            lines.append(f"\\multicolumn{{{ncol}}}{{@{{}}l@{{}}}}{{\\textit{{{block}}}}} \\\\")
            for _, r in grp.iterrows():
                if include_f1:
                    f1 = "---" if pd.isna(r["f1"]) else f"{r['f1']:.3f}"
                    lines.append(f" & {r['setting']} & ${r['gap_pp']:+.1f}$ & {f1} \\\\")
                else:
                    lines.append(f" & {r['setting']} & ${r['gap_pp']:+.1f}$ \\\\")
        lines += [
            "\\bottomrule",
            "\\end{tabular}",
            f"\\caption{{{caption}}}",
            f"\\label{{{label}}}",
            "\\end{table}",
        ]
        return lines

    vote_tex = "\n".join(
        _vote_table(
            vote,
            "NIMBY gap vs.\\ six-model vote threshold $v$ (gold 2-of-3). Zero-shot NIMBY gap "
            "falls from $5.7$\\,pp at $v\\geq\\frac{1}{2}$ to $1.9$\\,pp at $v\\geq\\frac{2}{3}$.",
            "tab:nimby_vote_mitigation",
        )
    )
    pseudo_tex = "\n".join(
        _table(
            pseudo,
            "NIMBY gap after GPT pseudo-label training (val-opt; gold test split). "
            "Encoders $\\approx 0$\\,pp; LoRA LLMs $\\pm 3$\\,pp.",
            "tab:nimby_pseudolabel_mitigation",
            include_f1=False,
        )
    )
    (out_dir / "nimby_mitigation_vote.tex").write_text(vote_tex + "\n", encoding="utf-8")
    (out_dir / "nimby_mitigation_pseudolabel.tex").write_text(pseudo_tex + "\n", encoding="utf-8")
    # Legacy combined table (vote + pseudo) for backward compatibility
    combined = "\n".join(
        _table(
            df,
            "Mitigating NIMBY prevalence gap (vote thresholds and pseudo-label fine-tuning).",
            "tab:nimby_mitigation",
        )
    )
    (out_dir / "nimby_mitigation.tex").write_text(combined + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=REPO_ROOT / "output" / "audits" / "mitigation",
    )
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    df = build_table()
    df.to_csv(args.out_dir / "nimby_mitigation.csv", index=False)
    write_latex(df, args.out_dir)
    paper = REPO_ROOT / "paper_includes"
    paper.mkdir(parents=True, exist_ok=True)
    for name in (
        "nimby_mitigation_vote.tex",
        "nimby_mitigation_pseudolabel.tex",
        "nimby_mitigation.tex",
    ):
        (paper / name).write_text((args.out_dir / name).read_text(encoding="utf-8"))
    print(df.to_string(index=False))
    print(f"Wrote {args.out_dir}")


if __name__ == "__main__":
    main()
