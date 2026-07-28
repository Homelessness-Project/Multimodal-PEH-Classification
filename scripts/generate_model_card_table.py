#!/usr/bin/env python3
"""Emit a model-card disclosure table (IDs + decoding params) as LaTeX/CSV."""

from __future__ import annotations

import argparse
import csv
from datetime import date
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))
from utils import get_model_config  # noqa: E402

MODELS = [
    ("gpt4", "GPT", "API", "gpt-4.1"),
    ("gemini", "Gemini", "API", "gemini-2.5-pro"),
    ("grok", "Grok", "API", "grok-4-latest"),
    ("llama", "LLaMA", "local", "Llama-3.2-3B-Instruct"),
    ("qwen", "Qwen", "local", "Qwen2.5-7B-Instruct"),
    ("phi4", "Phi-4", "local", "Phi-4-mini-instruct"),
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=REPO_ROOT / "output" / "openreview_artifacts" / "model_card",
    )
    ap.add_argument("--snapshot-date", default="2025-05--2025-08 collection window")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for key, display, backend, short_id in MODELS:
        cfg = get_model_config(key)
        rows.append(
            {
                "display_name": display,
                "backend": backend,
                "model_id": cfg.get("model_id", ""),
                "short_id": short_id,
                "temperature": cfg.get("temperature", 0.1 if backend == "API" else cfg.get("temperature", "")),
                "top_p": cfg.get("top_p", "API default" if backend == "API" else cfg.get("top_p", "")),
                "max_new_tokens": cfg.get("max_new_tokens", ""),
                "snapshot_note": args.snapshot_date,
            }
        )

    csv_path = args.out_dir / "model_card.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    lines = [
        "% Auto-generated model-card disclosure",
        "\\begin{table}[t]",
        "\\centering",
        "\\footnotesize",
        "\\setlength{\\tabcolsep}{3pt}",
        "\\begin{tabularx}{\\columnwidth}{@{}l l >{\\raggedright\\arraybackslash}X r@{}}",
        "\\toprule",
        "\\textbf{Model} & \\textbf{Be.} & \\textbf{ID} & $T$ \\\\",
        "\\midrule",
    ]
    for r in rows:
        temp = r["temperature"] if r["temperature"] != "" else "0.1"
        lines.append(
            f"{r['display_name']} & {r['backend']} & \\texttt{{{r['short_id']}}} & {temp} \\\\"
        )
    lines += [
        "\\bottomrule",
        "\\end{tabularx}",
        "\\caption{Model IDs and decode settings ($t_{\\max}{=}500$, $T{=}0.1$; local: top-$p{=}0.95$, rep.\\ penalty $1.1$). "
        "\\textbf{Be.}: API or local HF. All models inferred May--Aug 2025.}",
        "\\label{tab:model_card}",
        "\\end{table}",
    ]
    tex_path = args.out_dir / "model_card.tex"
    tex_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {csv_path}")
    print(f"Wrote {tex_path}")
    print(f"Generated on {date.today().isoformat()}")


if __name__ == "__main__":
    main()
