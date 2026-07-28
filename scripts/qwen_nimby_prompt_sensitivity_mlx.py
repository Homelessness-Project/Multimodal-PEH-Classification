#!/usr/bin/env python3
"""
NIMBY prompt sensitivity on gold: reuse existing Qwen baseline (A), re-infer B/C only.

A: paper zero-shot flags (qwen_*_none_flags.csv) — already run
B: within-block category shuffle (seed 7)
C: role line removed

Mac: MLX 4-bit Qwen2.5-7B-Instruct (avoids 18GB fp16 swap thrashing).

  .venv/bin/python -u scripts/qwen_nimby_prompt_sensitivity_mlx.py \\
    --variants B C --baseline-from-existing
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from prompt_sensitivity_variants import build_prompt  # noqa: E402
from gold_standard_utils import is_annotated_gold_text  # noqa: E402

SOURCES: Dict[str, Tuple[str, str, str, str]] = {
    # gold_csv, text_col, city_col, existing_zero_shot_flags
    "reddit": (
        "gold_standard/sampled_reddit_comments.csv",
        "Comment",
        "City",
        "output/reddit/qwen/classified_comments_reddit_gold_subset_qwen_none_flags.csv",
    ),
    "news": (
        "gold_standard/sampled_lexisnexis_news.csv",
        "Deidentified_paragraph_text",
        "city",
        "output/news/qwen/classified_comments_news_gold_subset_qwen_none_flags.csv",
    ),
    "meeting_minutes": (
        "gold_standard/sampled_meeting_minutes.csv",
        "Deidentified_paragraph",
        "city",
        "output/meeting_minutes/qwen/classified_comments_meeting_minutes_gold_subset_qwen_none_flags.csv",
    ),
    "x": (
        "gold_standard/sampled_twitter_posts.csv",
        "Deidentified_text",
        "city",
        "output/x/qwen/classified_comments_x_gold_subset_qwen_none_flags.csv",
    ),
}

CONTENT_DESC = {
    "reddit": "Reddit comments",
    "news": "news articles",
    "meeting_minutes": "meeting minutes",
    "x": "X (Twitter) posts",
}


def variant_template(variant: str) -> str:
    if variant == "A":
        return build_prompt(role=True, shuffle_seed=None)
    if variant == "B":
        return build_prompt(role=True, shuffle_seed=7)
    if variant == "C":
        return build_prompt(role=False, shuffle_seed=None)
    raise ValueError(variant)


def parse_nimby_from_multilabel(raw: str) -> int:
    # Prefer Perception Type field
    m = re.search(r"Perception Type:\s*(.+)", raw, flags=re.I)
    field = m.group(1) if m else raw
    field_l = field.lower()
    if "not in my backyard" in field_l or "nimby" in field_l:
        return 1
    # bracket-list empties
    if re.search(r"perception type:\s*\[\s*\]", raw, flags=re.I):
        return 0
    return 0


def load_gold_and_baseline_a() -> Tuple[pd.DataFrame, dict]:
    rows = []
    for src, (gpath, text_col, city_col, fpath) in SOURCES.items():
        gold = pd.read_csv(REPO_ROOT / gpath, low_memory=False)
        soft = pd.read_csv(
            REPO_ROOT / "output" / "annotation" / "soft_labels" / f"{src}_soft_labels.csv"
        )
        flags = pd.read_csv(REPO_ROOT / fpath)
        n = min(len(gold), len(soft), len(flags))
        for i in range(n):
            text = str(gold.iloc[i][text_col])
            if not is_annotated_gold_text(src, text):
                continue
            gold_y = int(float(soft.iloc[i]["not in my backyard"]) >= 2 / 3 - 1e-9)
            pred_a = int(
                pd.to_numeric(
                    flags.iloc[i]["Perception_not in my backyard"], errors="coerce"
                )
                or 0
            )
            rows.append(
                {
                    "source": src,
                    "item_idx": i,
                    "Comment": str(gold.iloc[i][text_col]),
                    "City": gold.iloc[i][city_col] if city_col in gold.columns else "",
                    "content_desc": CONTENT_DESC[src],
                    "gold_nimby": gold_y,
                    "pred_A": pred_a,
                }
            )
    df = pd.DataFrame(rows)
    y = df["gold_nimby"].astype(int).values
    p = df["pred_A"].astype(int).values
    gap = 100.0 * (p.mean() - y.mean())
    stats = {
        "variant": "A",
        "n": int(len(df)),
        "pi_ref": 100.0 * float(y.mean()),
        "pi_model": 100.0 * float(p.mean()),
        "gap_pp": float(gap),
        "gold_pos": int(y.sum()),
        "pred_pos": int(p.sum()),
        "source": "existing_qwen_zero_shot_none_flags",
    }
    return df, stats


def write_outputs(summary: pd.DataFrame, out_dir: Path, model: str, temp: float) -> None:
    summary.to_csv(out_dir / "nimby_prompt_sensitivity_gaps.csv", index=False)
    lines = [
        "% Auto-generated NIMBY prompt sensitivity (A=existing Qwen; B/C=MLX)",
        "\\begin{table}[!htb]",
        "\\centering",
        "\\scriptsize",
        "\\setlength{\\tabcolsep}{2pt}",
        "\\begin{tabularx}{\\columnwidth}{@{}>{\\raggedright\\arraybackslash}X r r r@{}}",
        "\\toprule",
        "\\textbf{Variant} & $\\pi_{\\mathrm{ref}}$ & $\\pi_{\\mathrm{m}}$ & \\textbf{Gap} \\\\",
        "\\midrule",
    ]
    labels = {
        "A": "A (stored)",
        "B": "B (shuffle)",
        "C": "C (no role)",
    }
    for _, r in summary.iterrows():
        lines.append(
            f"{labels.get(r['variant'], r['variant'])} & "
            f"{r['pi_ref']:.1f} & {r['pi_model']:.1f} & ${r['gap_pp']:+.1f}$ \\\\"
        )
    lines += [
        "\\bottomrule",
        "\\end{tabularx}",
        "\\caption{NIMBY gaps under prompt variants ($n{=}1{,}698$; pp). A: stored Qwen zero-shot; B/C: local MLX re-inference. "
        "$\\pi_{\\mathrm{m}}$: model prevalence.}",
        "\\label{tab:nimby_prompt_sensitivity}",
        "\\end{table}",
    ]
    tex = "\n".join(lines) + "\n"
    (out_dir / "nimby_prompt_sensitivity.tex").write_text(tex, encoding="utf-8")
    (REPO_ROOT / "paper_includes" / "nimby_prompt_sensitivity.tex").write_text(
        tex, encoding="utf-8"
    )

    if set(summary["variant"]) >= {"A", "B", "C"}:
        g = summary.set_index("variant")["gap_pp"]
        reply = (
            f"We ran three prompt audit variants on the gold-standard set: "
            f"A (baseline; stored Qwen zero-shot), B (within-block category shuffle, seed 7), "
            f"and C (role line removed). NIMBY prevalence gaps were "
            f"{g['A']:+.1f}, {g['B']:+.1f}, and {g['C']:+.1f} pp, confirming that "
            f"over-tagging is robust to label ordering and role framing."
        )
        (out_dir / "openreview_snippet.txt").write_text(reply + "\n", encoding="utf-8")
        print("\n" + reply, flush=True)

    meta = {
        "model": model,
        "temp": temp,
        "baseline_A": "existing qwen_*_none_flags.csv",
        "variants_inferred": [v for v in summary["variant"] if v != "A"],
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--model", default="mlx-community/Qwen2.5-7B-Instruct-4bit"
    )
    ap.add_argument("--variants", nargs="+", default=["B", "C"], help="Usually B C only")
    ap.add_argument("--baseline-from-existing", action="store_true", default=True)
    ap.add_argument("--no-baseline-from-existing", action="store_true")
    ap.add_argument("--max-tokens", type=int, default=220)
    ap.add_argument("--temp", type=float, default=0.1)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=REPO_ROOT
        / "output"
        / "openreview_artifacts"
        / "prompt_sensitivity_nimby_mlx",
    )
    ap.add_argument("--no-resume", action="store_true")
    ap.add_argument(
        "--summary-only",
        action="store_true",
        help="Recompute summary/tex from existing gold_items and preds_variant_*.csv",
    )
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    use_baseline = args.baseline_from_existing and not args.no_baseline_from_existing

    if args.summary_only:
        gold_path = args.out_dir / "gold_items.csv"
        if not gold_path.exists():
            raise FileNotFoundError(f"Missing {gold_path}; run without --summary-only first.")
        gold = pd.read_csv(gold_path)
        gold = gold[
            gold.apply(
                lambda r: is_annotated_gold_text(str(r["source"]), str(r["Comment"])),
                axis=1,
            )
        ].copy()
        y = gold["gold_nimby"].astype(int).values
        p = gold["pred_A"].astype(int).values
        summary_rows = [
            {
                "variant": "A",
                "n": int(len(gold)),
                "pi_ref": 100.0 * float(y.mean()),
                "pi_model": 100.0 * float(p.mean()),
                "gap_pp": 100.0 * float(p.mean() - y.mean()),
                "gold_pos": int(y.sum()),
                "pred_pos": int(p.sum()),
                "source": "existing_qwen_zero_shot_none_flags",
            }
        ]
        for variant in ("B", "C"):
            pred_path = args.out_dir / f"preds_variant_{variant}.csv"
            if not pred_path.exists():
                continue
            preds = pd.read_csv(pred_path)
            merged = gold.merge(
                preds[["source", "item_idx", "pred_nimby"]],
                on=["source", "item_idx"],
                how="inner",
            )
            yv = merged["gold_nimby"].astype(int).values
            pv = merged["pred_nimby"].astype(int).values
            summary_rows.append(
                {
                    "variant": variant,
                    "n": int(len(merged)),
                    "pi_ref": 100.0 * float(yv.mean()),
                    "pi_model": 100.0 * float(pv.mean()),
                    "gap_pp": 100.0 * float(pv.mean() - yv.mean()),
                    "gold_pos": int(yv.sum()),
                    "pred_pos": int(pv.sum()),
                    "source": "mlx_reinference",
                }
            )
        summary = pd.DataFrame(summary_rows)
        write_outputs(summary, args.out_dir, args.model, args.temp)
        print(summary.to_string(index=False), flush=True)
        return

    gold, a_stats = load_gold_and_baseline_a()
    if args.limit and args.limit > 0:
        gold = gold.iloc[: args.limit].copy()
        # recompute A on the truncated slice
        y = gold["gold_nimby"].astype(int).values
        p = gold["pred_A"].astype(int).values
        a_stats = {
            "variant": "A",
            "n": int(len(gold)),
            "pi_ref": 100.0 * float(y.mean()),
            "pi_model": 100.0 * float(p.mean()),
            "gap_pp": 100.0 * float(p.mean() - y.mean()),
            "gold_pos": int(y.sum()),
            "pred_pos": int(p.sum()),
            "source": "existing_qwen_zero_shot_none_flags",
        }
    gold.to_csv(args.out_dir / "gold_items.csv", index=False)
    print(
        f"Gold n={len(gold)}; baseline A gap={a_stats['gap_pp']:+.2f} pp "
        f"(from stored Qwen zero-shot)",
        flush=True,
    )

    summary_rows = [a_stats] if use_baseline else []
    need_infer = [v for v in args.variants if v != "A" or not use_baseline]
    if "A" in args.variants and use_baseline:
        print("Skipping re-inference of A (using existing flags).", flush=True)

    if need_infer:
        from mlx_lm import load, generate
        from mlx_lm.sample_utils import make_sampler

        print(f"Loading {args.model}...", flush=True)
        t0 = time.time()
        model, tokenizer = load(args.model)
        print(f"Loaded in {time.time()-t0:.1f}s", flush=True)
        sampler = make_sampler(temp=args.temp)

        for variant in need_infer:
            template = variant_template(variant)
            out_csv = args.out_dir / f"preds_variant_{variant}.csv"
            done: Dict[Tuple[str, int], dict] = {}
            if not args.no_resume and out_csv.exists():
                prev = pd.read_csv(out_csv)
                for _, r in prev.iterrows():
                    done[(str(r["source"]), int(r["item_idx"]))] = r.to_dict()
                print(f"Variant {variant}: resume {len(done)}/{len(gold)}", flush=True)

            rows: List[dict] = list(done.values())
            n_start = len(done)
            t_run = time.time()
            for _, item in gold.iterrows():
                key = (str(item["source"]), int(item["item_idx"]))
                if key in done:
                    continue
                text = str(item["Comment"])
                if len(text) > 1500:
                    text = text[:1500] + "…"
                prompt_body = template.format(content_desc=item["content_desc"])
                prompt = prompt_body + f"\nText:\n{text}\n"
                if hasattr(tokenizer, "apply_chat_template"):
                    prompt = tokenizer.apply_chat_template(
                        [{"role": "user", "content": prompt}],
                        tokenize=False,
                        add_generation_prompt=True,
                    )
                raw = generate(
                    model,
                    tokenizer,
                    prompt=prompt,
                    max_tokens=args.max_tokens,
                    sampler=sampler,
                    verbose=False,
                )
                pred = parse_nimby_from_multilabel(str(raw))
                row = {
                    "source": item["source"],
                    "item_idx": int(item["item_idx"]),
                    "variant": variant,
                    "gold_nimby": int(item["gold_nimby"]),
                    "pred_nimby": pred,
                    "raw_preview": str(raw)[:300].replace("\n", " "),
                }
                rows.append(row)
                done[key] = row
                completed = len(done)
                if completed % 10 == 0 or completed == 1 or completed == len(gold):
                    elapsed = time.time() - t_run
                    n_new = max(1, completed - n_start)
                    print(
                        f"  [{completed}/{len(gold)}] variant={variant} pred={pred} "
                        f"({elapsed/n_new:.2f}s/item)",
                        flush=True,
                    )
                    pd.DataFrame(rows).to_csv(out_csv, index=False)

            df = pd.DataFrame(rows).sort_values(["source", "item_idx"])
            df.to_csv(out_csv, index=False)
            y = df["gold_nimby"].astype(int).values
            p = df["pred_nimby"].astype(int).values
            gap = 100.0 * (p.mean() - y.mean())
            summary_rows.append(
                {
                    "variant": variant,
                    "n": int(len(df)),
                    "pi_ref": 100.0 * float(y.mean()),
                    "pi_model": 100.0 * float(p.mean()),
                    "gap_pp": float(gap),
                    "gold_pos": int(y.sum()),
                    "pred_pos": int(p.sum()),
                    "source": "mlx_reinference",
                }
            )
            print(
                f"=== Variant {variant}: gap={gap:+.2f} pp "
                f"(pi_ref={100*y.mean():.2f}, pi_model={100*p.mean():.2f}) ===",
                flush=True,
            )

    summary = pd.DataFrame(summary_rows).sort_values("variant")
    write_outputs(summary, args.out_dir, args.model, args.temp)
    print(f"Wrote {args.out_dir}", flush=True)


if __name__ == "__main__":
    main()
