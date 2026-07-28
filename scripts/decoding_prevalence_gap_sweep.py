#!/usr/bin/env python3
"""
Sweep decoding parameters and measure prevalence-gap impact on the gold set.

Answers Reviewer unx3: "What is the impact of different decoding parameters
on the prevalence gap?"

Default: local instruct models only (llama / qwen / phi4), zero-shot, stratified
gold subsample. Optional --allow-api for gpt4/gemini/grok (costs money).

Example (small pilot):
  .venv/bin/python scripts/decoding_prevalence_gap_sweep.py \\
    --models qwen --n-per-source 25 --temperatures 0.0 0.1 0.7 \\
    --seeds 0 1 --categories "not in my backyard" "provide a fact or claim"

Full offline analysis without re-inference is not possible (existing flags were
generated at temperature=0.1). This script re-queries models under new decode settings.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from utils import (  # noqa: E402
    COMMENT_TYPES,
    CRITIQUE_CATEGORIES,
    PERCEPTION_TYPES,
    RESPONSE_CATEGORIES,
    create_classification_prompt,
    create_output_row,
    extract_field,
    get_model_config,
)

SOURCES = ["reddit", "news", "meeting_minutes", "x"]
GOLD_TEXT = {
    "reddit": ("gold_standard/sampled_reddit_comments.csv", "Comment", "City"),
    "x": ("gold_standard/sampled_twitter_posts.csv", "Deidentified_text", "city"),
    "news": (
        "gold_standard/sampled_lexisnexis_news.csv",
        "Deidentified_paragraph_text",
        "city",
    ),
    "meeting_minutes": (
        "gold_standard/sampled_meeting_minutes.csv",
        "Deidentified_paragraph",
        "city",
    ),
}
LOCAL_MODELS = {"llama", "qwen", "phi4", "gemma3"}
API_MODELS = {"gpt4", "gemini", "grok"}

PRIORITY_DEFAULT = [
    "not in my backyard",
    "provide a fact or claim",
    "express their opinion",
    "harmful generalization",
]

SOFT_TO_FLAG = {
    "ask a genuine question": "Comment_ask a genuine question",
    "ask a rhetorical question": "Comment_ask a rhetorical question",
    "provide a fact or claim": "Comment_provide a fact or claim",
    "provide an observation": "Comment_provide an observation",
    "express their opinion": "Comment_express their opinion",
    "express others opinions": "Comment_express others opinions",
    "money aid allocation": "Critique_money aid allocation",
    "government critique": "Critique_government critique",
    "societal critique": "Critique_societal critique",
    "solutions/interventions": "Response_solutions/interventions",
    "personal interaction": "Perception_personal interaction",
    "media portrayal": "Perception_media portrayal",
    "not in my backyard": "Perception_not in my backyard",
    "harmful generalization": "Perception_harmful generalization",
    "deserving/undeserving": "Perception_deserving/undeserving",
    "racist": "Racist_Flag",
}


def _parse_bracketed_list(field: str) -> List[str]:
    if not field or str(field).strip().lower() in {
        "[]",
        "",
        "none",
        "n/a",
        "-",
        "no categories",
        "none applicable",
    }:
        return []
    field = str(field).strip()
    if field.startswith("[") and field.endswith("]"):
        field = field[1:-1]
    return [v.strip() for v in field.split(",") if v.strip()]


def _parse_bracketed_single(field: str) -> str:
    items = _parse_bracketed_list(field)
    return items[0] if items else ""


def _raw_to_flags(comment: str, city: str, output: str) -> Dict:
    comment_text = ", ".join(_parse_bracketed_list(extract_field(output, "Comment Type")))
    critique_text = ", ".join(
        _parse_bracketed_list(extract_field(output, "Critique Category"))
    )
    response_text = ", ".join(
        _parse_bracketed_list(extract_field(output, "Response Category"))
    )
    perception_text = ", ".join(
        _parse_bracketed_list(extract_field(output, "Perception Type"))
    )
    racist_value = _parse_bracketed_single(extract_field(output, "racist")).lower()
    if racist_value.startswith("racist:"):
        racist_value = racist_value.replace("racist:", "").strip()
    racist_flag = 1 if racist_value in {"yes", "true", "1"} else 0
    reasoning = extract_field(output, "Reasoning") or "No reasoning provided."
    return create_output_row(
        comment=comment,
        city=city,
        comment_text=comment_text,
        critique_text=critique_text,
        response_text=response_text,
        perception_text=perception_text,
        racist_flag=racist_flag,
        reasoning=reasoning,
        raw_response=output,
    )


def load_gold_sample(n_per_source: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for source in SOURCES:
        soft = pd.read_csv(
            REPO_ROOT / "output" / "annotation" / "soft_labels" / f"{source}_soft_labels.csv"
        )
        path, text_col, city_col = GOLD_TEXT[source]
        texts = pd.read_csv(REPO_ROOT / path, low_memory=False)
        n = min(len(soft), len(texts))
        soft = soft.iloc[:n].reset_index(drop=True)
        texts = texts.iloc[:n].reset_index(drop=True)
        k = min(n_per_source, n)
        idxs = rng.choice(n, size=k, replace=False)
        for i in idxs:
            city = texts.loc[i, city_col] if city_col in texts.columns else source
            rows.append(
                {
                    "source": source,
                    "item_idx": int(i),
                    "Comment": str(texts.loc[i, text_col]),
                    "City": str(city),
                    **{c: float(soft.loc[i, c]) if c in soft.columns else 0.0 for c in SOFT_TO_FLAG},
                }
            )
    return pd.DataFrame(rows)


def prevalence_gap_pp(gold: np.ndarray, pred: np.ndarray) -> float:
    return 100.0 * (float(pred.mean()) - float(gold.mean()))


def load_local_model(model_key: str):
    """Load local causal LM without HF pipeline (avoids second .to() / MPS OOM)."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    cfg = get_model_config(model_key)
    model_id = cfg["model_id"]
    tok = AutoTokenizer.from_pretrained(model_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    if torch.cuda.is_available():
        device = "cuda"
        dtype = torch.float16
    elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        device = "mps"
        dtype = torch.float16
    else:
        device = "cpu"
        dtype = torch.float32

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=dtype,
        low_cpu_mem_usage=True,
    )
    model.to(device)
    model.eval()
    return model, tok, cfg, device


def generate_local(
    model,
    tokenizer,
    prompt: str,
    *,
    temperature: float,
    top_p: float,
    max_new_tokens: int,
    seed: int,
    device: str,
) -> str:
    import torch

    torch.manual_seed(seed)
    if device == "cuda":
        torch.cuda.manual_seed_all(seed)

    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    do_sample = temperature > 1e-8
    gen_kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
        "pad_token_id": tokenizer.eos_token_id,
    }
    if do_sample:
        gen_kwargs["temperature"] = max(float(temperature), 1e-5)
        gen_kwargs["top_p"] = float(top_p)

    with torch.no_grad():
        out_ids = model.generate(**inputs, **gen_kwargs)
    # Decode only newly generated tokens
    new_tokens = out_ids[0, inputs["input_ids"].shape[1] :]
    out = tokenizer.decode(new_tokens, skip_special_tokens=True)
    idx = out.find("Analysis:")
    if idx != -1:
        out = out[idx + len("Analysis:") :].strip()
    return out


def generate_api(model_key: str, prompt: str, temperature: float, max_tokens: int) -> str:
    """Thin wrapper; OpenAI path supports temperature; others best-effort."""
    from utils import call_api_llm, get_model_config
    import os
    import time
    import random

    cfg = get_model_config(model_key)
    api = cfg.get("api")
    api_key = os.environ.get(cfg.get("api_key_env", ""), None)
    model_id = cfg.get("model_id")
    if api == "openai":
        from openai import OpenAI

        client = OpenAI(api_key=api_key)
        resp = client.chat.completions.create(
            model=model_id,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return resp.choices[0].message.content
    if api == "grok":
        import requests

        endpoint = "https://api.x.ai/v1/chat/completions"
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        data = {
            "model": model_id,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": False,
        }
        resp = requests.post(endpoint, headers=headers, json=data, timeout=120)
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]
    # Gemini SDK historically ignores temp in our wrapper; fall back to default call
    return call_api_llm(prompt, model_key, max_tokens=max_tokens)


def run_setting(
    sample: pd.DataFrame,
    *,
    model_key: str,
    temperature: float,
    top_p: float,
    seed: int,
    model=None,
    tokenizer=None,
    device: str = "cpu",
    max_new_tokens: int = 500,
) -> pd.DataFrame:
    rows = []
    for _, item in sample.iterrows():
        prompt = create_classification_prompt(
            item["Comment"], content_type=item["source"], few_shot_text="none"
        )
        if model_key in API_MODELS:
            raw = generate_api(model_key, prompt, temperature, max_new_tokens)
        else:
            raw = generate_local(
                model,
                tokenizer,
                prompt,
                temperature=temperature,
                top_p=top_p,
                max_new_tokens=max_new_tokens,
                seed=seed,
                device=device,
            )
        flags = _raw_to_flags(item["Comment"], item["City"], raw)
        row = {
            "source": item["source"],
            "item_idx": item["item_idx"],
            "model": model_key,
            "temperature": temperature,
            "top_p": top_p,
            "seed": seed,
        }
        for soft_col, flag_col in SOFT_TO_FLAG.items():
            gold = 1 if float(item[soft_col]) >= 2 / 3 - 1e-9 else 0
            pred = int(flags.get(flag_col, 0))
            row[f"gold__{soft_col}"] = gold
            row[f"pred__{soft_col}"] = pred
        rows.append(row)
    return pd.DataFrame(rows)


def summarize_gaps(long_df: pd.DataFrame, categories: List[str]) -> pd.DataFrame:
    rows = []
    for (model, temp, top_p, seed), sub in long_df.groupby(
        ["model", "temperature", "top_p", "seed"]
    ):
        for cat in categories:
            g = sub[f"gold__{cat}"].values.astype(int)
            p = sub[f"pred__{cat}"].values.astype(int)
            rows.append(
                {
                    "model": model,
                    "temperature": temp,
                    "top_p": top_p,
                    "seed": seed,
                    "category": cat,
                    "n": len(g),
                    "pi_ref": 100.0 * g.mean(),
                    "pi_model": 100.0 * p.mean(),
                    "gap_pp": prevalence_gap_pp(g, p),
                    "gold_positives": int(g.sum()),
                }
            )
    return pd.DataFrame(rows)


def write_latex(summary: pd.DataFrame, out_dir: Path) -> None:
    # Aggregate over seeds: mean ± std of gap
    agg = (
        summary.groupby(["model", "temperature", "top_p", "category"], as_index=False)
        .agg(gap_mean=("gap_pp", "mean"), gap_std=("gap_pp", "std"), n=("n", "first"))
        .sort_values(["category", "model", "temperature"])
    )
    lines = [
        "% Auto-generated decoding prevalence-gap sweep",
        "\\begin{table}[t]",
        "\\centering",
        "\\footnotesize",
        "\\begin{tabular}{@{}l l r r r@{}}",
        "\\toprule",
        "\\textbf{Model} & \\textbf{Category} & Temp. & Gap (pp) & $n$ \\\\",
        "\\midrule",
    ]
    for _, r in agg.iterrows():
        std = 0.0 if pd.isna(r["gap_std"]) else float(r["gap_std"])
        lines.append(
            f"{r['model']} & {r['category']} & {r['temperature']:.2f} & "
            f"${r['gap_mean']:+.1f}\\pm{std:.1f}$ & {int(r['n'])} \\\\"
        )
    lines += [
        "\\bottomrule",
        "\\end{tabular}",
        "\\caption{Prevalence gap ($\\hat{\\pi}-\\pi_{\\mathrm{ref}}$, pp) under different decoding "
        "temperatures on a stratified gold subsample (zero-shot). Mean$\\pm$std over random seeds. "
        "Paper main results used temperature $0.1$.}",
        "\\label{tab:decoding_prevalence_gap}",
        "\\end{table}",
    ]
    (out_dir / "decoding_prevalence_gap.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    agg.to_csv(out_dir / "decoding_gap_by_temp.csv", index=False)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["qwen"])
    ap.add_argument("--temperatures", nargs="+", type=float, default=[0.0, 0.1, 0.7])
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--seeds", nargs="+", type=int, default=[0])
    ap.add_argument("--n-per-source", type=int, default=25)
    ap.add_argument("--sample-seed", type=int, default=42)
    ap.add_argument("--categories", nargs="+", default=PRIORITY_DEFAULT)
    ap.add_argument("--allow-api", action="store_true")
    ap.add_argument("--max-new-tokens", type=int, default=400)
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=REPO_ROOT / "output" / "openreview_artifacts" / "decoding_sweep",
    )
    ap.add_argument(
        "--dry-run-sample-only",
        action="store_true",
        help="Only write the gold subsample (no model calls).",
    )
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    for m in args.models:
        if m in API_MODELS and not args.allow_api:
            raise SystemExit(
                f"Model {m} is an API model; pass --allow-api to run (costs money)."
            )

    sample = load_gold_sample(args.n_per_source, args.sample_seed)
    sample.to_csv(args.out_dir / "gold_subsample.csv", index=False)
    meta = {
        "n_items": len(sample),
        "n_per_source": args.n_per_source,
        "models": args.models,
        "temperatures": args.temperatures,
        "top_p": args.top_p,
        "seeds": args.seeds,
        "categories": args.categories,
        "note": "Main paper used temperature=0.1 elsewhere.",
    }
    (args.out_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Sampled {len(sample)} gold items -> {args.out_dir / 'gold_subsample.csv'}")

    if args.dry_run_sample_only:
        print("Dry run only; exiting.")
        return

    all_long = []
    for model_key in args.models:
        model = tokenizer = None
        device = "cpu"
        if model_key in LOCAL_MODELS:
            print(f"Loading local model {model_key}...")
            model, tokenizer, _, device = load_local_model(model_key)
            print(f"Loaded on {device}")
        for temp in args.temperatures:
            for seed in args.seeds:
                tag = f"{model_key}_T{temp}_seed{seed}"
                print(f"Running {tag} on {len(sample)} items...")
                long_df = run_setting(
                    sample,
                    model_key=model_key,
                    temperature=temp,
                    top_p=args.top_p,
                    seed=seed,
                    model=model,
                    tokenizer=tokenizer,
                    device=device,
                    max_new_tokens=args.max_new_tokens,
                )
                long_path = args.out_dir / f"preds_{tag}.csv"
                long_df.to_csv(long_path, index=False)
                all_long.append(long_df)

    long_all = pd.concat(all_long, ignore_index=True)
    long_all.to_csv(args.out_dir / "predictions_long.csv", index=False)
    summary = summarize_gaps(long_all, args.categories)
    summary.to_csv(args.out_dir / "prevalence_gap_by_decode.csv", index=False)
    write_latex(summary, args.out_dir)

    # Quick baseline delta vs T=0.1
    base = summary[np.isclose(summary["temperature"], 0.1)]
    print("\n=== Gaps at each temperature (mean over seeds) ===")
    print(
        summary.groupby(["model", "temperature", "category"])["gap_pp"]
        .mean()
        .unstack("category")
        .to_string()
    )
    if not base.empty:
        piv = summary.pivot_table(
            index=["model", "category"], columns="temperature", values="gap_pp", aggfunc="mean"
        )
        if 0.1 in piv.columns:
            for t in args.temperatures:
                if t == 0.1 or t not in piv.columns:
                    continue
                piv[f"delta_vs_T0.1@{t}"] = piv[t] - piv[0.1]
            piv.to_csv(args.out_dir / "gap_delta_vs_baseline_T0.1.csv")
            print("\nDelta vs paper baseline T=0.1:")
            print(piv.to_string())
    print(f"\nWrote outputs to {args.out_dir}")


if __name__ == "__main__":
    main()
