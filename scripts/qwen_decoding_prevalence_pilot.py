#!/usr/bin/env python3
"""
Qwen-only pilot: prevalence gap vs decoding temperature on a gold subsample.

Answers Reviewer unx3 (cheap empirical option B):
  T ∈ {0.0, 0.1, 0.7} at fixed top-p, zero-shot Qwen2.5-7B-Instruct.

Example (≈48 items, 3 temps = 144 gens; overnight-friendly on MPS):
  PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0 .venv/bin/python -u \\
    scripts/qwen_decoding_prevalence_pilot.py \\
    --n-per-source 12 --temperatures 0.0 0.1 0.7 --max-new-tokens 160

Quick smoke (8 items × 3 temps):
  .venv/bin/python -u scripts/qwen_decoding_prevalence_pilot.py \\
    --n-per-source 2 --max-new-tokens 120 --device mps
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from decoding_prevalence_gap_sweep import (  # noqa: E402
    SOFT_TO_FLAG,
    _raw_to_flags,
    load_gold_sample,
    summarize_gaps,
    write_latex,
)
from utils import create_classification_prompt, get_model_config  # noqa: E402

MODEL_KEY = "qwen"


def pick_device(requested: str) -> str:
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_qwen(device: str):
    cfg = get_model_config(MODEL_KEY)
    model_id = cfg["model_id"]
    print(f"Loading {model_id} on {device}...", flush=True)
    tok = AutoTokenizer.from_pretrained(model_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    dtype = torch.float16 if device in {"cuda", "mps"} else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=dtype,
        low_cpu_mem_usage=True,
    )
    print(f"Moving weights to {device}...", flush=True)
    model.to(device)
    model.eval()
    print(f"Ready on {device}.", flush=True)
    return model, tok, cfg


def generate_qwen(
    model,
    tok,
    prompt: str,
    *,
    temperature: float,
    top_p: float,
    max_new_tokens: int,
    seed: int,
    device: str,
    repetition_penalty: float,
) -> str:
    torch.manual_seed(seed)
    if device == "cuda":
        torch.cuda.manual_seed_all(seed)

    # Prefer chat template so Instruct stops cleanly
    if hasattr(tok, "apply_chat_template"):
        messages = [{"role": "user", "content": prompt}]
        text = tok.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    else:
        text = prompt

    inputs = tok(text, return_tensors="pt", truncation=True, max_length=3072)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    do_sample = float(temperature) > 1e-8
    gen_kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
        "pad_token_id": tok.eos_token_id,
        "eos_token_id": tok.eos_token_id,
        "repetition_penalty": float(repetition_penalty),
    }
    if do_sample:
        gen_kwargs["temperature"] = max(float(temperature), 1e-5)
        gen_kwargs["top_p"] = float(top_p)

    with torch.no_grad():
        out_ids = model.generate(**inputs, **gen_kwargs)
    new_tokens = out_ids[0, inputs["input_ids"].shape[1] :]
    out = tok.decode(new_tokens, skip_special_tokens=True).strip()
    idx = out.find("Analysis:")
    if idx != -1:
        out = out[idx + len("Analysis:") :].strip()
    return out


def run_temp(
    sample: pd.DataFrame,
    *,
    model,
    tok,
    temperature: float,
    top_p: float,
    seed: int,
    device: str,
    max_new_tokens: int,
    repetition_penalty: float,
    out_csv: Path,
    resume: bool,
) -> pd.DataFrame:
    done: Dict[tuple, dict] = {}
    if resume and out_csv.exists():
        prev = pd.read_csv(out_csv)
        for _, r in prev.iterrows():
            done[(str(r["source"]), int(r["item_idx"]))] = r.to_dict()
        print(f"  resume: {len(done)}/{len(sample)} already in {out_csv.name}", flush=True)

    rows: List[dict] = []
    t0 = time.time()
    for i, (_, item) in enumerate(sample.iterrows(), start=1):
        key = (str(item["source"]), int(item["item_idx"]))
        if key in done:
            rows.append(done[key])
            continue

        comment = str(item["Comment"])
        if len(comment) > 1200:
            comment = comment[:1200] + "…"
        prompt = create_classification_prompt(
            comment, content_type=item["source"], few_shot_text="none"
        )
        raw = generate_qwen(
            model,
            tok,
            prompt,
            temperature=temperature,
            top_p=top_p,
            max_new_tokens=max_new_tokens,
            seed=seed,
            device=device,
            repetition_penalty=repetition_penalty,
        )
        flags = _raw_to_flags(comment, str(item["City"]), raw)
        row = {
            "source": item["source"],
            "item_idx": int(item["item_idx"]),
            "model": MODEL_KEY,
            "temperature": temperature,
            "top_p": top_p,
            "seed": seed,
            "raw_preview": raw[:400].replace("\n", " "),
        }
        for soft_col, flag_col in SOFT_TO_FLAG.items():
            gold = 1 if float(item[soft_col]) >= 2 / 3 - 1e-9 else 0
            pred = int(flags.get(flag_col, 0))
            row[f"gold__{soft_col}"] = gold
            row[f"pred__{soft_col}"] = pred
        rows.append(row)

        # Checkpoint every item so kills are resumable
        pd.DataFrame(rows).to_csv(out_csv, index=False)
        elapsed = time.time() - t0
        print(
            f"  [{i}/{len(sample)}] T={temperature} {item['source']}#{item['item_idx']} "
            f"({elapsed/i:.1f}s/item)",
            flush=True,
        )
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-per-source", type=int, default=12)
    ap.add_argument("--sample-seed", type=int, default=42)
    ap.add_argument("--temperatures", nargs="+", type=float, default=[0.0, 0.1, 0.7])
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-new-tokens", type=int, default=160)
    ap.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    ap.add_argument(
        "--categories",
        nargs="+",
        default=["not in my backyard", "provide a fact or claim"],
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=REPO_ROOT
        / "output"
        / "audits"
        / "decoding_sweep_qwen",
    )
    ap.add_argument("--no-resume", action="store_true")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    device = pick_device(args.device)
    sample = load_gold_sample(args.n_per_source, args.sample_seed)
    sample.to_csv(args.out_dir / "gold_subsample.csv", index=False)

    cfg = get_model_config(MODEL_KEY)
    meta = {
        "model": MODEL_KEY,
        "model_id": cfg["model_id"],
        "n_items": len(sample),
        "n_per_source": args.n_per_source,
        "temperatures": args.temperatures,
        "top_p": args.top_p,
        "seed": args.seed,
        "max_new_tokens": args.max_new_tokens,
        "device": device,
        "categories": args.categories,
        "baseline_paper_temperature": 0.1,
    }
    (args.out_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Sampled {len(sample)} items -> {args.out_dir}", flush=True)

    model, tok, cfg = load_qwen(device)
    rep = float(cfg.get("repetition_penalty", 1.1))

    all_long = []
    for temp in args.temperatures:
        tag = f"qwen_T{temp}_seed{args.seed}"
        out_csv = args.out_dir / f"preds_{tag}.csv"
        print(f"\n=== {tag} ===", flush=True)
        long_df = run_temp(
            sample,
            model=model,
            tok=tok,
            temperature=temp,
            top_p=args.top_p,
            seed=args.seed,
            device=device,
            max_new_tokens=args.max_new_tokens,
            repetition_penalty=rep,
            out_csv=out_csv,
            resume=not args.no_resume,
        )
        all_long.append(long_df)

    long_all = pd.concat(all_long, ignore_index=True)
    long_all.to_csv(args.out_dir / "predictions_long.csv", index=False)
    summary = summarize_gaps(long_all, args.categories)
    summary.to_csv(args.out_dir / "prevalence_gap_by_decode.csv", index=False)
    write_latex(summary, args.out_dir)

    print("\n=== Gaps by temperature ===", flush=True)
    print(
        summary.groupby(["temperature", "category"])[["gap_pp", "pi_ref", "pi_model"]]
        .mean()
        .to_string()
    )
    base = summary[np.isclose(summary["temperature"], 0.1)]
    if not base.empty:
        piv = summary.pivot_table(
            index="category", columns="temperature", values="gap_pp", aggfunc="mean"
        )
        if 0.1 in piv.columns:
            for t in args.temperatures:
                if abs(t - 0.1) > 1e-9 and t in piv.columns:
                    piv[f"delta_vs_T0.1@{t}"] = piv[t] - piv[0.1]
            piv.to_csv(args.out_dir / "gap_delta_vs_baseline_T0.1.csv")
            print("\nDelta vs paper T=0.1:", flush=True)
            print(piv.to_string())
    paper = REPO_ROOT / "paper_includes" / "decoding_prevalence_gap_qwen.tex"
    tex = args.out_dir / "decoding_prevalence_gap.tex"
    if tex.exists():
        paper.write_text(tex.read_text(encoding="utf-8"), encoding="utf-8")
        print(f"Copied LaTeX -> {paper}", flush=True)
    print(f"\nDone. Outputs in {args.out_dir}", flush=True)


if __name__ == "__main__":
    main()
