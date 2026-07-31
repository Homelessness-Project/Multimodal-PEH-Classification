#!/usr/bin/env python3
"""
Train an OATH-style Flan-T5 seq2seq multilabel tagger on GPT-4.1 pseudo-labels
for our full 16-category taxonomy, then evaluate on gold (2-of-3 soft labels).

This is the Flan-T5 training path that mirrors OATH-Frames (Ranjit et al.):
generate comma-separated snake_case labels from a fixed prefix. It is the best
*OATH-matched* recipe for our setup — not necessarily the best absolute 16-label
F1 (prefer ModernBERT / LoRA LLMs in `finetune_on_gpt_pseudolabels.py` for that).

Why these defaults
------------------
Defaults target **Mac MPS + flan-t5-large + LoRA** (also fine on CUDA with larger
micro-batches). Effective batch size = batch_size × grad_accum (= 16 by default).

  model (flan-t5-large)
      Same family/size as the released OATH tagger. Base (--fast) is for smoke
      tests only; Small is debug-only.

  LoRA on (r=16, alpha=32, dropout=0.05, targets q/v)
      Full FT of ~780M params needs lots of memory and is slow on MPS. LoRA
      trains a few million adapters (~0.1–1% of weights). r=16 is the usual
      PEFT sweet spot for T5; alpha=2r keeps update scale ≈1; light dropout
      reduces adapter overfit on noisy GPT labels. q/v is the T5 LoRA default
      in PEFT / common HF examples.

  batch_size=1, grad_accum=16  →  effective batch 16
      Flan-T5-Large + 512-token inputs often OOM at micro-batch >1–2 on 16–36GB
      unified memory. Grad accumulation preserves a stable effective batch of 16
      (standard for seq2seq fine-tunes) without holding 16 copies of activations.
      On a roomy CUDA GPU, prefer e.g. --batch-size 4 --grad-accum 4 (same 16).

  eval_batch_size=4
      Inference has no backward graph, so we can pack more sequences than train.
      4 is a safe default; raise if VRAM allows.

  lr=2e-4, warmup_ratio=0.03
      LoRA wants a higher LR than full FT (full FT often ~1e-4–5e-5). 2e-4 is a
      common PEFT default. Short warmup (~3%) avoids early adapter spikes without
      burning many of ~50k-example steps.

  epochs=3
      GPT pseudo-labels are large (~50k after gold exclusion) but noisy; 2–3
      epochs is typical before val loss plateaus. More than ~5 usually overfits
      teacher quirks rather than improving gold F1.

  max_input_length=512, max_target_length=64
      512 matches OATH’s training prefix + tweet/text budget and our eval script.
      64 new tokens is enough for a comma-separated 16-label string (or "none").

  source=all, gold 50/50 val/test, gold texts dropped from train
      Multi-domain train matches the paper setting. Gold is scarce (n≈1,698) so
      we hold it out entirely from train and split 50/50 for selection vs test
      (same recipe as finetune_on_gpt_pseudolabels.py). Soft labels ≥0.5 = 2-of-3.

  float32 on MPS / bf16|fp16 on CUDA
      MPS fp16 backward is unreliable; float32 + LoRA still fits. CUDA mixed
      precision speeds training when available.

  seed=42
      Reproducible splits and init with the rest of the repo.

Mac notes: start with --fast --max-train 200 --epochs 1; full Large + all
sources is multi-hour / overnight. Install peft if missing.

Examples:
  # Smoke on Mac (base + LoRA, 200 train rows)
  .venv/bin/pip install peft
  .venv/bin/python -u scripts/finetune_flan_t5_oath_style.py \\
      --source reddit --fast --max-train 200 --epochs 1 --device mps

  # Paper-style Large on all GPT pseudo-labels (overnight on Mac)
  .venv/bin/python -u scripts/finetune_flan_t5_oath_style.py \\
      --source all --epochs 3 --batch-size 1 --grad-accum 16 --device mps

  # CUDA workstation (same effective batch 16, larger micro-batch)
  .venv/bin/python -u scripts/finetune_flan_t5_oath_style.py \\
      --source all --epochs 3 --batch-size 4 --grad-accum 4 --device cuda

  # Resume after Ctrl+C / SIGTERM (same output dir)
  .venv/bin/python -u scripts/finetune_flan_t5_oath_style.py \\
      --source all --epochs 3 --batch-size 4 --grad-accum 4 --device cuda --resume
"""

from __future__ import annotations

import argparse
import json
import re
import signal
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    set_seed,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from gold_standard_utils import (  # noqa: E402
    SOURCE_SPECS,
    SOURCES,
    annotated_text_set,
    load_annotated_soft_labels,
    normalize_text,
)

DEFAULT_MODEL = "google/flan-t5-large"
FAST_MODEL = "google/flan-t5-base"
SMALL_MODEL = "google/flan-t5-small"

# Display names (soft-label / GPT columns) → OATH-style snake_case tokens.
CATEGORIES: List[str] = [
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

TO_SNAKE: Dict[str, str] = {
    "ask a genuine question": "ask_a_genuine_question",
    "ask a rhetorical question": "ask_a_rhetorical_question",
    "provide a fact or claim": "provide_a_fact_or_claim",
    "provide an observation": "provide_an_observation",
    "express their opinion": "express_their_opinion",
    "express others opinions": "express_others_opinions",
    "money aid allocation": "money_aid_allocation",
    "government critique": "government_critique",
    "societal critique": "societal_critique",
    "solutions/interventions": "solutions_interventions",
    "personal interaction": "personal_interaction",
    "media portrayal": "media_portrayal",
    "not in my backyard": "not_in_my_backyard",
    "harmful generalization": "harmful_generalization",
    "deserving/undeserving": "deserving_undeserving",
    "racist": "racist",
}

SNAKE_TO_CAT = {v: k for k, v in TO_SNAKE.items()}

GPT_TO_CATEGORY = {
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

LABEL_LIST_STR = ",".join(TO_SNAKE[c] for c in CATEGORIES)
PROMPT_PREFIX = (
    f"Classify the given text into one or more of the following "
    f"{len(CATEGORIES)} labels: {LABEL_LIST_STR},none. Text: "
)


def pick_device(name: str) -> torch.device:
    name = name.lower()
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if name == "mps":
        if not (getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()):
            print("WARNING: MPS requested but unavailable; falling back to CPU.")
            return torch.device("cpu")
        return torch.device("mps")
    if name == "cuda":
        if not torch.cuda.is_available():
            print("WARNING: CUDA requested but unavailable; falling back to CPU.")
            return torch.device("cpu")
        return torch.device("cuda")
    return torch.device("cpu")


def labels_to_target(active: Sequence[str]) -> str:
    snakes = [TO_SNAKE[c] for c in CATEGORIES if c in active]
    return ",".join(snakes) if snakes else "none"


def parse_generation(text: str) -> np.ndarray:
    """Map generated snake_case / aliases back to a 16-dim binary vector."""
    norm = re.sub(r"\s+", " ", str(text).lower()).strip()
    tokens = re.split(r"[,;|/]+", norm)
    vec = np.zeros(len(CATEGORIES), dtype=np.float32)
    for tok in tokens:
        t = tok.strip().replace(" ", "_")
        if not t or t in {"none", "0", "0.", "null", "n/a"}:
            continue
        if t in SNAKE_TO_CAT:
            vec[CATEGORIES.index(SNAKE_TO_CAT[t])] = 1.0
            continue
        # Fuzzy: substring match on snake forms (longest first).
        for snake, cat in sorted(SNAKE_TO_CAT.items(), key=lambda x: -len(x[0])):
            if snake in t or t in snake:
                vec[CATEGORIES.index(cat)] = 1.0
                break
    return vec


def gpt_row_to_active(row: pd.Series) -> List[str]:
    active = []
    for col, cat in GPT_TO_CATEGORY.items():
        if col not in row.index:
            continue
        val = row[col]
        try:
            if float(val) >= 0.5:
                active.append(cat)
        except (TypeError, ValueError):
            if str(val).strip().lower() in {"1", "true", "yes"}:
                active.append(cat)
    return active


def soft_row_to_vector(row: pd.Series, threshold: float = 0.5) -> np.ndarray:
    vec = np.zeros(len(CATEGORIES), dtype=np.float32)
    for i, cat in enumerate(CATEGORIES):
        if cat in row.index:
            try:
                if float(row[cat]) >= threshold:
                    vec[i] = 1.0
            except (TypeError, ValueError):
                pass
    return vec


def load_gpt_train(source: str, exclude_few_shot: bool = True) -> pd.DataFrame:
    gpt_file = REPO_ROOT / f"output/{source}/gpt4/classified_comments_{source}_all_gpt4_{source}_flags.csv"
    if not gpt_file.exists():
        raise FileNotFoundError(gpt_file)
    df = pd.read_csv(gpt_file)
    n0 = len(df)
    df = df.drop_duplicates(subset=["Comment"], keep="first")
    n_dedup = n0 - len(df)
    n_few = 0
    if exclude_few_shot:
        few = set()
        for suffix in (f"{source}_flags", "none_flags"):
            p = REPO_ROOT / f"output/{source}/gpt4/classified_comments_{source}_gold_subset_gpt4_{suffix}.csv"
            if p.exists():
                few.update(pd.read_csv(p)["Comment"].astype(str).tolist())
        if few:
            before = len(df)
            df = df[~df["Comment"].astype(str).isin(few)]
            n_few = before - len(df)
    # Drop any remaining gold-annotated texts (eval leakage safety net).
    # Often 0 after few-shot exclusion because gold_subset ≈ gold sample.
    gold_texts = annotated_text_set(source)
    before = len(df)
    df = df[~df["Comment"].map(normalize_text).isin(gold_texts)]
    n_gold = before - len(df)
    print(
        f"  [{source}] GPT train rows: {len(df):,} "
        f"(dedup -{n_dedup}, few-shot/gold_subset -{n_few}, residual gold -{n_gold})"
    )
    return df.reset_index(drop=True)


def load_gold_eval(source: str) -> Tuple[List[str], np.ndarray]:
    soft = load_annotated_soft_labels(source)
    gold_path, gold_col, raw_path, raw_col = SOURCE_SPECS[source]
    raw = pd.read_csv(REPO_ROOT / raw_path)
    ann = annotated_text_set(source)
    mask = raw[raw_col].map(normalize_text).isin(ann)
    texts = raw.loc[mask.values, raw_col].astype(str).tolist()
    # soft is already filtered to annotated rows in the same order as raw mask
    assert len(texts) == len(soft), (len(texts), len(soft), source)
    y = np.vstack([soft_row_to_vector(soft.iloc[i]) for i in range(len(soft))])
    return texts, y


class Seq2SeqFrameDataset(Dataset):
    def __init__(
        self,
        texts: Sequence[str],
        targets: Sequence[str],
        tokenizer,
        max_input_length: int = 512,
        max_target_length: int = 64,
    ):
        self.texts = list(texts)
        self.targets = list(targets)
        self.tokenizer = tokenizer
        self.max_input_length = max_input_length
        self.max_target_length = max_target_length

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int):
        src = PROMPT_PREFIX + str(self.texts[idx])
        model_inputs = self.tokenizer(
            src,
            max_length=self.max_input_length,
            truncation=True,
            padding=False,
        )
        labels = self.tokenizer(
            text_target=self.targets[idx],
            max_length=self.max_target_length,
            truncation=True,
            padding=False,
        )
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs


def multilabel_report(y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
    per = []
    for i, cat in enumerate(CATEGORIES):
        yt, yp = y_true[:, i], y_pred[:, i]
        per.append(
            {
                "category": cat,
                "support": int(yt.sum()),
                "precision": float(precision_score(yt, yp, zero_division=0)),
                "recall": float(recall_score(yt, yp, zero_division=0)),
                "f1": float(f1_score(yt, yp, zero_division=0)),
                "pred_rate": float(yp.mean()),
                "gold_rate": float(yt.mean()),
                "gap_pp": float((yp.mean() - yt.mean()) * 100.0),
            }
        )
    return {
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "micro_f1": float(f1_score(y_true, y_pred, average="micro", zero_division=0)),
        "per_label": per,
    }


@torch.inference_mode()
def generate_preds(
    model,
    tokenizer,
    texts: Sequence[str],
    device: torch.device,
    batch_size: int = 4,
    max_input_length: int = 512,
    max_new_tokens: int = 64,
) -> np.ndarray:
    model.eval()
    preds = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        enc = tokenizer(
            [PROMPT_PREFIX + t for t in batch],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_input_length,
        )
        enc = {k: v.to(device) for k, v in enc.items()}
        out = model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            num_beams=1,
            do_sample=False,
        )
        decoded = tokenizer.batch_decode(out, skip_special_tokens=True)
        preds.extend(parse_generation(d) for d in decoded)
        if (start // batch_size) % 20 == 0:
            print(f"    eval generate {min(start + batch_size, len(texts))}/{len(texts)}")
    return np.vstack(preds)


def maybe_peft(model, use_lora: bool, r: int, alpha: int, dropout: float):
    if not use_lora:
        return model
    try:
        from peft import LoraConfig, TaskType, get_peft_model
    except ImportError as e:
        raise SystemExit(
            "peft is required for LoRA. Install with: .venv/bin/pip install peft"
        ) from e
    cfg = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        target_modules=["q", "v"],
        bias="none",
    )
    model = get_peft_model(model, cfg)
    model.print_trainable_parameters()
    return model


def latest_checkpoint(ckpt_root: Path) -> Optional[Path]:
    """Return newest checkpoints/checkpoint-* dir, if any."""
    if not ckpt_root.is_dir():
        return None
    cands = [
        p for p in ckpt_root.iterdir()
        if p.is_dir() and p.name.startswith("checkpoint-")
    ]
    if not cands:
        return None

    def _step(p: Path) -> int:
        try:
            return int(p.name.split("-", 1)[1])
        except (IndexError, ValueError):
            return -1

    return max(cands, key=_step)


def save_interrupt_checkpoint(trainer, tokenizer, out: Path) -> Path:
    """
    Persist a resumable HF checkpoint on Ctrl+C / SIGTERM.
    Prefers Trainer._save_checkpoint (weights + optimizer + scheduler).
    """
    step = int(getattr(trainer.state, "global_step", 0) or 0)
    ckpt_root = Path(trainer.args.output_dir)
    ckpt_root.mkdir(parents=True, exist_ok=True)
    print(
        f"\nInterrupt — saving checkpoint at step {step} "
        f"(Ctrl+C again to force quit without finishing the save)...",
        flush=True,
    )
    try:
        if hasattr(trainer, "_save_checkpoint"):
            trainer._save_checkpoint(trainer.model, trial=None)
            saved = latest_checkpoint(ckpt_root) or (ckpt_root / f"checkpoint-{step}")
        else:
            saved = ckpt_root / f"checkpoint-{step}"
            saved.mkdir(parents=True, exist_ok=True)
            trainer.save_model(str(saved))
            tokenizer.save_pretrained(str(saved))
            trainer.state.save_to_json(str(saved / "trainer_state.json"))

        # Convenience copy for humans / --resume fallback
        interrupt_dir = out / "interrupted"
        trainer.save_model(str(interrupt_dir))
        tokenizer.save_pretrained(str(interrupt_dir))
        marker = {
            "global_step": step,
            "epoch": float(getattr(trainer.state, "epoch", 0) or 0),
            "checkpoint": str(saved),
            "interrupted_copy": str(interrupt_dir),
        }
        (out / "interrupted.json").write_text(json.dumps(marker, indent=2))
        print(f"Saved: {saved}", flush=True)
        print(f"Resume: same command + --resume  (or --resume-from {saved})", flush=True)
        return Path(saved)
    except KeyboardInterrupt:
        print("\nForce quit during save — checkpoint may be incomplete.", flush=True)
        raise SystemExit(130) from None


def install_term_handlers() -> None:
    """Map SIGTERM (kill) to KeyboardInterrupt so we share one graceful path."""

    def _raise_keyboard(signum, frame):  # noqa: ARG001
        raise KeyboardInterrupt

    signal.signal(signal.SIGINT, signal.default_int_handler)
    try:
        signal.signal(signal.SIGTERM, _raise_keyboard)
    except (ValueError, OSError):
        pass


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--source",
        default="all",
        choices=["all", *SOURCES],
        help="Train domain(s). Default all = multi-domain paper setting.",
    )
    p.add_argument(
        "--model",
        default=None,
        help="HF model id (default: google/flan-t5-large, OATH-matched size).",
    )
    p.add_argument("--fast", action="store_true", help="flan-t5-base for smoke tests")
    p.add_argument("--small", action="store_true", help="flan-t5-small (debug only)")
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument(
        "--epochs",
        type=float,
        default=3.0,
        help="Default 3: enough for ~50k noisy GPT rows before gold overfit.",
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Micro-batch. Default 1 fits Large@512 on Mac MPS; pair with --grad-accum.",
    )
    p.add_argument(
        "--eval-batch-size",
        type=int,
        default=4,
        help="Larger than train OK (no backward). Default 4.",
    )
    p.add_argument(
        "--grad-accum",
        type=int,
        default=16,
        help="Default 16 → effective batch batch_size×16 (=16 on Mac defaults).",
    )
    p.add_argument(
        "--lr",
        type=float,
        default=2e-4,
        help="Default 2e-4: typical LoRA LR (higher than full FT).",
    )
    p.add_argument(
        "--warmup-ratio",
        type=float,
        default=0.03,
        help="Default 0.03: short warmup to stabilize early adapter updates.",
    )
    p.add_argument(
        "--max-input-length",
        type=int,
        default=512,
        help="Default 512: matches OATH training / our oath_gold_frame_eval.py.",
    )
    p.add_argument(
        "--max-target-length",
        type=int,
        default=64,
        help="Default 64: enough for comma-separated 16-label targets.",
    )
    p.add_argument("--max-train", type=int, default=None, help="Cap training rows (smoke)")
    p.add_argument("--seed", type=int, default=42, help="Default 42: repo-wide reproducibility")
    p.add_argument("--device", default="auto", choices=["auto", "mps", "cuda", "cpu"])
    p.add_argument(
        "--no-lora",
        action="store_true",
        help="Full fine-tune (slow / memory-heavy; not recommended on Mac).",
    )
    p.add_argument(
        "--lora-r",
        type=int,
        default=16,
        help="Default 16: standard PEFT rank for T5 adapters.",
    )
    p.add_argument(
        "--lora-alpha",
        type=int,
        default=32,
        help="Default 32 (=2×r): keeps LoRA update scale ≈1.",
    )
    p.add_argument(
        "--lora-dropout",
        type=float,
        default=0.05,
        help="Default 0.05: light regularization on noisy pseudo-labels.",
    )
    p.add_argument("--eval-only", action="store_true", help="Load checkpoint and eval gold test")
    p.add_argument(
        "--resume",
        action="store_true",
        help="Resume from latest checkpoint under output-dir/checkpoints/",
    )
    p.add_argument(
        "--resume-from",
        type=Path,
        default=None,
        help="Resume from an explicit checkpoint-* directory",
    )
    return p


def main() -> None:
    args = build_parser().parse_args()
    install_term_handlers()
    set_seed(args.seed)
    device = pick_device(args.device)
    use_lora = not args.no_lora

    if args.small:
        model_id = SMALL_MODEL
    elif args.fast:
        model_id = FAST_MODEL
    else:
        model_id = args.model or DEFAULT_MODEL

    sources = SOURCES if args.source == "all" else [args.source]
    out = args.output_dir or (
        REPO_ROOT / "output" / "flan_t5_oath_style" / f"{args.source}_{model_id.split('/')[-1]}"
    )
    out.mkdir(parents=True, exist_ok=True)

    print(f"Model: {model_id}")
    print(f"Device: {device} | LoRA: {use_lora}")
    print(f"Output: {out}")
    print("Tip: Ctrl+C saves a resumable checkpoint; Ctrl+C again force-quits.", flush=True)
    if device.type == "mps":
        print(
            "Mac tip: LoRA + float32 + small batch/grad-accum. "
            "If you OOM, try --fast or --max-train 2000."
        )

    # ---- data ----
    train_texts: List[str] = []
    train_targets: List[str] = []
    for src in sources:
        df = load_gpt_train(src)
        for _, row in df.iterrows():
            train_texts.append(str(row["Comment"]))
            train_targets.append(labels_to_target(gpt_row_to_active(row)))

    if args.max_train is not None and len(train_texts) > args.max_train:
        rng = np.random.default_rng(args.seed)
        idx = rng.choice(len(train_texts), size=args.max_train, replace=False)
        train_texts = [train_texts[i] for i in idx]
        train_targets = [train_targets[i] for i in idx]
        print(f"  Capped train to {len(train_texts):,} rows")

    gold_texts: List[str] = []
    gold_y: List[np.ndarray] = []
    for src in sources:
        t, y = load_gold_eval(src)
        gold_texts.extend(t)
        gold_y.append(y)
    gold_y_arr = np.vstack(gold_y)
    val_texts, test_texts, val_y, test_y = train_test_split(
        gold_texts, gold_y_arr, test_size=0.5, random_state=args.seed
    )
    val_targets = [
        labels_to_target([CATEGORIES[i] for i, v in enumerate(row) if v >= 0.5])
        for row in val_y
    ]

    print(f"Train: {len(train_texts):,} | Val gold: {len(val_texts)} | Test gold: {len(test_texts)}")

    meta = {
        "model_id": model_id,
        "sources": sources,
        "n_train": len(train_texts),
        "n_val": len(val_texts),
        "n_test": len(test_texts),
        "use_lora": use_lora,
        "device": str(device),
        "prompt_prefix": PROMPT_PREFIX,
        "categories": CATEGORIES,
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2))

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    # T5 pad = eos conventionally
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    torch_dtype = torch.float32
    if device.type == "cuda":
        torch_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    def _load_seq2seq(path_or_id: str):
        # transformers≥4.56 prefers `dtype=`; fall back to torch_dtype on older installs.
        try:
            return AutoModelForSeq2SeqLM.from_pretrained(path_or_id, dtype=torch_dtype)
        except TypeError:
            return AutoModelForSeq2SeqLM.from_pretrained(path_or_id, torch_dtype=torch_dtype)

    model = _load_seq2seq(model_id)
    model = maybe_peft(model, use_lora, args.lora_r, args.lora_alpha, args.lora_dropout)
    model.to(device)

    if args.eval_only:
        ckpt = out / "final"
        if not ckpt.exists():
            raise SystemExit(f"No checkpoint at {ckpt}")
        if (ckpt / "adapter_config.json").exists():
            from peft import PeftModel

            base = _load_seq2seq(model_id)
            model = PeftModel.from_pretrained(base, str(ckpt))
        else:
            model = _load_seq2seq(str(ckpt))
        model.to(device)
        print("Evaluating test split only...")
        pred = generate_preds(
            model, tokenizer, test_texts, device,
            batch_size=args.eval_batch_size,
            max_input_length=args.max_input_length,
        )
        report = multilabel_report(test_y, pred)
        print(f"Test macro-F1: {report['macro_f1']:.4f}  micro-F1: {report['micro_f1']:.4f}")
        pd.DataFrame(report["per_label"]).to_csv(out / "test_per_label.csv", index=False)
        (out / "test_metrics.json").write_text(json.dumps(report, indent=2))
        return

    train_ds = Seq2SeqFrameDataset(
        train_texts, train_targets, tokenizer, args.max_input_length, args.max_target_length
    )
    val_ds = Seq2SeqFrameDataset(
        val_texts, val_targets, tokenizer, args.max_input_length, args.max_target_length
    )
    collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

    # HF Trainer: pin MPS to no fp16; use grad accum for effective batch.
    use_fp16 = device.type == "cuda" and torch_dtype == torch.float16
    use_bf16 = device.type == "cuda" and torch_dtype == torch.bfloat16

    training_args = Seq2SeqTrainingArguments(
        output_dir=str(out / "checkpoints"),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        warmup_ratio=args.warmup_ratio,
        logging_steps=50,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        predict_with_generate=False,  # we do custom gold generation after train
        fp16=use_fp16,
        bf16=use_bf16,
        dataloader_pin_memory=(device.type == "cuda"),
        report_to=[],
        seed=args.seed,
        remove_unused_columns=False,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
    )

    trainer_kwargs = dict(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collator,
    )
    # transformers 5 removes `tokenizer=`; prefer processing_class when available.
    try:
        trainer = Seq2SeqTrainer(**trainer_kwargs, processing_class=tokenizer)
    except TypeError:
        trainer = Seq2SeqTrainer(**trainer_kwargs, tokenizer=tokenizer)

    resume_path: Optional[str] = None
    if args.resume_from is not None:
        resume_path = str(args.resume_from)
    elif args.resume:
        found = latest_checkpoint(out / "checkpoints")
        if found is None:
            raise SystemExit(f"No checkpoint-* under {out / 'checkpoints'} to --resume from")
        resume_path = str(found)
    if resume_path:
        print(f"Resuming from {resume_path}", flush=True)

    print("Starting training...", flush=True)
    try:
        trainer.train(resume_from_checkpoint=resume_path)
    except KeyboardInterrupt:
        save_interrupt_checkpoint(trainer, tokenizer, out)
        raise SystemExit(130) from None

    final_dir = out / "final"
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))
    print(f"Saved to {final_dir}")

    print("Generating on gold val/test...")
    model = trainer.model
    model.to(device)
    try:
        val_pred = generate_preds(
            model, tokenizer, val_texts, device,
            batch_size=args.eval_batch_size,
            max_input_length=args.max_input_length,
        )
        test_pred = generate_preds(
            model, tokenizer, test_texts, device,
            batch_size=args.eval_batch_size,
            max_input_length=args.max_input_length,
        )
    except KeyboardInterrupt:
        print(
            "\nInterrupted during gold generation. "
            f"Weights already saved at {final_dir}. Re-run with --eval-only.",
            flush=True,
        )
        raise SystemExit(130) from None

    val_report = multilabel_report(val_y, val_pred)
    test_report = multilabel_report(test_y, test_pred)
    print(f"Val  macro-F1: {val_report['macro_f1']:.4f}  micro-F1: {val_report['micro_f1']:.4f}")
    print(f"Test macro-F1: {test_report['macro_f1']:.4f}  micro-F1: {test_report['micro_f1']:.4f}")

    pd.DataFrame(val_report["per_label"]).to_csv(out / "val_per_label.csv", index=False)
    pd.DataFrame(test_report["per_label"]).to_csv(out / "test_per_label.csv", index=False)
    (out / "val_metrics.json").write_text(json.dumps(val_report, indent=2))
    (out / "test_metrics.json").write_text(json.dumps(test_report, indent=2))
    summary = {
        "model_id": model_id,
        "sources": sources,
        "use_lora": use_lora,
        "val_macro_f1": val_report["macro_f1"],
        "test_macro_f1": test_report["macro_f1"],
        "val_micro_f1": val_report["micro_f1"],
        "test_micro_f1": test_report["micro_f1"],
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
