#!/usr/bin/env python3
"""
Run OATH-Frames Flan-T5 (Hugging Face) on the gold-standard set and compare
all 9 OATH-overlapping labels against human soft labels (2-of-3).

Default model is **Flan-T5-Large** (`dill-lab/oath-frames-flant5-large`), matching
the main OATH paper claim (Ranjit et al., EMNLP 2024). Use --fast/--small only
for debugging.

Examples:
  # Paper-matched Large (default) on full gold n=1,698
  .venv/bin/python -u scripts/oath_gold_frame_eval.py --device mps

  # Debug with base
  .venv/bin/python -u scripts/oath_gold_frame_eval.py --fast --device mps --limit 32

  # Recompute metrics from checkpoint only
  .venv/bin/python -u scripts/oath_gold_frame_eval.py --summary-only
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from gold_standard_utils import (  # noqa: E402
    GOLD_STANDARD_N,
    REPO_ROOT as _GS_ROOT,
    SOURCE_SPECS,
    SOURCES,
    annotated_text_set,
    load_annotated_soft_labels,
    normalize_text,
)

assert _GS_ROOT == REPO_ROOT

DEFAULT_MODEL = "dill-lab/oath-frames-flant5-large"
FAST_MODEL = "dill-lab/oath-frames-flant5-base"
SMALL_MODEL = "dill-lab/oath-frames-flant5-small"

# Exact training prefix from OATH's trainer_deepspeed.sh (must match for the
# released Flan-T5 checkpoint to emit frame labels rather than NER/MC garbage).
OATH_SOURCE_PREFIX = (
    "Classify the given tweet into one or more of the following 10 labels: "
    "government_critique,money_aid_resource_allocation,societal_critique,"
    "deserving_undeserving_of_resources,harmful_generalization,not_in_my_backyard,"
    "media_portrayal,personal_interaction_observation_of_homelessness,"
    "solutions_interventions,0. Tweet: "
)

# Legacy / wrong prompt used in an earlier run (kept only for ablation).
LEGACY_PROMPT_TEMPLATE = (
    "Given the following post about homelessness, predict the OATH frames "
    "as a comma-separated list.\npost: {text}"
)

PROMPT_TEMPLATE = OATH_SOURCE_PREFIX + "{text}"

# Gold soft-label columns that overlap OATH's 9 frames.
OATH_GOLD_LABELS: List[str] = [
    "money aid allocation",
    "government critique",
    "societal critique",
    "solutions/interventions",
    "personal interaction",
    "media portrayal",
    "not in my backyard",
    "harmful generalization",
    "deserving/undeserving",
]

# Substring matchers applied to normalized OATH generation tokens.
# Prefer OATH snake_case training labels; keep short aliases as fallback.
FRAME_MATCHERS: Dict[str, Tuple[str, ...]] = {
    "money aid allocation": (
        "money_aid_resource_allocation",
        "moneyaid",
        "money_aid",
        "money aid",
        "money",
    ),
    "government critique": (
        "government_critique",
        "govcrit",
        "governmentcritique",
        "government critique",
        "government",
    ),
    "societal critique": (
        "societal_critique",
        "soccrit",
        "societalcritique",
        "societal critique",
        "societal",
    ),
    "solutions/interventions": (
        "solutions_interventions",
        "solnint",
        "solutions",
        "solution",
    ),
    "personal interaction": (
        "personal_interaction_observation_of_homelessness",
        "personal_interaction",
        "interact",
        "personalinteraction",
        "personal",
    ),
    "media portrayal": (
        "media_portrayal",
        "mediaport",
        "mediaportrayal",
        "media",
    ),
    "not in my backyard": (
        "not_in_my_backyard",
        "nimby",
        "backyard",
        "not in my backyard",
    ),
    "harmful generalization": (
        "harmful_generalization",
        "harmgen",
        "harmfulgeneralization",
        "harmful",
    ),
    "deserving/undeserving": (
        "deserving_undeserving_of_resources",
        "deserving_undeserving",
        "undeserv",
        "deserv",
        "(un)deserv",
    ),
}

DISPLAY = {
    "money aid allocation": "MoneyAid",
    "government critique": "GovCrit",
    "societal critique": "SocCrit",
    "solutions/interventions": "SolnInt",
    "personal interaction": "Interact",
    "media portrayal": "MediaPort",
    "not in my backyard": "NIMBY",
    "harmful generalization": "HarmGen",
    "deserving/undeserving": "(Un)Deserv",
}


@dataclass(frozen=True)
class PredRow:
    source: str
    row_index: int
    text: str
    oath_frames_raw: str
    pred_labels: Tuple[str, ...]


def resolve_model_id(model: Optional[str], *, fast: bool, small: bool) -> str:
    if model:
        return model
    if small:
        return SMALL_MODEL
    if fast:
        return FAST_MODEL
    return DEFAULT_MODEL


def truncate_text(text: str, max_chars: int = 1200) -> str:
    t = " ".join(str(text).split())
    if len(t) <= max_chars:
        return t
    return t[: max_chars - 1] + "…"


def _token_hits(raw: str) -> Dict[str, bool]:
    """Map OATH generation string → which of the 9 gold labels are predicted."""
    text = (raw or "").strip().lower()
    parts = [p.strip() for p in re.split(r"[,;/|]+", text) if p.strip()]
    # Also keep full string for whole-string fallbacks.
    candidates = parts + ([text] if text and text not in parts else [])
    hits = {lab: False for lab in OATH_GOLD_LABELS}
    for part in candidates:
        compact = re.sub(r"[^a-z0-9]+", "", part)
        spaced = re.sub(r"\s+", " ", part).strip()
        # Negation / garbage that often co-occurs with "societal"/"money"
        if spaced.startswith("not ") or spaced.startswith("not_") or spaced in {"0", "___"}:
            continue
        for lab, patterns in FRAME_MATCHERS.items():
            if hits[lab]:
                continue
            for pat in patterns:
                pat_c = re.sub(r"[^a-z0-9]+", "", pat)
                if pat_c and pat_c in compact:
                    hits[lab] = True
                    break
                if pat in spaced:
                    hits[lab] = True
                    break
    return hits


def parse_oath_frames(raw: str) -> List[str]:
    hits = _token_hits(raw)
    return [lab for lab in OATH_GOLD_LABELS if hits[lab]]


def load_gold_items(*, sources: Optional[Sequence[str]] = None) -> pd.DataFrame:
    """Load annotated gold items (n=1,698) with text + soft scores for OATH labels."""
    rows: List[dict] = []
    for source in sources or SOURCES:
        gold_path, gold_col, raw_path, raw_col = SOURCE_SPECS[source]
        raw = pd.read_csv(REPO_ROOT / raw_path, low_memory=False)
        soft = load_annotated_soft_labels(source)
        ann = annotated_text_set(source)
        mask = raw[raw_col].map(normalize_text).isin(ann)
        raw_m = raw.loc[mask.values].reset_index(drop=True)
        if len(raw_m) != len(soft):
            raise RuntimeError(
                f"{source}: raw annotated rows ({len(raw_m)}) != soft ({len(soft)})"
            )
        for i in range(len(soft)):
            text = str(raw_m.at[i, raw_col])
            row = {
                "source": source,
                "row_index": int(i),
                "text": text,
            }
            for lab in OATH_GOLD_LABELS:
                s = float(pd.to_numeric(soft.at[i, lab], errors="coerce") or 0.0)
                row[f"gold_{lab}"] = s
                row[f"y_{lab}"] = int(s >= 2.0 / 3.0)
            rows.append(row)
    df = pd.DataFrame(rows)
    if len(df) != GOLD_STANDARD_N and sources is None:
        print(
            f"WARNING: expected n={GOLD_STANDARD_N}, got {len(df)}",
            flush=True,
        )
    return df


def binary_prf1(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) else 0.0
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "support": int(y_true.sum()),
        "pred_pos": int(y_pred.sum()),
        "pi_ref": float(y_true.mean()) if len(y_true) else 0.0,
        "pi_model": float(y_pred.mean()) if len(y_pred) else 0.0,
        "gap_pp": 100.0 * (float(y_pred.mean()) - float(y_true.mean())) if len(y_true) else 0.0,
    }


def pick_device(pref: str):
    import torch

    pref = (pref or "auto").lower()
    if pref == "cpu":
        return torch.device("cpu")
    if pref == "mps":
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return torch.device("mps")
        print("MPS unavailable; falling back to CPU", flush=True)
        return torch.device("cpu")
    if pref == "cuda":
        if torch.cuda.is_available():
            return torch.device("cuda")
        print("CUDA unavailable; falling back to CPU", flush=True)
        return torch.device("cpu")
    # auto
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class OATHFramesClassifier:
    def __init__(
        self,
        model_id: str,
        *,
        device: str = "auto",
        max_new_tokens: int = 64,
        max_input_length: int = 512,
        batch_size: Optional[int] = None,
        use_fp16: Optional[bool] = None,
    ) -> None:
        self.model_id = model_id
        self.device_pref = device
        self.max_new_tokens = max_new_tokens
        self.max_input_length = max_input_length
        self.batch_size = batch_size
        self.use_fp16 = use_fp16
        self._model = None
        self._tokenizer = None
        self._torch_device = None

    def _load(self) -> None:
        if self._model is not None:
            return
        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        self._torch_device = pick_device(self.device_pref)
        print(f"OATH device={self._torch_device}  model={self.model_id}", flush=True)

        def _from_pretrained(cls, **kwargs):
            try:
                return cls.from_pretrained(self.model_id, local_files_only=True, **kwargs)
            except Exception as e_local:
                print(f"Local cache miss ({e_local}); trying network…", flush=True)
                return cls.from_pretrained(self.model_id, **kwargs)

        self._tokenizer = _from_pretrained(AutoTokenizer)
        want_fp16 = (
            self.use_fp16
            if self.use_fp16 is not None
            else self._torch_device.type in {"mps", "cuda"}
        )
        dtype = torch.float16 if want_fp16 else torch.float32
        # Prefer `dtype=` (new) with fallback to `torch_dtype=` (older transformers).
        try:
            self._model = _from_pretrained(AutoModelForSeq2SeqLM, dtype=dtype)
        except TypeError:
            self._model = _from_pretrained(AutoModelForSeq2SeqLM, torch_dtype=dtype)
        self._model.to(self._torch_device)
        self._model.eval()
        if self.batch_size is None:
            # Large needs smaller batches on Apple Silicon than base.
            if "large" in self.model_id.lower():
                self.batch_size = 2 if self._torch_device.type == "mps" else 4
            else:
                self.batch_size = 8 if self._torch_device.type == "mps" else 4
        print(
            f"OATH batch_size={self.batch_size}  dtype={dtype}  "
            f"max_in={self.max_input_length}",
            flush=True,
        )

    def generate(self, texts: Sequence[str], *, show_progress: bool = True) -> List[str]:
        self._load()
        import torch

        outs: List[str] = []
        bs = int(self.batch_size or 4)
        ranges = list(range(0, len(texts), bs))
        if show_progress:
            try:
                from tqdm import tqdm

                ranges = tqdm(ranges, desc="OATH", unit="batch")
            except ImportError:
                pass
        for i in ranges:
            chunk = [truncate_text(t) for t in texts[i : i + bs]]
            prompts = [PROMPT_TEMPLATE.format(text=t) for t in chunk]
            enc = self._tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_input_length,
            )
            enc = {k: v.to(self._torch_device) for k, v in enc.items()}
            with torch.inference_mode():
                gen = self._model.generate(
                    **enc,
                    max_new_tokens=self.max_new_tokens,
                    num_beams=1,
                    do_sample=False,
                    use_cache=True,
                )
            decoded = self._tokenizer.batch_decode(gen, skip_special_tokens=True)
            outs.extend(decoded)
        return outs


def metrics_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for lab in OATH_GOLD_LABELS:
        y = df[f"y_{lab}"].astype(int).to_numpy()
        p = df[f"pred_{lab}"].astype(int).to_numpy()
        m = binary_prf1(y, p)
        rows.append(
            {
                "label": lab,
                "short": DISPLAY[lab],
                "n_pos_gold": m["support"],
                "n_pos_pred": m["pred_pos"],
                "precision": m["precision"],
                "recall": m["recall"],
                "f1": m["f1"],
                "pi_ref_pct": 100.0 * m["pi_ref"],
                "pi_model_pct": 100.0 * m["pi_model"],
                "gap_pp": m["gap_pp"],
                "tp": m["tp"],
                "fp": m["fp"],
                "fn": m["fn"],
            }
        )
    out = pd.DataFrame(rows)
    # Macro over the 9 OATH labels
    macro = {
        "label": "MACRO (9 OATH)",
        "short": "MACRO",
        "n_pos_gold": int(out["n_pos_gold"].sum()),
        "n_pos_pred": int(out["n_pos_pred"].sum()),
        "precision": float(out["precision"].mean()),
        "recall": float(out["recall"].mean()),
        "f1": float(out["f1"].mean()),
        "pi_ref_pct": float(out["pi_ref_pct"].mean()),
        "pi_model_pct": float(out["pi_model_pct"].mean()),
        "gap_pp": float(out["gap_pp"].mean()),
        "tp": int(out["tp"].sum()),
        "fp": int(out["fp"].sum()),
        "fn": int(out["fn"].sum()),
    }
    return pd.concat([out, pd.DataFrame([macro])], ignore_index=True)


def write_latex(summary: pd.DataFrame, path: Path, *, model_id: str, n: int) -> None:
    lines = [
        "% Auto-generated OATH Flan-T5 vs gold (9 overlapping frames)",
        "\\begin{table}[!htb]",
        "\\centering",
        "\\scriptsize",
        "\\setlength{\\tabcolsep}{2pt}",
        "\\begin{tabularx}{\\columnwidth}{@{}>{\\raggedright\\arraybackslash}X r r r r r@{}}",
        "\\toprule",
        "\\textbf{Frame} & $n_{+}$ & P & R & F1 & Gap \\\\",
        "\\midrule",
    ]
    for _, r in summary.iterrows():
        if r["short"] == "MACRO":
            lines.append("\\midrule")
        lines.append(
            f"{r['short']} & {int(r['n_pos_gold'])} & {r['precision']:.2f} & "
            f"{r['recall']:.2f} & {r['f1']:.2f} & ${r['gap_pp']:+.1f}$ \\\\"
        )
    lines += [
        "\\bottomrule",
        "\\end{tabularx}",
        f"\\caption{{OATH Flan-T5 ({model_id.split('/')[-1]}) vs.\\ gold 2-of-3 "
        f"on the {n}-item audit set. Gap = $\\hat{{\\pi}}-\\pi_{{\\mathrm{{ref}}}}$ (pp). "
        "Nine OATH-overlapping labels only.}}",
        "\\label{tab:oath_gold_frame_eval}",
        "\\end{table}",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def apply_predictions(gold: pd.DataFrame, preds: pd.DataFrame) -> pd.DataFrame:
    merged = gold.merge(
        preds[["source", "row_index", "oath_frames_raw"]],
        on=["source", "row_index"],
        how="left",
        validate="one_to_one",
    )
    for lab in OATH_GOLD_LABELS:
        merged[f"pred_{lab}"] = 0
    for i, raw in enumerate(merged["oath_frames_raw"].fillna("").astype(str)):
        for lab in parse_oath_frames(raw):
            merged.at[i, f"pred_{lab}"] = 1
    return merged


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--model",
        default=None,
        help="HF model id override (default: dill-lab/oath-frames-flant5-large)",
    )
    ap.add_argument(
        "--fast",
        action="store_true",
        help="Debug only: Flan-T5-base (not paper-matched)",
    )
    ap.add_argument(
        "--small",
        action="store_true",
        help="Debug only: Flan-T5-small (not paper-matched)",
    )
    ap.add_argument("--device", default="auto", help="auto|mps|cuda|cpu")
    ap.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Generation batch size (Large on MPS: try 2–4)",
    )
    ap.add_argument("--max-new-tokens", type=int, default=64)
    ap.add_argument("--limit", type=int, default=None, help="Score only first N items")
    ap.add_argument(
        "--sources",
        nargs="+",
        default=None,
        choices=SOURCES,
        help="Subset of sources (default: all four)",
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=REPO_ROOT / "output" / "oath_gold_eval",
    )
    ap.add_argument(
        "--summary-only",
        action="store_true",
        help="Recompute metrics from existing preds CSV; no model load",
    )
    ap.add_argument("--no-fp16", action="store_true")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    preds_path = args.out_dir / "oath_preds.csv"
    summary_path = args.out_dir / "oath_vs_gold_metrics.csv"
    latex_path = args.out_dir / "oath_vs_gold_metrics.tex"
    meta_path = args.out_dir / "meta.json"

    gold = load_gold_items(sources=args.sources)
    if args.limit is not None:
        gold = gold.iloc[: args.limit].copy()
    print(f"Gold items: {len(gold)} (canonical n={GOLD_STANDARD_N})", flush=True)

    model_id = resolve_model_id(args.model, fast=args.fast, small=args.small)

    if args.summary_only:
        if not preds_path.is_file():
            raise SystemExit(f"--summary-only but missing {preds_path}")
        preds = pd.read_csv(preds_path)
        model_id = json.loads(meta_path.read_text()).get("model", model_id) if meta_path.is_file() else model_id
    else:
        # Resume: keep completed (source, row_index)
        done_keys: set[Tuple[str, int]] = set()
        prior_rows: List[dict] = []
        if preds_path.is_file():
            prior = pd.read_csv(preds_path)
            for _, r in prior.iterrows():
                key = (str(r["source"]), int(r["row_index"]))
                done_keys.add(key)
                prior_rows.append(r.to_dict())
            print(f"Resume: {len(done_keys)} already scored", flush=True)

        todo = gold[
            ~gold.apply(lambda r: (r["source"], int(r["row_index"])) in done_keys, axis=1)
        ].copy()
        print(f"To score: {len(todo)}", flush=True)

        if len(todo):
            clf = OATHFramesClassifier(
                model_id,
                device=args.device,
                max_new_tokens=args.max_new_tokens,
                batch_size=args.batch_size,
                use_fp16=False if args.no_fp16 else None,
            )
            # Checkpoint every N items so a long Large run can resume.
            chunk_n = max(int(args.batch_size or 2) * 10, 20)
            new_rows: List[dict] = []
            todo_list = list(todo.itertuples(index=False))
            for start in range(0, len(todo_list), chunk_n):
                chunk = todo_list[start : start + chunk_n]
                texts = [getattr(r, "text") for r in chunk]
                print(
                    f"Scoring items {start + 1}-{start + len(chunk)} / {len(todo_list)} "
                    f"(model={model_id})",
                    flush=True,
                )
                raws = clf.generate(texts, show_progress=True)
                for r, raw in zip(chunk, raws):
                    labs = parse_oath_frames(raw)
                    row = {
                        "source": r.source,
                        "row_index": int(r.row_index),
                        "oath_frames_raw": raw[:500],
                        "pred_labels": ";".join(labs),
                    }
                    for lab in OATH_GOLD_LABELS:
                        row[f"pred_{lab}"] = int(lab in labs)
                    new_rows.append(row)
                all_preds = pd.DataFrame(prior_rows + new_rows)
                all_preds = all_preds.drop_duplicates(
                    ["source", "row_index"], keep="last"
                ).sort_values(["source", "row_index"])
                all_preds.to_csv(preds_path, index=False)
                print(f"Checkpoint: {len(all_preds)} preds → {preds_path}", flush=True)
            preds = pd.read_csv(preds_path)
        else:
            preds = pd.read_csv(preds_path)

    merged = apply_predictions(gold, preds)
    # Prefer explicit pred_* columns from preds when present
    for lab in OATH_GOLD_LABELS:
        col = f"pred_{lab}"
        if col in preds.columns:
            tmp = gold.merge(
                preds[["source", "row_index", col]],
                on=["source", "row_index"],
                how="left",
                suffixes=("", "_p"),
            )
            merged[col] = tmp[col].fillna(0).astype(int)

    merged.to_csv(args.out_dir / "oath_joined_with_gold.csv", index=False)
    summary = metrics_table(merged)
    summary.to_csv(summary_path, index=False)
    write_latex(summary, latex_path, model_id=model_id, n=len(merged))

    meta = {
        "model": model_id,
        "n": int(len(merged)),
        "gold_n_canonical": GOLD_STANDARD_N,
        "tau": "2/3",
        "labels": OATH_GOLD_LABELS,
        "device": args.device,
        "fast": bool(args.fast),
        "small": bool(args.small),
        "macro_f1": float(summary.loc[summary["short"] == "MACRO", "f1"].iloc[0]),
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print("\n=== OATH vs gold (2-of-3) ===", flush=True)
    show = summary[
        ["short", "n_pos_gold", "n_pos_pred", "precision", "recall", "f1", "gap_pp"]
    ].copy()
    for c in ("precision", "recall", "f1"):
        show[c] = show[c].map(lambda x: f"{x:.3f}")
    show["gap_pp"] = show["gap_pp"].map(lambda x: f"{x:+.1f}")
    print(show.to_string(index=False), flush=True)
    print(f"\nWrote {summary_path}", flush=True)
    print(f"Wrote {latex_path}", flush=True)
    print(f"Wrote {preds_path}", flush=True)


if __name__ == "__main__":
    main()
