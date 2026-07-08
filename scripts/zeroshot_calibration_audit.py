#!/usr/bin/env python3
"""
Prompt calibration audit on the gold-standard set (six prompt LLMs), reported
separately for zero-shot vs few-shot.

Gold reference: three annotators; soft scores in {0, 1/3, 2/3, 1}. Default
reference = 2-of-3 (tau=2/3, Eq. 1). Sensitivity at 1-of-3 (tau=1/3) only.

Models: LLaMA, Phi-4, Qwen, GPT-4.1, Gemini, Grok.

Usage (repo root):
    .venv/bin/python scripts/zeroshot_calibration_audit.py
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = REPO_ROOT / "output" / "f1" / "zeroshot_calibration_audit"

SOURCES = ["reddit", "news", "meeting_minutes", "x"]
PROMPT_MODELS = ["llama", "phi4", "qwen", "gemini", "grok", "gpt4"]
SHOT_TYPES = ["zero_shot", "few_shot"]
N_MODELS = len(PROMPT_MODELS)
N_ANNOTATORS = 3

# Gold reference from three annotators: soft score s_ij in {0, 1/3, 2/3, 1}.
GOLD_TAU_LENIENT = 1 / N_ANNOTATORS  # >=1 of 3 annotators
GOLD_TAU_MAJORITY = 2 / N_ANNOTATORS  # >=2 of 3 (Eq. 1 in paper)
GOLD_TAU_UNANIMOUS = 1.0  # 3 of 3

PRIORITY_CATEGORIES = [
    "not in my backyard",
    "provide a fact or claim",
    "express their opinion",
    "harmful generalization",
]

LABEL_COLUMNS = [
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

MODEL_DISPLAY = {
    "llama": "LLaMA",
    "phi4": "Phi-4",
    "qwen": "Qwen",
    "gemini": "Gemini",
    "grok": "Grok",
    "gpt4": "GPT-4.1",
}

COUNTERFACTUAL_EDITS: Dict[str, callable] = {
    "remove_affordable_housing": lambda t: re.sub(
        r"\baffordable housing\b", "housing", t, flags=re.IGNORECASE
    ),
    "strip_question": lambda t: t.replace("?", ""),
    "add_opposition_suffix": lambda t: (
        t.rstrip() + " I don't want a shelter in my neighborhood."
    ),
    "remove_housing_lexicon": lambda t: re.sub(
        r"\b(affordable housing|homeless|unhoused|shelter|housing|encampment)\b",
        "community",
        t,
        flags=re.IGNORECASE,
    ),
}


def _prediction_path(source: str, model: str, shot_type: str) -> Path:
    """Gold-subset flags path for requested shot type."""
    if shot_type not in SHOT_TYPES:
        raise ValueError(f"Unknown shot_type: {shot_type}")
    suffix = "none" if shot_type == "zero_shot" else source
    return (
        REPO_ROOT
        / "output"
        / source
        / model
        / f"classified_comments_{source}_gold_subset_{model}_{suffix}_flags.csv"
    )


def _load_soft(source: str) -> pd.DataFrame:
    path = REPO_ROOT / "output" / "annotation" / "soft_labels" / f"{source}_soft_labels.csv"
    return pd.read_csv(path)


def _gold_binary(soft: pd.Series, tau: float) -> np.ndarray:
    """Binarize soft scores at annotator vote fraction tau (1/3, 2/3, or 1)."""
    return (pd.to_numeric(soft, errors="coerce").fillna(0).values >= tau - 1e-9).astype(int)


def _load_preds(source: str, model: str, shot_type: str) -> pd.DataFrame | None:
    path = _prediction_path(source, model, shot_type)
    if not path.exists():
        return None
    df = pd.read_csv(path, low_memory=False)
    out = pd.DataFrame(index=df.index)
    for model_col, soft_col in MODEL_COL_TO_SOFT.items():
        if model_col in df.columns:
            out[soft_col] = (
                pd.to_numeric(df[model_col], errors="coerce").fillna(0).astype(int).clip(0, 1)
            )
    if out.shape[1] != len(LABEL_COLUMNS):
        return None
    return out


def build_long_predictions(*, gold_tau: float = GOLD_TAU_MAJORITY, shot_type: str) -> pd.DataFrame:
    """One row per (source, item_idx, model, category) for one shot type."""
    rows: List[dict] = []
    for source in SOURCES:
        soft = _load_soft(source)
        for model in PROMPT_MODELS:
            pred = _load_preds(source, model, shot_type)
            if pred is None:
                continue
            n = min(len(soft), len(pred))
            gold_bin = np.column_stack(
                [_gold_binary(soft[cat].iloc[:n], gold_tau) for cat in LABEL_COLUMNS]
            )
            pred_arr = pred[LABEL_COLUMNS].iloc[:n].astype(int).values
            for i in range(n):
                for j, cat in enumerate(LABEL_COLUMNS):
                    rows.append(
                        {
                            "source": source,
                            "item_idx": i,
                            "model": model,
                            "shot_type": shot_type,
                            "category": cat,
                            "gold": int(gold_bin[i, j]),
                            "pred": int(pred_arr[i, j]),
                        }
                    )
    return pd.DataFrame(rows)


def build_item_consensus(long: pd.DataFrame) -> pd.DataFrame:
    """Per item × category: gold, mean vote across 6 zero-shot models."""
    return (
        long.groupby(["source", "item_idx", "category"], as_index=False)
        .agg(
            gold=("gold", "first"),
            vote_fraction=("pred", "mean"),
            n_votes=("pred", "sum"),
            n_models=("pred", "count"),
        )
    )


def prevalence_gap_pp(gold: np.ndarray, pred: np.ndarray) -> float:
    return 100.0 * (float(pred.mean()) - float(gold.mean()))


def bootstrap_gap_ci(
    gold: np.ndarray,
    scores: np.ndarray,
    *,
    threshold: float,
    n_boot: int,
    seed: int,
) -> Tuple[float, float, float]:
    pred = (scores >= threshold).astype(int)
    gap = prevalence_gap_pp(gold, pred)
    rng = np.random.default_rng(seed)
    boot = []
    n = len(gold)
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        p = (scores[idx] >= threshold).astype(int)
        boot.append(prevalence_gap_pp(gold[idx], p))
    boot_arr = np.array(boot)
    return gap, float(np.percentile(boot_arr, 2.5)), float(np.percentile(boot_arr, 97.5))


def run_per_model_gaps(long: pd.DataFrame) -> pd.DataFrame:
    """Prevalence gap per model, pooled over sources (expects one shot_type)."""
    rows = []
    shot = str(long["shot_type"].iloc[0]) if "shot_type" in long.columns and len(long) else ""
    for model in PROMPT_MODELS:
        for cat in PRIORITY_CATEGORIES:
            sub = long[(long["model"] == model) & (long["category"] == cat)]
            if sub.empty:
                continue
            gold = sub.groupby(["source", "item_idx"])["gold"].first().values
            pred = sub.groupby(["source", "item_idx"])["pred"].first().values
            rows.append(
                {
                    "model": model,
                    "model_display": MODEL_DISPLAY[model],
                    "shot_type": shot,
                    "category": cat,
                    "gap_pp": prevalence_gap_pp(gold, pred),
                    "pi_ref": gold.mean() * 100,
                    "pi_model": pred.mean() * 100,
                    "n": len(gold),
                    "gold_positives": int(gold.sum()),
                }
            )
    return pd.DataFrame(rows)


def run_per_model_by_source(long: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for source in SOURCES:
        for model in PROMPT_MODELS:
            for cat in PRIORITY_CATEGORIES:
                sub = long[
                    (long["source"] == source)
                    & (long["model"] == model)
                    & (long["category"] == cat)
                ]
                if sub.empty:
                    continue
                gold = sub.groupby("item_idx")["gold"].first().values
                pred = sub.groupby("item_idx")["pred"].first().values
                rows.append(
                    {
                        "source": source,
                        "model": model,
                        "category": cat,
                        "gap_pp": prevalence_gap_pp(gold, pred),
                        "gold_positives": int(gold.sum()),
                        "low_support": int(gold.sum()) < 5,
                    }
                )
    return pd.DataFrame(rows)


def run_gold_tau_sweep(long: pd.DataFrame) -> pd.DataFrame:
    """Sensitivity at 1-of-3 vs 2-of-3 annotator votes (soft scores 1/3 and 2/3)."""
    tau_specs = [
        (GOLD_TAU_LENIENT, "1_of_3", "≥1/3"),
        (GOLD_TAU_MAJORITY, "2_of_3", "≥2/3"),
    ]
    rows = []
    for source in SOURCES:
        soft = _load_soft(source)
        for tau, rule_id, rule_label in tau_specs:
            for cat in PRIORITY_CATEGORIES:
                if cat not in soft.columns:
                    continue
                sub = long[(long["source"] == source) & (long["category"] == cat)]
                if sub.empty:
                    continue
                n = min(len(soft), sub["item_idx"].nunique())
                gold = _gold_binary(soft[cat].iloc[:n], tau)
                vote = sub.groupby("item_idx")["pred"].mean().values[:n]
                gold = gold[: len(vote)]
                rows.append(
                    {
                        "gold_rule": rule_id,
                        "gold_rule_label": rule_label,
                        "gold_tau": tau,
                        "source": source,
                        "category": cat,
                        "gap_pp": prevalence_gap_pp(gold, (vote >= 0.5).astype(int)),
                        "pi_ref": gold.mean() * 100,
                        "pi_model": (vote >= 0.5).mean() * 100,
                        "n": len(gold),
                        "gold_positives": int(gold.sum()),
                    }
                )
    return pd.DataFrame(rows)


def run_vote_threshold_sweep(consensus: pd.DataFrame, thresholds: np.ndarray) -> pd.DataFrame:
    rows = []
    for cat in PRIORITY_CATEGORIES:
        sub = consensus[consensus["category"] == cat]
        if sub.empty:
            continue
        gold = sub["gold"].values.astype(int)
        votes = sub["vote_fraction"].values
        for thr in thresholds:
            pred = (votes >= thr).astype(int)
            rows.append(
                {
                    "category": cat,
                    "vote_threshold": float(thr),
                    "gap_pp": prevalence_gap_pp(gold, pred),
                    "f1": _f1_binary(gold, pred),
                    "pi_ref": gold.mean() * 100,
                    "pi_pred": pred.mean() * 100,
                }
            )
    return pd.DataFrame(rows)


def _f1_binary(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    return (2 * prec * rec / (prec + rec)) if (prec + rec) else 0.0


def expected_calibration_error(
    y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10
) -> float:
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(y_true)
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        if i < n_bins - 1:
            mask = (y_prob >= lo) & (y_prob < hi)
        else:
            mask = (y_prob >= lo) & (y_prob <= hi)
        if not mask.any():
            continue
        acc = y_true[mask].mean()
        conf = y_prob[mask].mean()
        ece += mask.sum() / n * abs(acc - conf)
    return float(ece)


def reliability_bins(
    y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10
) -> pd.DataFrame:
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    rows = []
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        if i < n_bins - 1:
            mask = (y_prob >= lo) & (y_prob < hi)
        else:
            mask = (y_prob >= lo) & (y_prob <= hi)
        rows.append(
            {
                "bin_lo": lo,
                "bin_hi": hi,
                "count": int(mask.sum()),
                "mean_vote_fraction": float(y_prob[mask].mean()) if mask.any() else np.nan,
                "empirical_positive_rate": float(y_true[mask].mean()) if mask.any() else np.nan,
            }
        )
    return pd.DataFrame(rows)


def run_source_prevalence_gaps(consensus: pd.DataFrame, n_boot: int) -> pd.DataFrame:
    """6-model vote fraction >= 0.5, by source."""
    rows = []
    for source in SOURCES:
        for cat in PRIORITY_CATEGORIES:
            sub = consensus[(consensus["source"] == source) & (consensus["category"] == cat)]
            if sub.empty:
                continue
            gold = sub["gold"].values.astype(int)
            votes = sub["vote_fraction"].values
            gap, lo, hi = bootstrap_gap_ci(
                gold, votes, threshold=0.5, n_boot=n_boot, seed=42 + hash(cat) % 1000
            )
            rows.append(
                {
                    "source": source,
                    "category": cat,
                    "gap_pp": gap,
                    "ci_lo": lo,
                    "ci_hi": hi,
                    "pi_ref": gold.mean() * 100,
                    "pi_model": (votes >= 0.5).mean() * 100,
                    "n": len(gold),
                    "gold_positives": int(gold.sum()),
                    "low_support": int(gold.sum()) < 5,
                }
            )
    return pd.DataFrame(rows)


def run_pooled_bootstrap(consensus: pd.DataFrame, n_boot: int) -> pd.DataFrame:
    rows = []
    for cat in PRIORITY_CATEGORIES:
        sub = consensus[consensus["category"] == cat]
        gold = sub["gold"].values.astype(int)
        votes = sub["vote_fraction"].values
        gap, lo, hi = bootstrap_gap_ci(gold, votes, threshold=0.5, n_boot=n_boot, seed=7)
        rows.append(
            {
                "category": cat,
                "gap_pp": gap,
                "ci_lo": lo,
                "ci_hi": hi,
                "n": len(gold),
                "gold_positives": int(gold.sum()),
            }
        )
    return pd.DataFrame(rows)


def _trigger_flags(text: str) -> Dict[str, bool]:
    t = str(text).lower()
    return {
        "affordable_housing": bool(re.search(r"\baffordable housing\b", t)),
        "question_mark": "?" in t,
        "opposition_verbs": bool(
            re.search(
                r"\b(oppose|opposed|against|don't want|do not want|block|prevent)\b", t
            )
        ),
        "housing_general": bool(
            re.search(r"\b(housing|shelter|homeless|unhoused)\b", t)
        ),
    }


def load_texts_for_nimby() -> pd.DataFrame:
    chunks = []
    for source in SOURCES:
        soft = _load_soft(source)
        path = _prediction_path(source, "gpt4", "zero_shot")
        if not path.exists():
            continue
        raw = pd.read_csv(path, low_memory=False)
        text_col = "Comment" if "Comment" in raw.columns else raw.columns[0]
        preds = []
        for model in PROMPT_MODELS:
            p = _load_preds(source, model, "zero_shot")
            if p is not None:
                preds.append(p["not in my backyard"].values)
        if not preds:
            continue
        n = min(len(soft), len(raw), *(len(p) for p in preds))
        votes = np.mean(np.stack([p[:n] for p in preds], axis=0), axis=0)
        gold = _gold_binary(soft["not in my backyard"].iloc[:n], GOLD_TAU_MAJORITY)
        chunks.append(
            pd.DataFrame(
                {
                    "source": source,
                    "text": raw[text_col].iloc[:n].astype(str),
                    "gold_nimby": gold,
                    "vote_fraction": votes,
                    "n_model_votes": (votes * N_MODELS).round().astype(int),
                    "consensus_fp": ((gold == 0) & (votes >= 3 / N_MODELS)).astype(int),
                }
            )
        )
    return pd.concat(chunks, ignore_index=True)


def run_counterfactual_lexical(nimby_df: pd.DataFrame) -> pd.DataFrame:
    fps = nimby_df[nimby_df["consensus_fp"] == 1].copy()
    rows = []
    for _, row in fps.iterrows():
        orig = row["text"]
        before = _trigger_flags(orig)
        for edit_name, fn in COUNTERFACTUAL_EDITS.items():
            edited = fn(orig)
            if edited == orig:
                continue
            after = _trigger_flags(edited)
            rows.append(
                {
                    "source": row["source"],
                    "edit": edit_name,
                    "vote_fraction_original": row["vote_fraction"],
                    "original_text": orig[:300],
                    "counterfactual_text": edited[:300],
                    "triggers_removed": sum(before[k] and not after[k] for k in before),
                    **{f"before_{k}": before[k] for k in before},
                    **{f"after_{k}": after[k] for k in after},
                }
            )
    return pd.DataFrame(rows)


def build_reliability_bin_table(consensus: pd.DataFrame, min_count: int = 20) -> pd.DataFrame:
    """Long table of vote-fraction bins for paper (non-empty bins with enough support)."""
    rows: List[dict] = []
    for cat in PRIORITY_CATEGORIES:
        sub = consensus[consensus["category"] == cat]
        if sub.empty:
            continue
        y = sub["gold"].values.astype(float)
        p = sub["vote_fraction"].values
        for _, b in reliability_bins(y, p).iterrows():
            if b["count"] < min_count or np.isnan(b["mean_vote_fraction"]):
                continue
            rows.append(
                {
                    "category": cat,
                    "bin_lo": b["bin_lo"],
                    "bin_hi": b["bin_hi"],
                    "count": b["count"],
                    "mean_vote_fraction": b["mean_vote_fraction"],
                    "gold_positive_rate": b["empirical_positive_rate"],
                    "calibration_gap_pp": 100.0
                    * (b["empirical_positive_rate"] - b["mean_vote_fraction"]),
                }
            )
    return pd.DataFrame(rows)


def write_latex_fragments(
    pooled: pd.DataFrame,
    by_source: pd.DataFrame,
    per_model: pd.DataFrame,
    ece_table: pd.DataFrame,
    reliability_bins_df: pd.DataFrame,
    out_dir: Path,
) -> None:
    latex_dir = out_dir / "latex"
    latex_dir.mkdir(parents=True, exist_ok=True)

    def _ci_cell(lo: float, hi: float) -> str:
        return f"[{lo:.1f}, {hi:.1f}]"

    cat_display = {
        "not in my backyard": "Not in my backyard",
        "provide a fact or claim": "Provide fact/claim",
        "express their opinion": "Express opinion",
        "harmful generalization": "Harmful generalization",
    }

    lines = [
        "% Zero-shot prompt LLMs only (6 models)",
        "\\begin{table}[t]",
        "\\centering",
        "\\small",
        "\\begin{tabular}{@{}l r c r@{}}",
        "\\toprule",
        "\\textbf{Category} & \\textbf{Gap} & \\textbf{95\\% CI} & $n_{+}$ \\\\",
        "\\midrule",
    ]
    for _, r in pooled.iterrows():
        lines.append(
            f"{cat_display.get(r['category'], r['category'])} & {r['gap_pp']:+.1f} & "
            f"{_ci_cell(r['ci_lo'], r['ci_hi'])} & {int(r['gold_positives'])} \\\\"
        )
    lines += [
        "\\bottomrule",
        "\\end{tabular}",
        "\\caption{Pooled prevalence gaps: six zero-shot prompt LLMs, vote fraction $\\geq 0.5$.}",
        "\\label{tab:prevalence_gap_ci}",
        "\\end{table}",
    ]
    (latex_dir / "prevalence_gap_ci.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")

    # Per-model calibration table (priority categories)
    pm = per_model.pivot(index="category", columns="model_display", values="gap_pp")
    pm = pm.reindex(PRIORITY_CATEGORIES)
    tbl = [
        "% Per-model zero-shot prevalence gaps (pp)",
        "\\begin{table*}[!tbp]",
        "\\centering",
        "\\footnotesize",
        "\\begin{tabular}{@{}l" + "r" * len(PROMPT_MODELS) + "r@{}}",
        "\\toprule",
        "\\textbf{Category} & "
        + " & ".join(MODEL_DISPLAY[m] for m in PROMPT_MODELS)
        + " & \\textbf{Mean} \\\\",
        "\\midrule",
    ]
    for cat in PRIORITY_CATEGORIES:
        row = per_model[per_model["category"] == cat]
        if row.empty:
            continue
        cells = []
        for m in PROMPT_MODELS:
            v = row.loc[row["model"] == m, "gap_pp"]
            cells.append(f"{v.iloc[0]:+.1f}" if len(v) else "---")
        mean_gap = row["gap_pp"].mean()
        tbl.append(
            f"{cat_display.get(cat, cat)} & "
            + " & ".join(cells)
            + f" & {mean_gap:+.1f} \\\\"
        )
    tbl += [
        "\\bottomrule",
        "\\end{tabular}",
        "\\caption{Prevalence gap (pp) by zero-shot model on gold ($\\hat{\\pi}-\\pi_{\\mathrm{ref}}$). Mean = unweighted average of six models.}",
        "\\label{tab:calibration_by_model_zeroshot}",
        "\\end{table*}",
    ]
    (latex_dir / "calibration_by_model_zeroshot.tex").write_text("\n".join(tbl) + "\n", encoding="utf-8")

    src_lines = [
        "% Source-wise gaps, six zero-shot models, vote $\\geq 0.5$",
        "\\begin{table*}[t]",
        "\\centering",
        "\\footnotesize",
        "\\begin{tabular}{@{}l l r c r@{}}",
        "\\toprule",
        "\\textbf{Source} & \\textbf{Category} & \\textbf{Gap} & \\textbf{95\\% CI} & $n_{+}$ \\\\",
        "\\midrule",
    ]
    for _, r in by_source.sort_values(["source", "category"]).iterrows():
        flag = "{$^*$}" if r["low_support"] else ""
        src_lines.append(
            f"{r['source'].replace('_', ' ').title()} & {cat_display.get(r['category'], r['category'])} & "
            f"{r['gap_pp']:+.1f} & {_ci_cell(r['ci_lo'], r['ci_hi'])}{flag} & {int(r['gold_positives'])} \\\\"
        )
    src_lines += [
        "\\bottomrule",
        "\\end{tabular}",
        "\\caption{Prevalence gaps by source (six zero-shot models). {$^*$}$<5$ gold positives.}",
        "\\label{tab:prevalence_gap_by_source}",
        "\\end{table*}",
    ]
    (latex_dir / "prevalence_gap_by_source.tex").write_text("\n".join(src_lines) + "\n", encoding="utf-8")

    if not ece_table.empty:
        ece_lines = [
            "% ECE: vote fraction vs gold (6 zero-shot models)",
            "\\begin{tabular}{@{}l r@{}}\\toprule",
            "\\textbf{Category} & \\textbf{ECE} \\\\\\midrule",
        ]
        for _, r in ece_table.iterrows():
            ece_lines.append(
                f"{cat_display.get(r['category'], r['category'])} & {r['ece']:.3f} \\\\"
            )
        ece_lines += ["\\bottomrule\\end{tabular}"]
        (latex_dir / "ece_summary.tex").write_text("\n".join(ece_lines) + "\n", encoding="utf-8")

    if not reliability_bins_df.empty:
        reliability_bins_df.to_csv(out_dir / "reliability_bins_paper.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prompt calibration audit on the gold-standard set"
    )
    parser.add_argument("--bootstrap", type=int, default=5000)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--shots", nargs="+", default=["zero_shot", "few_shot"], choices=SHOT_TYPES)
    args = parser.parse_args()
    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)

    all_rows = []
    for shot in args.shots:
        print(
            f"Loading {shot} predictions ({N_MODELS} models); "
            f"gold reference = 2-of-{N_ANNOTATORS} annotators (τ=2/3)..."
        )
        long = build_long_predictions(gold_tau=GOLD_TAU_MAJORITY, shot_type=shot)
        long.to_csv(out / f"predictions_long_{shot}.csv", index=False)

        consensus = build_item_consensus(long)
        consensus.to_csv(out / f"item_vote_fraction_{shot}.csv", index=False)

        per_model = run_per_model_gaps(long)
        per_model.to_csv(out / f"prevalence_gap_per_model_{shot}.csv", index=False)

        print(f"[{shot}] Gold τ sweep + vote-fraction threshold sweep...")
        tau_df = run_gold_tau_sweep(long)
        tau_df.to_csv(out / f"gold_tau_threshold_sweep_{shot}.csv", index=False)
        thr_grid = np.round(np.arange(1 / N_MODELS, 1.0, 1 / N_MODELS), 4)
        thr_df = run_vote_threshold_sweep(consensus, thr_grid)
        thr_df.to_csv(out / f"vote_threshold_sweep_{shot}.csv", index=False)

        print(f"[{shot}] Bootstrap CIs (pooled + by source, vote >= 0.5)...")
        pooled = run_pooled_bootstrap(consensus, args.bootstrap)
        pooled["shot_type"] = shot
        pooled.to_csv(out / f"prevalence_gap_bootstrap_pooled_{shot}.csv", index=False)
        by_source = run_source_prevalence_gaps(consensus, args.bootstrap)
        by_source["shot_type"] = shot
        by_source.to_csv(out / f"prevalence_gap_by_source_{shot}.csv", index=False)

        print(f"[{shot}] ECE and reliability bins (vote fraction)...")
        ece_rows = []
        for cat in PRIORITY_CATEGORIES:
            sub = consensus[consensus["category"] == cat]
            if sub.empty or sub["gold"].sum() < 3:
                continue
            y = sub["gold"].values.astype(float)
            p = sub["vote_fraction"].values
            ece = expected_calibration_error(y, p)
            ece_rows.append(
                {"category": cat, "ece": ece, "n": len(y), "gold_positives": int(y.sum()), "shot_type": shot}
            )
            reliability_bins(y, p).to_csv(
                out / f"reliability_bins_{cat.replace('/', '_')}_{shot}.csv", index=False
            )
        ece_df = pd.DataFrame(ece_rows)
        ece_df.to_csv(out / f"ece_summary_{shot}.csv", index=False)
        rel_bin_df = build_reliability_bin_table(consensus, min_count=20)
        if not rel_bin_df.empty:
            rel_bin_df["shot_type"] = shot
            rel_bin_df.to_csv(out / f"reliability_bins_paper_{shot}.csv", index=False)

        write_latex_fragments(pooled, by_source, per_model, ece_df, rel_bin_df, out / shot)

        all_rows.append(
            pooled[["category", "gap_pp", "ci_lo", "ci_hi", "gold_positives", "shot_type"]].copy()
        )

    if all_rows:
        both = pd.concat(all_rows, ignore_index=True)
        both.to_csv(out / "pooled_gap_compare_zero_vs_few.csv", index=False)
        print("\n=== Pooled gap compare (vote>=0.5) ===")
        print(both.pivot(index="category", columns="shot_type", values="gap_pp").to_string())

    print(f"\nWrote outputs to {out}")


if __name__ == "__main__":
    main()
