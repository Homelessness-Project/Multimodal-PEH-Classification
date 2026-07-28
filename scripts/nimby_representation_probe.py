#!/usr/bin/env python3
"""
Beyond lexical triggers: NIMBY error analysis in representation space.

Addresses Reviewer unx3 (point 4): surface-level lexical analysis is insufficient.
This script probes *semantic/geometry* failures of NIMBY tagging using a frozen
encoder + linear probes and controlled counterfactual edits:

1) Centroid geometry: consensus FP / TN / gold TP in CLS embedding space
2) Linear probes: predict gold NIMBY vs predict multi-model consensus FP
3) Counterfactual edit sensitivity (representation-mediated): re-encode edited
   text and measure probe score Δ. Edits include surface (strip '?') and
   *semantic* interventions that keep housing lexicon while removing opposition
   meaning (and vice versa).

Closed APIs do not expose internals; we treat a public encoder representation as a
shared analysis substrate for describing the geometry of texts that models
systematically mis-tag (standard probing design in NLP).

Usage (repo root):
    MPLCONFIGDIR=$PWD/.mplconfig XDG_CACHE_HOME=$PWD/.cache MPLBACKEND=Agg \\
      .venv/bin/python scripts/nimby_representation_probe.py
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

HAS_TRANSFORMERS = False  # lazy-checked only for --backend transformer
torch = None  # type: ignore
AutoModel = AutoTokenizer = None  # type: ignore

REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCES = ["reddit", "news", "meeting_minutes", "x"]
PROMPT_MODELS = ["llama", "phi4", "qwen", "gemini", "grok", "gpt4"]
N_MODELS = len(PROMPT_MODELS)


def _soft_path(source: str) -> Path:
    return REPO_ROOT / "output" / "annotation" / "soft_labels" / f"{source}_soft_labels.csv"


def _gold_text_spec(source: str) -> Tuple[Path, str]:
    m = {
        "reddit": (REPO_ROOT / "gold_standard" / "sampled_reddit_comments.csv", "Comment"),
        "x": (REPO_ROOT / "gold_standard" / "sampled_twitter_posts.csv", "Deidentified_text"),
        "news": (
            REPO_ROOT / "gold_standard" / "sampled_lexisnexis_news.csv",
            "Deidentified_paragraph_text",
        ),
        "meeting_minutes": (
            REPO_ROOT / "gold_standard" / "sampled_meeting_minutes.csv",
            "Deidentified_paragraph",
        ),
    }
    return m[source]


def _pred_path(source: str, model: str) -> Path:
    return (
        REPO_ROOT
        / "output"
        / source
        / model
        / f"classified_comments_{source}_gold_subset_{model}_none_flags.csv"
    )


def load_items() -> pd.DataFrame:
    rows = []
    for source in SOURCES:
        soft = pd.read_csv(_soft_path(source))
        text_path, text_col = _gold_text_spec(source)
        texts = pd.read_csv(text_path, low_memory=False)
        n = min(len(soft), len(texts))
        soft = soft.iloc[:n].reset_index(drop=True)
        texts = texts.iloc[:n].reset_index(drop=True)

        votes = []
        for model in PROMPT_MODELS:
            p = _pred_path(source, model)
            if not p.exists():
                continue
            df = pd.read_csv(p, low_memory=False)
            col = "Perception_not in my backyard"
            if col not in df.columns:
                continue
            y = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int).clip(0, 1).values
            votes.append(y[:n])
        if not votes:
            continue
        vote_mat = np.stack(votes, axis=0)
        vote_frac = vote_mat.mean(axis=0)
        gold = (pd.to_numeric(soft["not in my backyard"], errors="coerce").fillna(0).values >= 2 / 3 - 1e-9).astype(
            int
        )
        consensus_fp = ((gold == 0) & (vote_frac >= 3 / N_MODELS)).astype(int)
        tn = ((gold == 0) & (vote_frac < 3 / N_MODELS)).astype(int)
        tp = (gold == 1).astype(int)

        for i in range(n):
            rows.append(
                {
                    "source": source,
                    "item_idx": i,
                    "text": str(texts.loc[i, text_col]),
                    "gold": int(gold[i]),
                    "vote_fraction": float(vote_frac[i]),
                    "consensus_fp": int(consensus_fp[i]),
                    "is_tn": int(tn[i]),
                    "is_tp": int(tp[i]),
                }
            )
    return pd.DataFrame(rows)


EDIT_FNS = {
    "strip_question": lambda t: t.replace("?", ""),
    "remove_housing_lexicon": lambda t: re.sub(
        r"\b(affordable housing|homeless|unhoused|shelter|housing|encampment)\b",
        "community",
        t,
        flags=re.IGNORECASE,
    ),
    "keep_housing_neutralize_opposition": lambda t: re.sub(
        r"\b(oppose|opposed|against|don't want|do not want|dont want|block|prevent|reject|refuse|fight)\b",
        "discuss",
        t,
        flags=re.IGNORECASE,
    ),
    "add_explicit_opposition": lambda t: (
        t.rstrip() + " I don't want a shelter in my neighborhood."
    ),
    "near_to_far_proximity": lambda t: re.sub(
        r"\b(in my (neighborhood|neighbourhood|area|backyard)|nearby|next door|local)\b",
        "far away in another city",
        t,
        flags=re.IGNORECASE,
    ),
}


def fit_tfidf_svd(
    texts: List[str], *, n_components: int, seed: int
) -> Tuple[Pipeline, np.ndarray]:
    """Latent semantic subspace: stronger than binary regex triggers, offline-safe."""
    tfidf = TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, 2),
        min_df=2,
        max_features=40000,
        sublinear_tf=True,
    )
    X_tfidf = tfidf.fit_transform(texts)
    n_comp = min(n_components, max(2, X_tfidf.shape[1] - 1), X_tfidf.shape[0] - 1)
    svd = TruncatedSVD(n_components=n_comp, random_state=seed)
    X = svd.fit_transform(X_tfidf)
    pipe = Pipeline([("tfidf", tfidf), ("svd", svd)])
    return pipe, np.asarray(X)


def encode_with_pipe(pipe: Pipeline, texts: List[str]) -> np.ndarray:
    return np.asarray(pipe.transform(texts))


def encode_texts_transformer(
    texts: List[str],
    *,
    model_name: str,
    batch_size: int,
    max_length: int,
    device: str,
) -> np.ndarray:
    global HAS_TRANSFORMERS, torch, AutoModel, AutoTokenizer
    if not HAS_TRANSFORMERS:
        import torch as _torch
        from transformers import AutoModel as _AM, AutoTokenizer as _AT

        torch = _torch
        AutoModel, AutoTokenizer = _AM, _AT
        HAS_TRANSFORMERS = True
    assert torch is not None
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
    model.eval()
    vecs = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            enc = tok(
                batch,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            enc = {k: v.to(device) for k, v in enc.items()}
            out = model(**enc)
            hidden = out.last_hidden_state
            mask = enc["attention_mask"].unsqueeze(-1).float()
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
            vecs.append(pooled.cpu().numpy())
    return np.vstack(vecs)


def centroid(X: np.ndarray) -> np.ndarray:
    return X.mean(axis=0)


def cos(a: np.ndarray, b: np.ndarray) -> float:
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na < 1e-12 or nb < 1e-12:
        return float("nan")
    return float(np.dot(a, b) / (na * nb))


def cv_auc(X: np.ndarray, y: np.ndarray, seed: int = 0) -> float:
    y = y.astype(int)
    if y.sum() < 5 or (len(y) - y.sum()) < 5:
        return float("nan")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    scores = []
    for tr, te in skf.split(X, y):
        scaler = StandardScaler()
        Xtr = scaler.fit_transform(X[tr])
        Xte = scaler.transform(X[te])
        clf = LogisticRegression(max_iter=2000, class_weight="balanced")
        clf.fit(Xtr, y[tr])
        p = clf.predict_proba(Xte)[:, 1]
        scores.append(roc_auc_score(y[te], p))
    return float(np.mean(scores))


def fit_probe(X: np.ndarray, y: np.ndarray) -> Tuple[StandardScaler, LogisticRegression]:
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    clf = LogisticRegression(max_iter=2000, class_weight="balanced")
    clf.fit(Xs, y.astype(int))
    return scaler, clf


def probe_scores(
    X: np.ndarray, scaler: StandardScaler, clf: LogisticRegression
) -> np.ndarray:
    return clf.predict_proba(scaler.transform(X))[:, 1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--backend",
        choices=["tfidf_svd", "transformer"],
        default="tfidf_svd",
        help="tfidf_svd works offline; transformer needs HF download (RoBERTa/BERT).",
    )
    ap.add_argument("--encoder", default="roberta-base")
    ap.add_argument("--svd-components", type=int, default=128)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--max-length", type=int, default=256)
    ap.add_argument("--max-fp-edits", type=int, default=117)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=REPO_ROOT / "output" / "openreview_artifacts" / "nimby_representation",
    )
    ap.add_argument("--device", default=None)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading items; encoding with backend={args.backend}...")
    df = load_items()
    df.to_csv(args.out_dir / "items_with_labels.csv", index=False)

    encode_edited = None
    if args.backend == "tfidf_svd":
        pipe, X = fit_tfidf_svd(
            df["text"].tolist(), n_components=args.svd_components, seed=args.seed
        )
        encoder_label = f"TF-IDF+SVD({args.svd_components})"

        def encode_edited(texts: List[str]) -> np.ndarray:
            return encode_with_pipe(pipe, texts)

    else:
        device = args.device or (
            "cuda" if HAS_TRANSFORMERS and torch.cuda.is_available() else "cpu"
        )
        X = encode_texts_transformer(
            df["text"].tolist(),
            model_name=args.encoder,
            batch_size=args.batch_size,
            max_length=args.max_length,
            device=device,
        )
        encoder_label = args.encoder

        def encode_edited(texts: List[str]) -> np.ndarray:
            return encode_texts_transformer(
                texts,
                model_name=args.encoder,
                batch_size=args.batch_size,
                max_length=args.max_length,
                device=device,
            )

    np.save(args.out_dir / "embeddings.npy", X)
    # stash for caption
    args._encoder_label = encoder_label  # type: ignore[attr-defined]
    args._encode_edited = encode_edited  # type: ignore[attr-defined]

    fp_mask = df["consensus_fp"].values == 1
    tn_mask = df["is_tn"].values == 1
    tp_mask = df["is_tp"].values == 1

    c_fp, c_tn, c_tp = centroid(X[fp_mask]), centroid(X[tn_mask]), centroid(X[tp_mask])
    geom = pd.DataFrame(
        [
            {
                "pair": "FP vs TN",
                "cosine": cos(c_fp, c_tn),
                "n_a": int(fp_mask.sum()),
                "n_b": int(tn_mask.sum()),
            },
            {
                "pair": "FP vs TP",
                "cosine": cos(c_fp, c_tp),
                "n_a": int(fp_mask.sum()),
                "n_b": int(tp_mask.sum()),
            },
            {
                "pair": "TN vs TP",
                "cosine": cos(c_tn, c_tp),
                "n_a": int(tn_mask.sum()),
                "n_b": int(tp_mask.sum()),
            },
        ]
    )
    geom.to_csv(args.out_dir / "centroid_cosine.csv", index=False)
    print(geom.to_string(index=False))

    # If FP closer to TP than TN → models latch onto TP-like geometry without gold opposition
    # If FP closer to TN → failures are thin decision boundary / overconfidence on near-TN items
    auc_gold = cv_auc(X, df["gold"].values)
    auc_fp = cv_auc(X, df["consensus_fp"].values)
    # Also: can we linearly separate FP from TN among gold-negatives?
    neg = df["gold"].values == 0
    auc_fp_vs_tn = cv_auc(X[neg], df.loc[neg, "consensus_fp"].values)
    probe_summary = pd.DataFrame(
        [
            {"target": "gold_NIMBY", "cv_auc": auc_gold},
            {"target": "consensus_FP", "cv_auc": auc_fp},
            {"target": "FP_vs_TN_among_gold_neg", "cv_auc": auc_fp_vs_tn},
        ]
    )
    probe_summary.to_csv(args.out_dir / "probe_auc.csv", index=False)
    print(probe_summary.to_string(index=False))

    # Fit probe to predict consensus FP (among all items) for edit sensitivity
    scaler, clf = fit_probe(X, df["consensus_fp"].values)
    base_scores = probe_scores(X, scaler, clf)
    df["fp_probe_score"] = base_scores

    fps = df[df["consensus_fp"] == 1].head(args.max_fp_edits).copy()
    edit_rows = []
    for edit_name, fn in EDIT_FNS.items():
        edited = [fn(t) for t in fps["text"].tolist()]
        # Skip no-ops
        changed = [i for i, (a, b) in enumerate(zip(fps["text"], edited)) if a != b]
        if not changed:
            continue
        sub = fps.iloc[changed]
        edited_sub = [edited[i] for i in changed]
        Xe = args._encode_edited(edited_sub)  # type: ignore[attr-defined]
        s1 = probe_scores(Xe, scaler, clf)
        s0 = sub["fp_probe_score"].values
        delta = s1 - s0
        edit_rows.append(
            {
                "edit": edit_name,
                "n_changed": len(changed),
                "mean_delta_fp_score": float(delta.mean()),
                "frac_score_down": float((delta < -0.02).mean()),
                "frac_score_up": float((delta > 0.02).mean()),
                "mean_before": float(s0.mean()),
                "mean_after": float(s1.mean()),
            }
        )
    edits = pd.DataFrame(edit_rows)
    edits.to_csv(args.out_dir / "edit_sensitivity.csv", index=False)
    print(edits.to_string(index=False))

    # LaTeX fragment
    lines = [
        "% Auto-generated NIMBY representation probe",
        "\\begin{table}[t]",
        "\\centering",
        "\\small",
        "\\begin{tabular}{@{}lrr@{}}",
        "\\toprule",
        "\\textbf{Centroid pair} & \\textbf{Cosine} & $n$ \\\\",
        "\\midrule",
    ]
    for _, r in geom.iterrows():
        lines.append(
            f"{r['pair']} & {r['cosine']:.3f} & {int(r['n_a'])}/{int(r['n_b'])} \\\\"
        )
    lines += [
        "\\midrule",
        f"Probe AUC (gold NIMBY) & \\multicolumn{{2}}{{l}}{{{auc_gold:.3f}}} \\\\",
        f"Probe AUC (consensus FP) & \\multicolumn{{2}}{{l}}{{{auc_fp:.3f}}} \\\\",
        f"Probe AUC (FP vs TN $|$ gold-) & \\multicolumn{{2}}{{l}}{{{auc_fp_vs_tn:.3f}}} \\\\",
        "\\bottomrule",
        "\\end{tabular}",
        "\\caption{TF--IDF+SVD(128) centroid cosines and linear-probe AUC for gold NIMBY vs.\\ consensus FPs "
        "($\\geq 3/6$). Edit sensitivity: Table~\\ref{tab:nimby_edit_sensitivity}.}",
        "\\label{tab:nimby_representation}",
        "\\end{table}",
    ]
    (args.out_dir / "nimby_representation.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")

    edit_label = {
        "strip_question": "Strip ``?''",
        "remove_housing_lexicon": "Drop housing lex.",
        "keep_housing_neutralize_opposition": "Neutralize opp.",
        "add_explicit_opposition": "Add opposition",
        "near_to_far_proximity": "Far proximity",
    }
    elines = [
        "% Auto-generated edit sensitivity",
        "\\begin{table}[!htb]",
        "\\centering",
        "\\scriptsize",
        "\\setlength{\\tabcolsep}{2pt}",
        "\\begin{tabularx}{\\columnwidth}{@{}>{\\raggedright\\arraybackslash}X r r r r@{}}",
        "\\toprule",
        "\\textbf{Edit} & $n$ & $\\Delta$ & $\\downarrow$ & $\\uparrow$ \\\\",
        "\\midrule",
    ]
    for _, r in edits.iterrows():
        key = str(r["edit"])
        name = edit_label.get(key, key.replace("_", " "))
        elines.append(
            f"{name} & {int(r['n_changed'])} & {r['mean_delta_fp_score']:+.3f} & "
            f"{100*r['frac_score_down']:.0f}\\% & {100*r['frac_score_up']:.0f}\\% \\\\"
        )
    elines += [
        "\\bottomrule",
        "\\end{tabularx}",
        "\\caption{Counterfactual edits on consensus NIMBY FPs ($n{=}117$). "
        "$\\Delta$: mean probe-score shift (FP-trained, not gold); "
        "$\\downarrow$/$\\uparrow$: share with lower/higher score.}",
        "\\label{tab:nimby_edit_sensitivity}",
        "\\end{table}",
    ]
    (args.out_dir / "nimby_edit_sensitivity.tex").write_text(
        "\n".join(elines) + "\n", encoding="utf-8"
    )

    note = """\\section{Beyond Lexical Triggers: Representation Probes}
\\label{sec:nimby_representation}
The main-text NIMBY error analysis reports co-occurring lexical cues on consensus false
positives. To address whether failures reflect only surface form, we probe a shared
embedding substrate (default: TF--IDF + truncated SVD latent semantics; optional: frozen
RoBERTa mean pooling when GPU/HF access is available). Closed LLM APIs used in our
benchmarks do not expose internals, so we ask whether the \\emph{texts} that models
systematically over-tag occupy a distinctive region of a shared semantic embedding space,
and whether controlled edits move those texts along a false-positive decision direction.
Table~\\ref{tab:nimby_representation} reports centroid geometry and linear-probe AUC;
Table~\\ref{tab:nimby_edit_sensitivity} reports representation-mediated sensitivities to
surface vs.\\ semantic counterfactuals. We treat this as complementary mechanistic evidence,
not a claim about specific layers inside GPT-4.1 / Gemini / Grok.
"""
    (args.out_dir / "nimby_representation_note.tex").write_text(note, encoding="utf-8")
    print(f"Wrote {args.out_dir}")


if __name__ == "__main__":
    main()
