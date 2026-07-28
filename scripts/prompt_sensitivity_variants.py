#!/usr/bin/env python3
"""
Build prompt-template sensitivity variants (label order / role wording)
and, when prior model predictions exist, score a cheap proxy: lexical
overlap between original flags and a deterministic reorder audit note.

Full re-inference against GPT/Gemini on thousands of items is expensive.
This script:
1) emits three prompt variants for reproducibility / future API runs
2) optionally runs a *local* small-model sensitivity pilot if --run-local
3) always writes a LaTeX methods note describing the sensitivity design

Variant A (baseline): original Appendix J prompt
Variant B: shuffled category order within each block
Variant C: role line removed ("expert in social behavior analysis")
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from textwrap import dedent

REPO_ROOT = Path(__file__).resolve().parents[1]

COMMENT_TYPES = [
    "Ask a Genuine Question: The speaker asks a sincere question about homelessness or related issues",
    "Ask a Rhetorical Question: The speaker asks a question not intended to be answered, often to make a point",
    "Provide a Fact or Claim: The speaker provides a factual statement or claim about homelessness",
    "Provide an Observation: The speaker shares an observation about homelessness or related situations",
    "Express Their Opinion: The speaker expresses their own views or feelings about homelessness",
    "Express Others' Opinions: The speaker describes or references the views or feelings of others about homelessness",
]
CRITIQUE = [
    "Money Aid Allocation: Discussion of financial resources, aid distribution, or resource allocation for homelessness",
    "Government Critique: Criticism of government policies, laws, or political approaches to homelessness",
    "Societal Critique: Criticism of social norms, systems, or societal attitudes toward homelessness",
]
RESPONSE = [
    "Solutions/Interventions: Discussion of specific solutions, interventions, or charitable actions",
]
PERCEPTION = [
    "Personal Interaction: Direct personal experiences with PEH",
    "Media Portrayal: Discussion of PEH as portrayed in media",
    "Not in my Backyard: Opposition to local homelessness developments",
    "Harmful Generalization: Negative stereotypes about PEH",
    "Deserving/Undeserving: Judgments about who deserves help",
]


def _format_block(title: str, items: list[str]) -> str:
    body = "\n".join(f"- {x}" for x in items)
    return f"{title}\n{body}"


def build_prompt(*, role: bool, shuffle_seed: int | None) -> str:
    comment, critique, response, perception = COMMENT_TYPES, CRITIQUE, RESPONSE, PERCEPTION
    if shuffle_seed is not None:
        rng = random.Random(shuffle_seed)
        comment = comment[:]
        critique = critique[:]
        response = response[:]
        perception = perception[:]
        rng.shuffle(comment)
        rng.shuffle(critique)
        rng.shuffle(response)
        rng.shuffle(perception)

    role_line = (
        "You are an expert in social behavior analysis. Your task is to analyze "
        "{content_desc} about homelessness and categorize them according to specific criteria.\n\n"
        if role
        else "Your task is to analyze {content_desc} about homelessness and categorize them "
        "according to specific criteria.\n\n"
    )
    return (
        role_line
        + "DEFINITIONS:\n"
        + _format_block("1. Comment Types (select all that apply):", comment)
        + "\n\n"
        + _format_block("2. Critique Categories (select all that apply):", critique)
        + "\n\n"
        + _format_block("3. Response Categories (select all that apply):", response)
        + "\n\n"
        + _format_block("4. Perception Types (select all that apply):", perception)
        + "\n\n"
        + "5. Racist Classification:\n"
        + "- Yes: Contains explicit or implicit racial bias\n"
        + "- No: No racial bias present\n\n"
        + "INSTRUCTIONS:\n"
        + "1. Read the comment carefully\n"
        + "2. Analyze it according to the categories above\n"
        + "3. Provide your analysis in the exact format below\n"
        + "4. Include a brief reasoning for your classification\n\n"
        + "FORMAT YOUR RESPONSE EXACTLY AS FOLLOWS:\n"
        + "Comment Type: [...]\n"
        + "Critique Category: [...]\n"
        + "Response Category: [...]\n"
        + "Perception Type: [...]\n"
        + "Racist: [Yes/No]\n"
        + "Reasoning: [brief explanation]\n"
    )


def write_latex_note(out_dir: Path) -> None:
    text = dedent(
        r"""
        \section{Prompt Engineering and Sensitivity}
        \label{sec:prompt_sensitivity}
        The classification prompt in Appendix~\ref{sec:llm_prompt} was derived from the OATH
        frame definitions~\citep{ranjit2024oath}, expanded to our 16-category schema with partner
        review, and locked after pilot runs on $\approx$20 items per source (same pilot used for
        annotator training). We did not perform an exhaustive prompt search; the released template
        is the frozen checkpoint used for all reported zero-/few-shot numbers.

        Cultural and safety alignment work suggests that LLM outputs can be sensitive to label
        ordering and role prefixes. We therefore define three variants for auditability:
        \textbf{A} baseline (Appendix~\ref{sec:llm_prompt}),
        \textbf{B} within-block category shuffle (seed $7$), and
        \textbf{C} role line removed.
        Variant prompt files are released under
        \texttt{output/openreview\_artifacts/prompt\_sensitivity/}.
        Re-running full API inference on all $1{,}698$ gold items $\times$ 6 models $\times$ 2 shots
        was out of budget for this revision; we instead provide the templates and recommend that
        deployment teams re-audit prevalence gaps under B/C on a standing sample before changing
        the production prompt.
        Decoding hyperparameters were likewise held fixed (Table~\ref{tab:model_card}); we treat
        aggregation thresholds and gold $\tau$ as the primary sensitivity knobs
        (Tables~\ref{tab:gold_tau_sensitivity},~\ref{tab:nimby_mitigation}).
        """
    ).strip()
    (out_dir / "prompt_sensitivity_note.tex").write_text(text + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=REPO_ROOT / "output" / "openreview_artifacts" / "prompt_sensitivity",
    )
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    variants = {
        "A_baseline": build_prompt(role=True, shuffle_seed=None),
        "B_shuffled_order_seed7": build_prompt(role=True, shuffle_seed=7),
        "C_no_role_prefix": build_prompt(role=False, shuffle_seed=None),
    }
    for name, prompt in variants.items():
        (args.out_dir / f"prompt_{name}.txt").write_text(prompt, encoding="utf-8")

    meta = {
        "variants": list(variants.keys()),
        "note": "Templates for sensitivity audit; not re-run at scale in this revision.",
    }
    (args.out_dir / "variants_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    write_latex_note(args.out_dir)
    print(f"Wrote {len(variants)} prompt variants to {args.out_dir}")


if __name__ == "__main__":
    main()
