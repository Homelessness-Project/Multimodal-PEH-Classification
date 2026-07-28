# IAA low-κ diagnosis

- Mean pairwise observed agreement $P_o$ across categories: **0.860**
- Mean κ-like: **0.309**; mean PABAK: **0.720**
- Mean soft-score unanimity: **78.4%** (paper reports 78.38% mean per-cell unanimity)
- Mechanism cell counts: {'sparsity_degeneracy': 29, 'boundary_subjectivity': 19, 'moderate_agreement': 12, 'high_base_rate_kappa_paradox': 4}
- Spearman(κ_paper, prevalence) on cells with ≥5 gold+: **0.318**
- Spearman(κ_paper, split%): **-0.047**
- Meeting-minutes fact/claim: {'P_o': 0.8076923076923077, 'kappa_like': 0.031096839614669945, 'PABAK': 0.6153846153846154, 'split_rate': 0.28846153846153844, 'prevalence_2of3': 0.9642857142857143, 'mechanism': 'high_base_rate_kappa_paradox', 'paper_kappa': 0.31}

Conclusion: low κ is a *mixture* of (i) sparse-label chance agreement on absences,
(ii) high-base-rate κ paradox on ubiquitous frames like fact/claim, and
(iii) genuine boundary subjectivity on opinionative/policy frames—not evidence that
the gold set is random.
