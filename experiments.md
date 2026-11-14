# Experiments log

One line per run. Model selection uses **IDSSE leave-one-match-out CV only**. The public BAR-ATH
window is a bug check, never a selection criterion (RULES.md, EVALUATION.md).

| date | version | IDSSE-CV spellF1 | IDSSE-CV macroF1 | BAR-ATH pred. possessions (mean s) | public sanity spellF1 | note |
|---|---|---|---|---|---|---|
| 2026-11-14 | M0 baseline50 (organisers' rule, our pipeline) | – | – | 669 (7.6 s) | 0.3256 (26 poss., 10.8 s; GT 17, 14.2 s) | Byte-identical to organisers' baseline_*.csv (all four windows). Pipeline check only. |
| 2026-11-14 | M1 v1 draft (team switch cost 6.0) | – | – | 172 (22.0 s) | – (not scored) | GT-free check: 34 ≥1 s close-control spells by the other team ignored; the assumed 1.5 nats/frame evidence was really 1.2 (measured, no labels). Cost restated from the definition as 0.75 s of typical close control = 4.5. |
| 2026-11-14 | **M1 v1** (rules + kickoffs + tags + min-duration Viterbi) | not yet (M2) | not yet (M2) | 193 (19.8 s), 101 D stretches, D share 0.40 | 0.4429 (11 poss., 25.2 s; GT 17, 14.2 s), macroF1 0.7182 | First submission. sha256 80a5f235…0f09. Public run once, after the config was fixed. |
| 2026-11-14 | M1 v1 without Smart Tagging (ablation) | – | – | 218 (17.3 s) | 0.4000, macroF1 0.6829 | Sanity ablation only; not used for any choice. |
| 2026-11-14 | M2 v1 rules on degraded IDSSE (no tags) | **0.3888** | 0.7878 | – | – | Pooled over the 7 held-out corruptions; 1148 poss. vs 2017 GT. The reference for v2. |
| 2026-11-14 | M2 v2, HGB emissions + v1 decoder settings | 0.6334 | 0.8955 | – | – | LOMO OOF; frame argmax accuracy 0.907. |
| 2026-11-14 | M2 v2, decoder tuned (coordinate descent, pooled OOF) | 0.6522 (per-match mean 0.654, 90% CI 0.621–0.682) | 0.8973 | – | – | Decoder: weight 3.0, team cost 2.0, enter D 10, leave D 5, min poss 12 steps (2.4 s), min D 20 steps (4 s), D bias +0.5. Surface flat: first grid gave 0.6532 with different values. |
| 2026-11-14 | **M2 v2 NESTED** (decoder re-tuned without the held-out match) | **0.6427** | 0.8954 | 226 (15.8 s), 129 D stretches, D share 0.44 | 0.5143 (11 poss., 25.1 s), macroF1 0.7683 | Honest CV estimate. Shipped: model trained on all 7 matches (sha-identical on retrain). sha256 f258bdea…41e6. Public run once. |
| 2026-11-14 | M2 v2 without Smart Tagging (ablation) | – | – | 225 (15.8 s) | 0.4615, macroF1 0.7634 | Sanity ablation only. |
