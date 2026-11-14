# Experiments log

One line per run. Model selection uses **IDSSE leave-one-match-out CV only**. The public BAR-ATH
window is a bug check, never a selection criterion (RULES.md, EVALUATION.md).

| date | version | IDSSE-CV spellF1 | IDSSE-CV macroF1 | BAR-ATH pred. possessions (mean s) | public sanity spellF1 | note |
|---|---|---|---|---|---|---|
| 2026-11-14 | M0 baseline50 (organisers' rule, our pipeline) | – | – | 669 (7.6 s) | 0.3256 (26 poss., 10.8 s; GT 17, 14.2 s) | Byte-identical to organisers' baseline_*.csv (all four windows). Pipeline check only. |
| 2026-11-14 | M1 v1 draft (team switch cost 6.0) | – | – | 172 (22.0 s) | – (not scored) | GT-free check: 34 ≥1 s close-control spells by the other team ignored; the assumed 1.5 nats/frame evidence was really 1.2 (measured, no labels). Cost restated from the definition as 0.75 s of typical close control = 4.5. |
| 2026-11-14 | **M1 v1** (rules + kickoffs + tags + min-duration Viterbi) | not yet (M2) | not yet (M2) | 193 (19.8 s), 101 D stretches, D share 0.40 | 0.4429 (11 poss., 25.2 s; GT 17, 14.2 s), macroF1 0.7182 | First submission. sha256 80a5f235…0f09. Public run once, after the config was fixed. |
| 2026-11-14 | M1 v1 without Smart Tagging (ablation) | – | – | 218 (17.3 s) | 0.4000, macroF1 0.6829 | Sanity ablation only; not used for any choice. |
