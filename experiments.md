# Experiments log

One line per run. Model selection uses **IDSSE leave-one-match-out CV only**. The public BAR-ATH
window is a bug check, never a selection criterion (RULES.md, EVALUATION.md).

| date | version | IDSSE-CV spellF1 | IDSSE-CV macroF1 | BAR-ATH pred. possessions (mean s) | public sanity spellF1 | note |
|---|---|---|---|---|---|---|
| 2026-11-14 | M0 baseline50 (organisers' rule, our pipeline) | – | – | 669 (7.6 s) | 0.3256 (26 poss., 10.8 s; GT 17, 14.2 s) | Byte-identical to organisers' baseline_*.csv (all four windows). Pipeline check only. |
