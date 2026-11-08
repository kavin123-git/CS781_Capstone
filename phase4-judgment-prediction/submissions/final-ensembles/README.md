# Phase 4 · submissions · `final-ensembles`

Codabench zips written by the notebooks in [`../../notebooks/final-ensembles/`](../../notebooks/final-ensembles/) (5 files).

Analysis and the final ensemble run on CPU only. The ensemble shifts each member so its own tuned threshold sits at 0, averages them, and tunes the ensemble threshold. It is checked honestly by choosing on one half of the selection set and scoring the other. The final ensemble of 5 scored **0.7547**.

| File | What it is | Score |
|---|---|---|
| `p4_ensemble_all_good.zip` |  | – |
| `p4_ensemble_best_single.zip` |  | – |
| `p4_ensemble_greedy.zip` |  | – |
| `p4_ensemble_top3.zip` |  | – |
| `p4_ensemble_top5.zip` | **final**: MoE partial + GQA partial-4 (with and without KD) + two GQA LoRMA KD students | 0.7547 |

A dash means the score is not listed in the phase README (it is on Codabench).

Naming: `<notebook>_<setup>[_variant].zip`, so each file traces back to the notebook and setting that produced it.
