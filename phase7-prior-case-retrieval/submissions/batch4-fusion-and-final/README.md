# Phase 7 · submissions · `batch4-fusion-and-final`

Codabench zips written by the notebooks in [`../../notebooks/batch4-fusion-and-final/`](../../notebooks/batch4-fusion-and-final/) (11 files).

**Question.** Can fusing TF-IDF, BM25 and all dense setups beat any single system? **Finding.** Learned logistic weights beat every fixed-weight hybrid; rank fusion and boosted trees did worse. A 5×2-fold cross-validation over dev queries matched test within 0.002. The leaderboard entry `p7_learned` (0.4448) also uses the Supreme Court header citation link; **without** that feature the result is 0.4294 (`p7_learned_noheader`).

| File | What it is | Score |
|---|---|---|
| `p7_dense_ensemble.zip` | **best Phase-1-only system**: MoE + KD-GQA + MoE | 0.3479 |
| `p7_dense_top3.zip` |  | – |
| `p7_dense_top5.zip` |  | – |
| `p7_fusion.zip` |  | – |
| `p7_fusion_greedy.zip` |  | – |
| `p7_fusion_top3.zip` |  | – |
| `p7_fusion_top5.zip` |  | – |
| `p7_learned.zip` | **leaderboard entry**: learned combination incl. header-citation link | 0.4448 |
| `p7_learned_noheader.zip` | the same **without** the header feature (leak-free) | 0.4294 |
| `p7_learned_noheader_gbt.zip` |  | – |
| `p7_rrf_best.zip` |  | – |

A dash means the score is not listed in the phase README (it is on Codabench).

Naming: `<notebook>_<setup>[_variant].zip`, so each file traces back to the notebook and setting that produced it.
