# Phase 6 · submissions · `final-analysis`

Codabench zips written by the notebooks in [`../../notebooks/final-analysis/`](../../notebooks/final-analysis/) (2 files).

Analysis without training: every result with bootstrap confidence intervals, extractive baselines, paired tests, length analysis, the MBR ensemble and the 400-word cut. The cut raises BERTScore only because the scorer truncates at about 400 words (ROUGE-1 drops). **Final: 0.8393.**

| File | What it is | Score |
|---|---|---|
| `p6_ensemble_mbr8.zip` | MBR ensemble over the 8 best systems | 0.8387 |
| `p6_postproc_dec_moe2_norep3_overlap3_first_400_words.zip` | **final**: MoE LoRMA, 3 overlapping windows, 3-gram blocking, cut to 400 words | 0.8393 |

Naming: `<notebook>_<setup>[_variant].zip`, so each file traces back to the notebook and setting that produced it.
