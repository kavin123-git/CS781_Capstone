# Phase 4 · submissions · `batch1-every-backbone`

Codabench zips written by the notebooks in [`../../notebooks/batch1-every-backbone/`](../../notebooks/batch1-every-backbone/) (10 files).

**Question.** How does one common recipe (LoRMA) do on every Phase-1 model? **Finding.** Every learning curve was still rising, and at a 0.5 threshold the models under-predicted label 1, so choosing epochs at 0.5 had kept the worst epoch. From batch 2 on, epochs and thresholds are chosen with the threshold tuned on the selection set. MoR stayed at chance (0.3910).

| File | What it is | Score |
|---|---|---|
| `p4_gdn_gdn_lorma_thr05.zip` |  | – |
| `p4_gdn_gdn_lorma_tuned.zip` |  | – |
| `p4_gqa_gqa_lorma_thr05.zip` |  | – |
| `p4_gqa_gqa_lorma_tuned.zip` |  | – |
| `p4_lstm_lstm_full_ft_thr05.zip` |  | – |
| `p4_lstm_lstm_full_ft_tuned.zip` |  | – |
| `p4_moe_moe_lorma_thr05.zip` | batch-1 MoE LoRMA | 0.6301 |
| `p4_moe_moe_lorma_tuned.zip` |  | – |
| `p4_mor_mor_lorma_thr05.zip` |  | – |
| `p4_mor_mor_lorma_tuned.zip` |  | – |

A dash means the score is not listed in the phase README (it is on Codabench).

Naming: `<notebook>_<setup>[_variant].zip`, so each file traces back to the notebook and setting that produced it.
