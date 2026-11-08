# Phase 7 · submissions · `batch1-every-backbone`

Codabench zips written by the notebooks in [`../../notebooks/batch1-every-backbone/`](../../notebooks/batch1-every-backbone/) (24 files).

**Question.** How does a bi-encoder on each backbone compare with TF-IDF, and which design choices (pooling, objective, negatives, chunk choice) matter? **Finding.** Every dense model stayed below TF-IDF and the adaptation techniques tied. The two ablation notebooks gave the leads: chunks spread evenly over the case, ListNet, no projection and late interaction. In-batch negatives were essential. The GDN encoder never formed a similarity space (0.0248).

| File | What it is | Score |
|---|---|---|
| `p7_gdn_gdn_lorma_dense.zip` | GDN (no similarity space) | 0.0248 |
| `p7_gdn_gdn_lorma_hybrid.zip` |  | – |
| `p7_gdn_gdn_partial4_dense.zip` |  | – |
| `p7_gdn_gdn_partial4_hybrid.zip` |  | – |
| `p7_gqa_gqa_full_ft_dense.zip` |  | – |
| `p7_gqa_gqa_full_ft_hybrid.zip` |  | – |
| `p7_gqa_gqa_ia3_dense.zip` |  | – |
| `p7_gqa_gqa_ia3_hybrid.zip` |  | – |
| `p7_gqa_gqa_lorma_dense.zip` |  | – |
| `p7_gqa_gqa_lorma_hybrid.zip` |  | – |
| `p7_gqa_gqa_partial4_dense.zip` |  | – |
| `p7_gqa_gqa_partial4_hybrid.zip` |  | – |
| `p7_gqa_gqa_qlora_dense.zip` |  | – |
| `p7_gqa_gqa_qlora_hybrid.zip` |  | – |
| `p7_lstm_lstm_full_ft_dense.zip` |  | – |
| `p7_lstm_lstm_full_ft_hybrid.zip` |  | – |
| `p7_lstm_lstm_partial1_dense.zip` |  | – |
| `p7_lstm_lstm_partial1_hybrid.zip` |  | – |
| `p7_moe_moe_lorma_dense.zip` | batch-1 MoE LoRMA | 0.3012 |
| `p7_moe_moe_lorma_hybrid.zip` |  | – |
| `p7_moe_moe_lorma_strip_dense.zip` |  | – |
| `p7_moe_moe_lorma_strip_hybrid.zip` |  | – |
| `p7_moe_moe_partial2_dense.zip` |  | – |
| `p7_moe_moe_partial2_hybrid.zip` |  | – |

A dash means the score is not listed in the phase README (it is on Codabench).

Naming: `<notebook>_<setup>[_variant].zip`, so each file traces back to the notebook and setting that produced it.
