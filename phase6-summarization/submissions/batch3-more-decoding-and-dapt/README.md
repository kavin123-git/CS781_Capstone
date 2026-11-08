# Phase 6 · submissions · `batch3-more-decoding-and-dapt`

Codabench zips written by the notebooks in [`../../notebooks/batch3-more-decoding-and-dapt/`](../../notebooks/batch3-more-decoding-and-dapt/) (15 files).

**Question.** Do overlapping windows, all 9 techniques on one budget, domain-adaptive pretraining, or other optimizers help? **Finding.** Overlapping windows gave the best system (0.8382 before the cut). Domain-adaptive pretraining and the optimizer choice changed little. `p6-ablate-gqa` and `p6-opt-moe` produced tables rather than submissions.

| File | What it is | Score |
|---|---|---|
| `p6_dapt_gqa_dapt_gqa_full_ft.zip` |  | – |
| `p6_dapt_gqa_dapt_gqa_houlsby.zip` |  | – |
| `p6_dapt_gqa_hp_batch4.zip` |  | – |
| `p6_dapt_gqa_hp_batch64.zip` |  | – |
| `p6_dapt_gqa_hp_lr1e-3.zip` |  | – |
| `p6_dapt_gqa_hp_lr1e-4.zip` |  | – |
| `p6_dapt_gqa_hp_rank4.zip` |  | – |
| `p6_dapt_gqa_hp_rank64.zip` |  | – |
| `p6_dapt_gqa_hp_ref.zip` |  | – |
| `p6_decode_gdn_lorma_reppen1.2.zip` |  | – |
| `p6_decode_gdn_lorma_reppen1.2_norep3.zip` | best GDN | 0.8129 |
| `p6_decode_lstm_kd_norep3_sample.zip` |  | – |
| `p6_decode_lstm_kd_reppen1.2_norep3.zip` | best LSTM | 0.8174 |
| `p6_decode_moe2_norep3_chunks3.zip` |  | – |
| `p6_decode_moe2_norep3_overlap3.zip` | MoE, overlapping windows (before the cut) | 0.8382 |

A dash means the score is not listed in the phase README (it is on Codabench).

Naming: `<notebook>_<setup>[_variant].zip`, so each file traces back to the notebook and setting that produced it.
