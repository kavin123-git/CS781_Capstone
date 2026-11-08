# Phase 5 · submissions · `batch2-ablations-and-promotion`

Codabench zips written by the notebooks in [`../../notebooks/batch2-ablations-and-promotion/`](../../notebooks/batch2-ablations-and-promotion/) (11 files).

**Question.** Which techniques, losses, chunk budgets and heads are best on a 6,000-case subset, and do the winners survive on full data (16 epochs)? **Finding.** Two subset winners failed on full data (LoRMA and rare-statute oversampling): partial fine-tuning won on full data. Every promotion was therefore re-checked.

| File | What it is | Score |
|---|---|---|
| `p5_gdn2_gdn_full_ft_per_label.zip` |  | – |
| `p5_gdn2_gdn_partial4_per_label.zip` |  | – |
| `p5_gqa_long_gqa_lora_long_shrunk.zip` |  | – |
| `p5_gqa_long_gqa_lorma_long_shrunk.zip` |  | – |
| `p5_gqa_long_gqa_partial4_long_shrunk.zip` |  | – |
| `p5_kd_long_kd_gqa_lorma_long_shrunk.zip` | GQA LoRMA + KD (no KD: 0.3481) | 0.3595 |
| `p5_kd_long_kd_gqa_partial4_long_shrunk.zip` | GQA partial-4 + KD (no KD: 0.3653) | 0.3720 |
| `p5_moe_lorma_moe_lorma_long_shrunk.zip` |  | – |
| `p5_mor2_mor_full_ft_per_label.zip` |  | – |
| `p5_mor2_mor_lorma_per_label.zip` |  | – |
| `submission_p5_moe_long_moe_partial2_long_shrunk.zip` |  | – |

A dash means the score is not listed in the phase README (it is on Codabench).

Naming: `<notebook>_<setup>[_variant].zip`, so each file traces back to the notebook and setting that produced it.
