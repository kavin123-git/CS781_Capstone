# Phase 5 · submissions · `batch4-attention-head-and-rare`

Codabench zips written by the notebooks in [`../../notebooks/batch4-attention-head-and-rare/`](../../notebooks/batch4-attention-head-and-rare/) (4 files).

**Question.** Does an attention head over the chunks, or rare-statute oversampling, improve the best model? **Finding.** The attention head gave the best single MoE (0.3812); oversampling helped on the subset but hurt on full data.

| File | What it is | Score |
|---|---|---|
| `p5_kd_attn_kd_gqa_lorma_attn_long_shrunk.zip` |  | – |
| `p5_kd_attn_kd_gqa_partial4_attn_long_shrunk.zip` |  | – |
| `p5_moe_attn_moe_attn_long_shrunk.zip` | best single MoE (attention head, 16 epochs) | 0.3812 |
| `p5_moe_attn_rare_moe_attn_rare_long_shrunk.zip` |  | – |

A dash means the score is not listed in the phase README (it is on Codabench).

Naming: `<notebook>_<setup>[_variant].zip`, so each file traces back to the notebook and setting that produced it.
