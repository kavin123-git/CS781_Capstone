# Phase 2 · submissions · `batch3-moe-one-factor`

Codabench zips written by the notebooks in [`../../notebooks/batch3-moe-one-factor/`](../../notebooks/batch3-moe-one-factor/) (11 files).

**Question.** Which single factor helps the MoE: learning rate, window size, rare-label oversampling, adapter size, a document header, or other techniques on the stacked recipe? **Finding.** The winners gained much less on the test fold than on our own fold (best single model 0.5485), so instead of tuning further we turned to unlabelled text in batch 4.

| File | What it is | Score |
|---|---|---|
| `p2s_moe2_adapter_dim128.zip` |  | – |
| `p2s_moe2_lr_high_long.zip` |  | – |
| `p2s_moe2_oversample_rare.zip` |  | – |
| `p2s_moe2_window1024.zip` |  | – |
| `p2s_moe3_stack_header256.zip` |  | – |
| `p2s_moe3_stack_oversample.zip` |  | – |
| `p2s_moe3_stack_window512.zip` | best batch-3 single model | 0.5485 |
| `p2s_moe5_stack_full_ft.zip` |  | – |
| `p2s_moe5_stack_lora.zip` |  | – |
| `p2s_moe5_stack_pfeiffer.zip` |  | – |
| `p2s_moe5_stack_prompt.zip` |  | – |

A dash means the score is not listed in the phase README (it is on Codabench).

Naming: `<notebook>_<setup>[_variant].zip`, so each file traces back to the notebook and setting that produced it.
