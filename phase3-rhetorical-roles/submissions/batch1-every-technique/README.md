# Phase 3 · submissions · `batch1-every-technique`

Codabench zips written by the notebooks in [`../../notebooks/batch1-every-technique/`](../../notebooks/batch1-every-technique/) (15 files).

**Question.** Which technique is best, on which of the five Phase-1 backbones, and should CL and IT have separate models? **Finding.** The MoE was best and the techniques tied. Separate CL/IT models lost on CL, so we kept **one pooled model**.

| File | What it is | Score |
|---|---|---|
| `p3s_gqa160_full_ft.zip` |  | – |
| `p3s_gqa160_lora_pooled.zip` |  | – |
| `p3s_gqa160_lora_separate.zip` |  | – |
| `p3s_gqa160_rslora.zip` |  | – |
| `p3s_lstm_frozen_head_only.zip` |  | – |
| `p3s_lstm_full_ft.zip` |  | – |
| `p3s_moe_full_ft.zip` |  | – |
| `p3s_moe_lora.zip` |  | – |
| `p3s_moe_lorma.zip` | batch 1, MoE LoRMA | 0.4879 |
| `p3s_mor_full_ft.zip` |  | – |
| `p3s_mor_lora.zip` |  | – |
| `p3s_mor_lorma.zip` |  | – |
| `submission_p3s_gdn_full_ft.zip` |  | – |
| `submission_p3s_gdn_lora.zip` |  | – |
| `submission_p3s_gdn_lorma.zip` |  | – |

A dash means the score is not listed in the phase README (it is on Codabench).

Naming: `<notebook>_<setup>[_variant].zip`, so each file traces back to the notebook and setting that produced it.
