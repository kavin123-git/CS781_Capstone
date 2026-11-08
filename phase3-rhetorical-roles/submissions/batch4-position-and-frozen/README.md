# Phase 3 · submissions · `batch4-position-and-frozen`

Codabench zips written by the notebooks in [`../../notebooks/batch4-position-and-frozen/`](../../notebooks/batch4-position-and-frozen/) (8 files).

**Question.** A unit sees only part of a judgment, so does adding each sentence's **position** in the whole document help, and how useful are the Phase-1 representations alone? **Finding.** Position helps, and a frozen MoE with only the head trained comes close to the tuned one.

| File | What it is | Score |
|---|---|---|
| `p3s_gdn_frozen_frozen_head_only.zip` |  | – |
| `p3s_gqa160_repr_pool_last_s48.zip` |  | – |
| `p3s_gqa160_repr_pool_mean_s48.zip` |  | – |
| `p3s_gqa160_repr_pos_s48.zip` |  | – |
| `p3s_moe_frozen_frozen_s48.zip` |  | – |
| `p3s_moe_pos_pos_lora_s48.zip` | MoE, LoRA, 48 sentences + position | 0.5397 |
| `p3s_moe_pos_pos_lorma_s48.zip` |  | – |
| `p3s_mor_frozen_frozen_head_only.zip` |  | – |

A dash means the score is not listed in the phase README (it is on Codabench).

Naming: `<notebook>_<setup>[_variant].zip`, so each file traces back to the notebook and setting that produced it.
