# Phase 3 · submissions · `batch5-final-combinations`

Codabench zips written by the notebooks in [`../../notebooks/batch5-final-combinations/`](../../notebooks/batch5-final-combinations/) (22 files).

**Question.** Do the winners combine, and does any remaining idea help? **Finding.** Only longer units helped (96 sentences, the final model). Three seeds of one setup spread by 0.04 on test, larger than most differences. The random-initialisation control (0.26 vs 0.46 in CV) shows the Phase-1 weights matter.

| File | What it is | Score |
|---|---|---|
| `p3s_ensemble_ens.zip` | greedy CV ensemble of 3 (best by CV) | 0.5184 |
| `p3s_kd_moe_kd_pos_dom_lora.zip` |  | – |
| `p3s_kd_moe_kd_pos_ia3.zip` |  | – |
| `p3s_kd_moe_kd_pos_lora.zip` |  | – |
| `p3s_kd_moe_kd_pos_lora_a7.zip` |  | – |
| `p3s_moe_combo1_pos_cwinv_lora.zip` |  | – |
| `p3s_moe_combo1_pos_dom_lora.zip` |  | – |
| `p3s_moe_combo1_pos_lora_avg.zip` |  | – |
| `p3s_moe_combo1_pos_lora_ls.zip` |  | – |
| `p3s_moe_combo1_pos_lora_seed1.zip` |  | – |
| `p3s_moe_combo1_pos_lora_seed2.zip` |  | – |
| `p3s_moe_combo2_pos_adalora.zip` |  | – |
| `p3s_moe_combo2_pos_dom_ia3.zip` |  | – |
| `p3s_moe_combo2_pos_frozen.zip` | MoE **frozen**, only the head trained | 0.5376 |
| `p3s_moe_combo2_pos_ia3.zip` |  | – |
| `p3s_moe_combo2_pos_qlora.zip` | MoE, QLoRA | 0.5274 |
| `p3s_moe_combo2_pos_rslora.zip` |  | – |
| `p3s_moe_long_pos_ia3_s96.zip` | **final**: MoE, IA³, 96-sentence units + position | 0.5418 |
| `p3s_moe_long_pos_lora_s128.zip` |  | – |
| `p3s_moe_long_pos_lora_s96.zip` |  | – |
| `p3s_moe_long_rand_frozen_s48.zip` | **random-init control** (no Phase-1 weights) | 0.2584 |
| `p3s_moe_long_rand_full_ft_s48.zip` |  | – |

A dash means the score is not listed in the phase README (it is on Codabench).

Naming: `<notebook>_<setup>[_variant].zip`, so each file traces back to the notebook and setting that produced it.
