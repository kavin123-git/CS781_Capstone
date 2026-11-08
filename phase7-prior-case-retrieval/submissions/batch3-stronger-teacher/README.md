# Phase 7 · submissions · `batch3-stronger-teacher`

Codabench zips written by the notebooks in [`../../notebooks/batch3-stronger-teacher/`](../../notebooks/batch3-stronger-teacher/) (22 files).

**Question.** Does a stronger, recipe-matched teacher and reading 16 chunks help? **Finding.** The distilled 111M GQA matched its 219M teacher on dev. The MoE with late interaction and 16 chunks was the best single dense model (0.3314 test). Dense models still trailed TF-IDF.

| File | What it is | Score |
|---|---|---|
| `p7_gqa3_gqa_uni_maxsim_dense.zip` |  | – |
| `p7_gqa3_gqa_uni_maxsim_hybrid.zip` |  | – |
| `p7_gqa3_kd3_gqa_uni_list_proj0_dense.zip` | GQA LoRMA + KD (no KD: 0.3001) | 0.3136 |
| `p7_gqa3_kd3_gqa_uni_list_proj0_hybrid.zip` |  | – |
| `p7_gqa3_kd3_gqa_uni_maxsim_dense.zip` |  | – |
| `p7_gqa3_kd3_gqa_uni_maxsim_hybrid.zip` | best single hybrid (GQA KD student + TF-IDF) | 0.4152 |
| `p7_lstm3_kd3_lstm_uni_list_proj0_dense.zip` | best LSTM (full fine-tuning + KD) | 0.2115 |
| `p7_lstm3_kd3_lstm_uni_list_proj0_hybrid.zip` |  | – |
| `p7_lstm3_lstm_uni_list_proj0_dense.zip` |  | – |
| `p7_lstm3_lstm_uni_list_proj0_hybrid.zip` |  | – |
| `p7_moe3_moe_uni_maxsim_16_dense.zip` | best single dense model (MoE, late interaction, 16 chunks) | 0.3314 |
| `p7_moe3_moe_uni_maxsim_16_hybrid.zip` |  | – |
| `p7_moe3_moe_uni_maxsim_list_dense.zip` |  | – |
| `p7_moe3_moe_uni_maxsim_list_hybrid.zip` |  | – |
| `p7_moe3_moe_uni_maxsim_t_dense.zip` |  | – |
| `p7_moe3_moe_uni_maxsim_t_hybrid.zip` |  | – |
| `p7_mor3_kd3_mor_ft_uni_list_proj0_dense.zip` |  | – |
| `p7_mor3_kd3_mor_ft_uni_list_proj0_hybrid.zip` |  | – |
| `p7_mor3_mor_ft_uni_list_proj0_dense.zip` |  | – |
| `p7_mor3_mor_ft_uni_list_proj0_hybrid.zip` |  | – |
| `p7_mor3_mor_uni_list_proj0_dense.zip` | best MoR (LoRMA) | 0.2144 |
| `p7_mor3_mor_uni_list_proj0_hybrid.zip` |  | – |

A dash means the score is not listed in the phase README (it is on Codabench).

Naming: `<notebook>_<setup>[_variant].zip`, so each file traces back to the notebook and setting that produced it.
