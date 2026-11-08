# Phase 3 · submissions · `batch3-48-sentence-units-and-ablations`

Codabench zips written by the notebooks in [`../../notebooks/batch3-48-sentence-units-and-ablations/`](../../notebooks/batch3-48-sentence-units-and-ablations/) (14 files).

**Question.** With 48-sentence units, which backbones and techniques remain, and which parts of the MoE recipe matter? **Finding.** Context and the BiLSTM head carry the model; class weights hardly matter.

| File | What it is | Score |
|---|---|---|
| `p3s_gqa160_s48_full_ft_s48.zip` | best GQA | 0.4732 |
| `p3s_gqa160_s48_lora_s48.zip` |  | – |
| `p3s_gqa160_s48_lorma_s48.zip` |  | – |
| `p3s_gqa160_s48_qlora_s48.zip` |  | – |
| `p3s_gqa160_s48_rslora_s48.zip` |  | – |
| `p3s_lstm_s48_full_ft_s48.zip` | best LSTM | 0.4181 |
| `p3s_moe_ablate_abl_cw_inv.zip` |  | – |
| `p3s_moe_ablate_abl_cw_none.zip` |  | – |
| `p3s_moe_ablate_abl_head_linear.zip` |  | – |
| `p3s_moe_ablate_abl_sentence_only.zip` | ablation: sentences alone, linear head | 0.4064 |
| `p3s_moe_s48_full_ft_s48.zip` |  | – |
| `p3s_moe_s48_ia3_s48.zip` |  | – |
| `p3s_moe_s48_lorma_s48.zip` |  | – |
| `p3s_moe_s48_rslora_s48.zip` |  | – |

A dash means the score is not listed in the phase README (it is on Codabench).

Naming: `<notebook>_<setup>[_variant].zip`, so each file traces back to the notebook and setting that produced it.
