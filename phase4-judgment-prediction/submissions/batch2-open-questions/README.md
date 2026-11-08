# Phase 4 · submissions · `batch2-open-questions`

Codabench zips written by the notebooks in [`../../notebooks/batch2-open-questions/`](../../notebooks/batch2-open-questions/) (20 files).

**Questions (one notebook each).** Does longer training help? Which technique is best? Which chunks matter? Why did MoR and GDN not learn? Does KD from the batch-1 teacher help? **Finding.** Partial fine-tuning of the top 2 MoE layers won (0.7330); the end of the judgment carries most of the signal; MoR and GDN stayed far behind; KD from the weak batch-1 teacher gave nothing.

| File | What it is | Score |
|---|---|---|
| `p4_gdn_lr_gdn_full_ft_tuned.zip` |  | – |
| `p4_gdn_lr_gdn_lorma_lr1e3_tuned.zip` |  | – |
| `p4_gdn_lr_gdn_lorma_lr3e3_tuned.zip` |  | – |
| `p4_kd_gqa_gqa_lorma_nokd_thr05.zip` |  | – |
| `p4_kd_gqa_gqa_lorma_nokd_tuned.zip` |  | – |
| `p4_kd_gqa_kd_gqa_lorma_thr05.zip` |  | – |
| `p4_kd_gqa_kd_gqa_lorma_tuned.zip` | GQA LoRMA KD student (weak batch-1 teacher) | 0.6849 |
| `p4_moe_ablate_a_headtail4_tuned.zip` |  | – |
| `p4_moe_ablate_a_last1_tuned.zip` |  | – |
| `p4_moe_ablate_a_last4_headattn_tuned.zip` |  | – |
| `p4_moe_ablate_a_last4_overlap_tuned.zip` |  | – |
| `p4_moe_full_moe_lorma_v2_thr05.zip` |  | – |
| `p4_moe_full_moe_lorma_v2_tuned.zip` |  | – |
| `p4_moe_tech_t_full_ft_tuned.zip` |  | – |
| `p4_moe_tech_t_partial__full_thr05.zip` |  | – |
| `p4_moe_tech_t_partial__full_tuned.zip` | MoE partial top-2, full data (the teacher) | 0.7330 |
| `p4_moe_tech_t_partial_tuned.zip` |  | – |
| `p4_mor_lr_mor_full_ft_tuned.zip` |  | – |
| `p4_mor_lr_mor_lorma_lr1e3_tuned.zip` |  | – |
| `p4_mor_lr_mor_partial_tuned.zip` |  | – |

A dash means the score is not listed in the phase README (it is on Codabench).

Naming: `<notebook>_<setup>[_variant].zip`, so each file traces back to the notebook and setting that produced it.
