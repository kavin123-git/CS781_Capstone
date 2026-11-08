# Phase 4 · submissions · `batch3-partial-ft-and-strong-teacher`

Codabench zips written by the notebooks in [`../../notebooks/batch3-partial-ft-and-strong-teacher/`](../../notebooks/batch3-partial-ft-and-strong-teacher/) (8 files).

**Question.** Does partial fine-tuning carry over to the GQA, and does KD work from the strong MoE teacher? **Finding.** Every student improved, and the 111M GQA LoRMA student (0.7370) caught up with its 219M teacher.

| File | What it is | Score |
|---|---|---|
| `p4_gqa_partial_gqa_full_ft_tuned.zip` |  | – |
| `p4_gqa_partial_gqa_partial2_tuned.zip` |  | – |
| `p4_gqa_partial_gqa_partial4_tuned.zip` |  | – |
| `p4_kd_gqa2_kd2_gqa_lorma_tuned.zip` | GQA LoRMA KD student (strong teacher) | 0.7370 |
| `p4_kd_gqa2_kd2_gqa_partial2_tuned.zip` |  | – |
| `p4_moe_partial2_m_partial2_ht4_tuned.zip` |  | – |
| `p4_moe_partial2_m_partial4_ht4_tuned.zip` |  | – |
| `p4_moe_partial2_m_partial4_tuned.zip` |  | – |

A dash means the score is not listed in the phase README (it is on Codabench).

Naming: `<notebook>_<setup>[_variant].zip`, so each file traces back to the notebook and setting that produced it.
