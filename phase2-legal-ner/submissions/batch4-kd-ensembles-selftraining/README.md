# Phase 2 · submissions · `batch4-kd-ensembles-selftraining`

Codabench zips written by the notebooks in [`../../notebooks/batch4-kd-ensembles-selftraining/`](../../notebooks/batch4-kd-ensembles-selftraining/) (17 files).

**Question.** Can distillation, ensembling and self-training use the unlabelled text? **Finding.** Distillation helped only when the student also learned the teacher's tags on unlabelled text (LSTM 0.5458, which beat its teacher). A span-voting ensemble was best (0.5843), and self-training one model on the ensemble's spans let a single model match it (`p2-17b`, **0.5852, final**). Ensembles 16a, 16b, 16c and 16g were built by the same notebook family; only their zips are kept.

| File | What it is | Score |
|---|---|---|
| `p2_15a_kd_moe13_to_lstm_labelled_only.zip` |  | – |
| `p2_15b_kd_moe13_to_lstm_plus_unlabelled.zip` | KD LSTM student with unlabelled text | 0.5458 |
| `p2_15c_kd_moe13_to_gqa_plus_unlabelled.zip` |  | – |
| `p2_16a_ensemble_moe_only_A.zip` |  | – |
| `p2_16a_ensemble_moe_only_B.zip` |  | – |
| `p2_16b_ensemble_cross_arch_A.zip` |  | – |
| `p2_16b_ensemble_cross_arch_B.zip` |  | – |
| `p2_16c_ensemble_moe_plus_student_A.zip` |  | – |
| `p2_16c_ensemble_moe_plus_student_B.zip` |  | – |
| `p2_16e_ensemble_final_moe3_plus_students_A.zip` |  | – |
| `p2_16e_ensemble_final_moe3_plus_students_B.zip` |  | – |
| `p2_16f_ensemble_final_with_lora_full_selftrain_A.zip` |  | – |
| `p2_16f_ensemble_final_with_lora_full_selftrain_B.zip` |  | – |
| `p2_16g_ensemble_final_with_two_selftrain_A.zip` | final ensemble | – |
| `p2_16g_ensemble_final_with_two_selftrain_B.zip` |  | – |
| `p2_17_selftrain_moe_from_ensemble.zip` | first self-trained model | 0.5793 |
| `p2_17b_selftrain_header256_seed1.zip` | **final single model**, self-trained | 0.5852 |

A dash means the score is not listed in the phase README (it is on Codabench).

Naming: `<notebook>_<setup>[_variant].zip`, so each file traces back to the notebook and setting that produced it.
