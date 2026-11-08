# Phase 3 · submissions · `batch2-kd-context-domains`

Codabench zips written by the notebooks in [`../../notebooks/batch2-kd-context-domains/`](../../notebooks/batch2-kd-context-domains/) (14 files).

**Questions.** Does distillation help, how much does context matter, and are separate domain models better on the MoE? **Finding.** Context is the lever: more sentences per unit helped, a longer token window did not. The KD gain in CV was a leak and vanished on test.

| File | What it is | Score |
|---|---|---|
| `p3s_gqa160_extra_adalora.zip` |  | – |
| `p3s_gqa160_extra_prefix.zip` |  | – |
| `p3s_gqa160_extra_prompt.zip` |  | – |
| `p3s_gqa160_extra_qlora.zip` |  | – |
| `p3s_kd_gqa160_kd_full_ft.zip` | KD student (GQA) | 0.4631 |
| `p3s_kd_gqa160_kd_lora.zip` |  | – |
| `p3s_kd_lstm_kd_full_ft.zip` |  | – |
| `p3s_moe_ctx_ctx_alone.zip` |  | – |
| `p3s_moe_ctx_ctx_seq12.zip` |  | – |
| `p3s_moe_ctx_ctx_seq48.zip` | context study: 48-sentence units | 0.5128 |
| `p3s_moe_ctx_ctx_win2048_seq64.zip` |  | – |
| `p3s_moe_ctx_ia3.zip` |  | – |
| `p3s_moe_typeB_full_ft_separate.zip` |  | – |
| `p3s_moe_typeB_lora_separate.zip` |  | – |

A dash means the score is not listed in the phase README (it is on Codabench).

Naming: `<notebook>_<setup>[_variant].zip`, so each file traces back to the notebook and setting that produced it.
