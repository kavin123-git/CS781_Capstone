# Phase 5 · submissions · `batch3-distil-into-remaining`

Codabench zips written by the notebooks in [`../../notebooks/batch3-distil-into-remaining/`](../../notebooks/batch3-distil-into-remaining/) (3 files).

**Question.** Does distilling the MoE into the LSTM and MoR help? **Finding.** Yes, KD helped every student, the weakest most (MoR 0.2745 → 0.3053). The LSTM ran 13 epochs to fit one Kaggle session; its 16-epoch first attempt was cut off by the 12-hour limit.

| File | What it is | Score |
|---|---|---|
| `p5_lstm_long_kd_lstm_full_ft_long_shrunk.zip` | LSTM + KD | 0.3094 |
| `p5_mor_long_kd_mor_full_ft_long_shrunk.zip` | MoR + KD (no KD: 0.2745) | 0.3053 |
| `p5_mor_long_mor_full_ft_long_per_label.zip` |  | – |

A dash means the score is not listed in the phase README (it is on Codabench).

Naming: `<notebook>_<setup>[_variant].zip`, so each file traces back to the notebook and setting that produced it.
