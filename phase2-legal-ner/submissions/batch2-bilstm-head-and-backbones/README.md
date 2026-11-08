# Phase 2 · submissions · `batch2-bilstm-head-and-backbones`

Codabench zips written by the notebooks in [`../../notebooks/batch2-bilstm-head-and-backbones/`](../../notebooks/batch2-bilstm-head-and-backbones/) (19 files).

**Question.** Our decoders read only to the left, but a party's role is usually written to its right ("…, counsel for the respondent"). Does a BiLSTM head fix this, on every backbone? **Finding.** Yes, it helped every technique and every backbone. Adapters + BiLSTM (`p2-13`, 0.5335) became the teacher for batch 4. The `sweep-p2s-*` notebooks run the same sweep on the GQA, LSTM, MoR and GDN.

| File | What it is | Score |
|---|---|---|
| `p2_12_moe_hc_full_ft_bilstm_clsw.zip` |  | – |
| `p2_13_moe_hc_adapter_bilstm_clsw.zip` | adapters + BiLSTM head (teacher) | 0.5335 |
| `p2_14_moe_hc_lorma_bilstm_clsw.zip` |  | – |
| `p2s_gdn_adapter.zip` |  | – |
| `p2s_gdn_adapter_bilstm.zip` |  | – |
| `p2s_gdn_full_ft_bilstm.zip` |  | – |
| `p2s_gdn_lorma_bilstm.zip` |  | – |
| `p2s_gqa160_adapter_bilstm.zip` | GQA, adapters + BiLSTM | 0.5200 |
| `p2s_gqa160_adapter_pfeiffer_bilstm.zip` |  | – |
| `p2s_gqa160_lorma_bilstm.zip` |  | – |
| `p2s_gqa160_lorma_pi_bilstm.zip` |  | – |
| `p2s_lstm84_adapter_bilstm.zip` |  | – |
| `p2s_lstm84_adapter_pfeiffer_bilstm.zip` |  | – |
| `p2s_lstm84_frozen_bilstm.zip` |  | – |
| `p2s_lstm84_partial_ft_top1_bilstm.zip` |  | – |
| `p2s_mor2_bitfit_bilstm.zip` |  | – |
| `p2s_mor2_lora_bilstm.zip` |  | – |
| `p2s_mor2_lorma_bilstm.zip` |  | – |
| `p2s_mor2_lorma_pi_bilstm.zip` |  | – |

A dash means the score is not listed in the phase README (it is on Codabench).

Naming: `<notebook>_<setup>[_variant].zip`, so each file traces back to the notebook and setting that produced it.
