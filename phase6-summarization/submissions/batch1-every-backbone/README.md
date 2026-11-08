# Phase 6 · submissions · `batch1-every-backbone`

Codabench zips written by the notebooks in [`../../notebooks/batch1-every-backbone/`](../../notebooks/batch1-every-backbone/) (13 files).

**Question.** How does each backbone do with each technique (LoRMA, LoRA, Houlsby, full fine-tuning), and does sequence-level KD help? **Finding.** The best model (MoE 0.8302) still lost to a trivial baseline, the last 400 words of the judgment. Sequence-level KD made the student copy the teacher's style (0.8195). The LSTM took about 4.5 hours per epoch, so `p6-lstm2` re-ran it on 2,500 documents.

| File | What it is | Score |
|---|---|---|
| `p6_gdn_gdn_lora.zip` |  | – |
| `p6_gdn_gdn_lorma.zip` |  | – |
| `p6_gqa_gqa_full_ft.zip` |  | – |
| `p6_gqa_gqa_houlsby.zip` |  | – |
| `p6_gqa_gqa_lora.zip` |  | – |
| `p6_gqa_gqa_lorma.zip` |  | – |
| `p6_gqa_kd_gqa_lorma.zip` | sequence-level KD (hurt) | 0.8195 |
| `p6_lstm2_kd_lstm_ft_2500.zip` |  | – |
| `p6_lstm2_lstm_ft_2500.zip` |  | – |
| `p6_moe_moe_lora.zip` |  | – |
| `p6_moe_moe_lorma.zip` | batch-1 MoE LoRMA | 0.8302 |
| `p6_mor_mor_lora.zip` |  | – |
| `p6_mor_mor_lorma.zip` | MoR LoRMA | 0.7921 |

A dash means the score is not listed in the phase README (it is on Codabench).

Naming: `<notebook>_<setup>[_variant].zip`, so each file traces back to the notebook and setting that produced it.
