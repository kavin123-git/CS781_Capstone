# Phase 2 · submissions · `batch1-moe-techniques`

Codabench zips written by the notebooks in [`../../notebooks/batch1-moe-techniques/`](../../notebooks/batch1-moe-techniques/) (11 files).

**Question.** With the output side fixed (Viterbi + edge trimming), which adaptation technique works best on the MoE? **Finding.** They finished close together, so the technique was not the bottleneck. Prefix tuning failed (0.12), removing the causal mask hurt, and splitting punctuation fixed span edges but raised the Phase-1 loss. The baseline `p2-01` scored 0.4343 on Codabench.

| File | What it is | Score |
|---|---|---|
| `p2_01_moe_hc_full_ft.zip` | baseline | 0.4343 |
| `p2_02_moe_hc_lorma.zip` |  | – |
| `p2_03_moe_hc_lora.zip` |  | – |
| `p2_04_moe_hc_full_ft_bidir.zip` |  | – |
| `p2_05_moe_hc_full_ft_punct.zip` |  | – |
| `p2_06_moe_hc_full_ft_clsw.zip` |  | – |
| `p2_07_moe_hc_lorma_clsw.zip` |  | – |
| `p2_08_moe_hc_lora_clsw.zip` |  | – |
| `p2_09_moe_hc_adapter_clsw.zip` |  | – |
| `p2_10_moe_hc_bitfit_clsw.zip` |  | – |
| `p2_11_moe_hc_prefix_clsw.zip` | prefix tuning (failed) | 0.1220 |

A dash means the score is not listed in the phase README (it is on Codabench).

Naming: `<notebook>_<setup>[_variant].zip`, so each file traces back to the notebook and setting that produced it.
