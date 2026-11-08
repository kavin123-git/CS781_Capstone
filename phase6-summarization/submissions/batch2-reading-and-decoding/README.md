# Phase 6 · submissions · `batch2-reading-and-decoding`

Codabench zips written by the notebooks in [`../../notebooks/batch2-reading-and-decoding/`](../../notebooks/batch2-reading-and-decoding/) (9 files).

**Question.** Since training did not beat the extractive baseline, does changing how the model reads and decodes help? **Finding.** N-gram blocking (3-gram in particular) was the largest single gain for every model, and chunked reading helped; hierarchical summaries hurt. Word-level KD gave the best student, a half-size GQA with Houlsby adapters (0.8374).

| File | What it is | Score |
|---|---|---|
| `p6_decode_gqa_ft_norep3_overlap3.zip` |  | – |
| `p6_decode_gqa_hb_norep3_chunks3.zip` |  | – |
| `p6_decode_gqa_kdlhb_norep3_chunks3.zip` | GQA Houlsby + word-level KD (best student, half the size) | 0.8374 |
| `p6_decode_moe_chunks3.zip` |  | – |
| `p6_decode_moe_norep3.zip` |  | – |
| `p6_gdn2_gdn_full_ft.zip` |  | – |
| `p6_gdn2_gdn_houlsby.zip` |  | – |
| `p6_kd_logit_gqa_kdl_gqa_houlsby.zip` |  | – |
| `p6_kd_logit_gqa_kdl_gqa_lorma.zip` |  | – |

A dash means the score is not listed in the phase README (it is on Codabench).

Naming: `<notebook>_<setup>[_variant].zip`, so each file traces back to the notebook and setting that produced it.
