# Phase 4 · submissions · `batch4-seed-and-partial4-student`

Codabench zips written by the notebooks in [`../../notebooks/batch4-seed-and-partial4-student/`](../../notebooks/batch4-seed-and-partial4-student/) (2 files).

**Question.** How big is seed noise on the best KD run, and can KD into GQA partial top-4 add ensemble members? **Finding.** The second seed of the GQA LoRMA student scored 0.7451, the best single model; the partial top-4 student was the best single model on the selection set.

| File | What it is | Score |
|---|---|---|
| `p4_kd_gqa3_kd3_gqa_lorma_s1_tuned.zip` | GQA LoRMA KD student, second seed (best single model) | 0.7451 |
| `p4_kd_gqa3_kd3_gqa_partial4_tuned.zip` |  | – |

A dash means the score is not listed in the phase README (it is on Codabench).

Naming: `<notebook>_<setup>[_variant].zip`, so each file traces back to the notebook and setting that produced it.
