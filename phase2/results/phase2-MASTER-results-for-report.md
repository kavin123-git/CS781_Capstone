# Phase 2 (L-NER) — MASTER RESULTS FILE for the final report (everything explored so far)

Last updated: Sun 2026-10-11, ~11:00 IST. Deadline: Sun 2026-10-11 23:59 IST.
This file consolidates, in one place, every experiment run so far and what it showed. Detailed per-setup tables (per-label F1, error breakdowns, boundary kinds, header/length analysis) stay in the companion docs listed in section 10. All numbers below are copied from the notebooks' printed outputs that the student ran and returned; nothing is estimated unless marked.

How to read the scores: **fold2 F1** = strict macro-F1 on fold2 for a model trained on fold1 (validation estimate, 35 documents, so differences below about 0.015 are noise). **Codabench** = strict macro-F1 on the hidden test set (fold3) for the same recipe refit on fold1+fold2. Codabench is consistently 0.045–0.09 above fold2 (more training data, easier test set). Seed-to-seed noise measured directly: 0.003 (two identical MoE runs, 0.4539 vs 0.4564).

---
## 1. Requirement checklist (brief §8.1–8.5)

| Requirement | Status | Evidence |
|---|---|---|
| Own Phase-1 models only, token/span classification head | Done | all runs |
| All four Phase-1 architectures | Done | MoE (decoder-only, main), GDN enc-dec (encoder), LSTM 84M (lightweight), and for the 4th type GQA 160M and MoR 160M (the canon-layers variant has no results recorded in the project docs) — see §3 |
| ≥ 6 substantially different submissions with different Phase-1 models/techniques | Done (27 Codabench submissions so far) | §2 |
| ≥ 5 fine-tuning/PEFT techniques | Done: full FT, partial FT, LoRA, LoRMA (+ and π variants), Houlsby adapters, Pfeiffer adapters, BitFit, prefix tuning, frozen-backbone + head | §4 |
| LoRMA compulsory | Done on MoE, GQA, MoR, GDN (LoRMA+ and LoRMA_π) | §4 |
| ≥ 1 knowledge distillation | Done: KD A (labelled only) and KD B (+ unlabelled text) | §6 |
| Compare F1, exact boundaries, labels, compute, model complexity | Done | per-setup error breakdowns + efficiency table §7 |
| LLM-use declaration (course policy) | TO WRITE in the report | §9 |
| Prompt tuning (on the §8.2.1 menu) | Not run (optional; prefix tuning failed) | §10 |

Submissions used: 27 of the 40 assumed safe (14 MoE + 4 GQA + 2 MoR + 1 GDN + 4 LSTM + 2 KD). The Codabench page shows 80 / 20 per day, the project document says 40 / 10 per day, so ~13 remain under the conservative assumption. Not yet submitted: the 4 zips of the second MoE sweep (`p2s_moe2`), the ensemble zips (16a/16b/16c), and the follow-up sweep (`p2s_moe3`).

---
## 2. Codabench leaderboard of everything submitted (strict macro-F1, test set)

| Rank | Submission | Architecture | Technique / change | Fold2 F1 | **Codabench** |
|---|---|---|---|---|---|
| 1 | p2_15b KD + unlabelled text | LSTM 84M student (teacher MoE) | KD, top LSTM layer + BiLSTM head | 0.4745 | **0.5458** |
| 2 | p2_13 adapter_bilstm_clsw | MoE 218.7M | Houlsby + BiLSTM + class weights | 0.4562 | 0.5335 |
| 3 | p2_14 lorma_bilstm_clsw | MoE | LoRMA+ + BiLSTM | 0.4766 | 0.5274 |
| 4 | p2_12 full_ft_bilstm_clsw | MoE | full FT + BiLSTM | 0.4635 | 0.5217 |
| 5 | GQA adapter_bilstm | GQA 111M | Houlsby + BiLSTM | 0.4655 | 0.5200 |
| 6 | GQA adapter_pfeiffer_bilstm | GQA | Pfeiffer + BiLSTM | 0.4486 | 0.5113 |
| 7 | p2_15a KD labelled only | LSTM student | KD | 0.4245 | 0.5109 |
| 8 | GQA lorma_pi_bilstm | GQA | LoRMA_π + BiLSTM | 0.4515 | 0.5071 |
| 9 | LSTM partial_ft_top1_bilstm | LSTM 84M | top layer FT + BiLSTM | 0.4285 | 0.5008 |
| 9 | LSTM adapter_bilstm | LSTM | Houlsby + BiLSTM | 0.4090 | 0.5008 |
| 11 | GQA lorma_bilstm | GQA | LoRMA+ + BiLSTM | 0.4552 | 0.4999 |
| 12 | LSTM adapter_pfeiffer_bilstm | LSTM | Pfeiffer + BiLSTM | 0.4099 | 0.4921 |
| 13 | LSTM frozen_bilstm | LSTM | frozen backbone, head only | 0.4053 | 0.4909 |
| 14 | p2_09 adapter_clsw | MoE | Houlsby, linear head | 0.4257 | 0.4809 |
| 15 | p2_06 full_ft_clsw | MoE | full FT, class weights | 0.4048 | 0.4758 |
| 16 | p2_08 lora_clsw | MoE | LoRA, linear head | 0.3957 | 0.4419 |
| 17 | p2_07 lorma_clsw | MoE | LoRMA+, linear head | 0.3822 | 0.4394 |
| 18 | p2_01 full_ft | MoE | full FT, no class weights | 0.3690 | 0.4343 |
| 19 | p2_10 bitfit_clsw | MoE | BitFit, linear head | 0.3714 | 0.4246 |
| 20 | MoR lorma_pi_bilstm | MoR 121M | LoRMA_π + BiLSTM | 0.3406 | 0.4189 |
| 21 | p2_05 full_ft_punct | MoE | punctuation split before BPE | 0.3858 | 0.4179 |
| 22 | MoR bitfit_bilstm | MoR | BitFit + BiLSTM | 0.3308 | 0.4079 |
| 23 | p2_04 full_ft_bidir | MoE | causal mask removed | 0.3327 | 0.4033 |
| 24 | p2_03 lora (no clsw) | MoE | LoRA, linear head | 0.3522 | 0.4002 |
| 25 | GDN adapter_bilstm | GDN encoder 144M | Houlsby + BiLSTM | 0.3341 | 0.3946 |
| 26 | p2_02 lorma (no clsw) | MoE | LoRMA+, linear head | 0.3315 | 0.3841 |
| 27 | p2_11 prefix_clsw | MoE | prefix tuning | 0.0916 | 0.1220 |

(Sorted by the Codabench column. Full MoE run details p2_01–p2_14 are in `phase2-lner-plan-and-results.md`.)

Best by group: **single model 0.5458 (KD B)**; best without using test text: **0.5335 (MoE p2_13)**; best non-MoE: GQA 0.5200.

---
## 3. Architecture comparison (best setup per family)

| Family | Params | Phase-1 BERTScore | NER-text LM loss (nats) | Best setup (BiLSTM head) | Two-fold mean F1 | Codabench | Peak GB |
|---|---|---|---|---|---|---|---|
| MoE high-court (decoder-only, 111M active) | 218.7M | 0.8376 | 3.106 | Houlsby / LoRMA+ | fold2 0.456 / 0.477 | 0.5335 / 0.5274 | 2.5 |
| GQA 160M (decoder-only, dense) | 111.1M | 0.8254 | 3.366 | Houlsby | 0.4719 | 0.5200 | 1.4 |
| LSTM 84M (lightweight) | 83.9M | 0.8237 | 4.018 | partial FT top layer | 0.4355 | 0.5008 | 1.0 |
| MoR 160M (recursive decoder) | 120.8M | 0.7954 | 5.217 | LoRMA_π | 0.3471 | 0.4189 | 1.65 |
| GDN enc-dec (encoder used) | 144.3M | 0.8010 | 3.513 | Houlsby | 0.3364 | 0.3946 | 2.1 |

Findings:
- **A1. Codabench order MoE > GQA > LSTM >> MoR > GDN equals the fold order and the Phase-1 BERTScore order** (MoR and GDN swap). Phase-1 representations transfer; weak Phase-1 models transfer worst.
- **A2. LM loss does not predict NER across architecture types**: LSTM (4.02 nats) beats GDN (3.51) and MoR (5.22 nats) still beats GDN encoder on folds. The GDN encoder was only ever trained to feed a decoder; its states are less directly usable for tagging.
- **A3. Size is not the limit**: the 84M LSTM is within 0.033 of the 218M MoE on Codabench (and with KD-B it exceeds it). With 35 training documents the representation, not the parameter count, is the bottleneck. GQA 111M ≈ MoE 218M on folds (0.4655 vs 0.4562).
- **A4. MoR fine-tuning needs frozen routing** (routing thresholds updated inside forward() drifted to NaN under fp16; freezing them gave 0 skipped batches in all 12 setups).
- **A5. GDN and LSTM are slow** (GDN ~2× slower per setup than attention models; LSTM ~2.5× slower per epoch than GQA because of sequential processing).

---
## 4. Technique coverage matrix (fold2 F1 with the BiLSTM head unless marked "lin" = linear head)

| Technique | MoE | GQA | MoR | GDN | LSTM |
|---|---|---|---|---|---|
| Full fine-tuning | lin 0.369 (CB 0.434); +clsw 0.405; +BiLSTM **0.4635** (CB 0.5217) | lin 0.372; BiLSTM 0.417 | lin 0.206; BiLSTM 0.286 | lin 0.228; BiLSTM 0.242 | lin 0.335; BiLSTM 0.400 |
| Partial FT (top block(s)) | – | top-2: 0.444 | top-2: 0.315 | top-2 lin 0.205 | top-1: **0.4285** (CB 0.5008) |
| LoRA | lin 0.352; +clsw 0.396 | 0.4449 | 0.3306 | lin 0.229 | n/a (no attention projections) |
| LoRMA+ | lin 0.332; +clsw 0.382; +BiLSTM **0.4766** (CB 0.5274) | 0.4552 (CB 0.4999) | 0.3372 | 0.3025 | n/a |
| LoRMA_π | – | 0.4515 (CB 0.5071) | **0.3406** (CB 0.4189) | lin 0.244 | n/a |
| Houlsby adapters | lin 0.426 (CB 0.481); BiLSTM 0.4562 (CB **0.5335**) | **0.4655** (CB 0.5200) | 0.3244 | **0.3341** (CB 0.3946) | 0.4090 (CB 0.5008) |
| Pfeiffer adapters | – | 0.4486 (CB 0.5113) | 0.3287 | lin 0.232 | 0.4099 (CB 0.4921) |
| BitFit | lin 0.371 (CB 0.425) | 0.4429 | 0.3308 (CB 0.4079) | lin 0.216 | 0.4043 |
| Prefix tuning | 0.092 (CB 0.122) — failed | – | – | – | – |
| Frozen backbone + head | – | – | 0.3173 | lin 0.161 | 0.4053 (CB 0.4909) |
| Knowledge distillation | teacher p2_13 | – | – | – | student A 0.4245 (CB 0.5109), B 0.4745 (CB **0.5458**) |

Findings:
- **T1. The prediction head matters far more than the adaptation technique.** BiLSTM head over the frozen/adapted features: MoE adapter +0.030, LoRMA +0.094, full FT +0.059; GQA adapter +0.058, LoRMA +0.104, full FT +0.046; MoR adapter +0.115, LoRMA +0.108, full FT +0.080; GDN adapter +0.079, LoRMA +0.068; LSTM adapter +0.067, full FT +0.065 (fold2). It also removes over-prediction (MoR LoRMA spurious spans 3407 → 785; test entity counts fall from 4.8–5.9k to 4.1–4.3k). The gain exists on a bidirectional encoder (GDN) too, so it is not only "adding right context"; the head also gives sequential smoothing and a task-specific layer.
- **T2. Once the BiLSTM head is in, techniques are within noise of each other** (MoE: full FT / adapter / LoRMA 0.456–0.477, Codabench 0.522–0.534; GQA: 0.443–0.466; MoR: 0.324–0.341; LSTM: 0.405–0.429). Houlsby adapters win on MoE, GQA, GDN; LoRMA_π on MoR; partial FT on LSTM.
- **T3. PEFT ≥ full FT, at about half the memory.** Full FT updates 59–86% of weights on 35 documents, memorises (train loss ~0.001, best epoch = last of 10) and is below every BiLSTM-PEFT setup on GQA and MoR, and below the adapter/LoRMA setups on GDN and LSTM. On MoE with the BiLSTM head it ties (0.464 vs 0.456–0.477) at 5.4 GB vs 2.5 GB.
- **T4. LoRMA vs LoRA**: LoRMA+ beat LoRA by +0.010 fold2 / +0.014 two-fold mean (GQA) and LoRMA+/LoRMA_π beat LoRA by +0.006 / +0.015 two-fold mean (MoR); LoRMA+ with the BiLSTM head gave the best MoE fold2 of the first 14 runs (0.4766; LoRA+BiLSTM was not run on MoE). With a linear head LoRA was slightly ahead on MoE (0.352 vs 0.332, noise). LoRMA trains 22% fewer parameters than LoRA at the same rank (511K vs 658K on MoE).
- **T5. Pfeiffer (after FFN only, ~21% fewer params) ≈ Houlsby** (−0.012 mean on GQA; ≈ on MoR/LSTM).
- **T6. Prefix tuning failed** (0.092 / CB 0.122): 81K parameters, no MLP re-parameterisation, train loss stuck at ~0.47, predicts entities everywhere (18,355 spurious). Negative result.
- **T7. A frozen backbone with only the BiLSTM head is only 0.014–0.024 below the best setup** on LSTM and MoR → the backbone barely adapts; the head does the work.
- **T8. PEFT needs ~2× more epochs than full FT** (best epochs 16–20 of 20); many runs were still improving at the cap, which the second MoE sweep confirmed (§5).

---
## 5. Hyper-parameter / design sweep on the best backbone (MoE) — `p2s_moe2` (detail in `phase2-moe-sweep-results.md`)

Each setup changes one thing from the p2_13 recipe. Fold2 / fold1 / mean:

| Setup | Fold2 | Fold1 | Mean | Verdict |
|---|---|---|---|---|
| lr_high_long (LR 1e-3, 30 epochs) | 0.4882 | 0.4945 | **0.4914** | best; schedule was the main weakness (+0.03) |
| window1024 (stride 512) | 0.4868 | 0.4942 | 0.4905 | +0.03, higher precision (0.54), fastest |
| oversample_rare (APP/RESP/WIT windows ×3) | 0.4829 | 0.4824 | 0.4827 | helps APP (0.29 vs 0.25), stable across folds |
| adapter_dim128 | 0.4779 | 0.4858 | 0.4819 | +0.02, more parameters, not better than the above |
| bilstm2layer | 0.4771 | 0.4627 | 0.4699 | fold2 only; unstable |
| base (p2_13 recipe) | 0.4539 | 0.4652 | 0.4596 | reference (seed 1: 0.4564) |
| bilstm512 | 0.4579 | – | – | no gain |
| focal γ=2 | 0.4219 | – | – | hurts (−0.03) |
| label smoothing 0.1 | 0.4016 | – | – | hurts most (−0.05): precision 0.35 |
| span head + span decoding | 0.3766 | – | – | P 0.56, R 0.31; same model decoded by Viterbi 0.4431 → span decoding worse |
| header256, combo_hdr_512_ls | crashed | | | my window bug (WINDOW 2048 + HEADER 256 > 2048 positions); fixed, re-tested in `p2s_moe3` |

Decoding ablation (same trained model): Viterbi ≈ merge-average (±0.003) ≫ argmax (−0.04 to −0.08). Constrained BIO decoding is essential.

Produced zips (test entities): lr_high_long 4072, window1024 4088, oversample_rare 3999, adapter_dim128 4171. Expected Codabench ≈ fold2 + 0.05–0.09 ≈ 0.54–0.58. Recommended submission order: lr_high_long, window1024, (oversample_rare); skip adapter_dim128. The top four lie within 0.01 → treat as interchangeable.

---
## 6. Knowledge distillation (detail in `phase2-kd-results.md`)

Teacher p2_13 (MoE 218.7M, Houlsby + BiLSTM), student Phase-1 LSTM 84M (top layer + BiLSTM head, 22.6M trainable). Loss 0.5·CE + 0.5·T²·KL, T=2.

| Run | Fold2 F1 | Codabench | % of teacher F1 (fold2) |
|---|---|---|---|
| Teacher | 0.4562 | 0.5335 | 100 |
| Student, no KD | 0.4285 | 0.5008 | 93.9 |
| A: KD, labelled data only (clean KD) | 0.4245 | 0.5109 | 93.1 (CB 95.8) |
| B: KD + teacher-labelled unlabelled text (fold2 + test text) | 0.4745 | **0.5458** | 104.0 (CB 102.3) |

- **K1** KD on labelled data alone: no fold2 gain (−0.004, noise), +0.010 on Codabench; retains 93–96% of the teacher.
- **K2** The large gain of B comes from the extra (teacher-labelled) input text, i.e. self-training / ensembling effect, not from softer labels: precision 0.43 → 0.52, spurious spans 993 → 768; B beats its own teacher by +0.012 on Codabench.
- **K3** Efficiency: 2.2× fewer parameters (86M vs 192M NER-model params) but NOT faster or lighter on the T4: 4.1–4.3k tokens/s vs 11.4–12.1k, peak 1.8 vs 0.97 GB, because the LSTM is sequential. A Transformer student would be the efficiency choice.
- **K4 (disclosure)**: B uses the test documents' text (never labels) — transductive. State this in the report; A is the clean KD result.

---
## 7. Efficiency / size / memory (T4, fp16)

| Model + setup | Total params | Trainable | Peak GB | Train s (fold1 CV) | Training tok/s |
|---|---|---|---|---|---|
| MoE full FT + BiLSTM | 218.7M (111M active) | 190.0M (86%) | 5.4 | 563 | 6.6k |
| MoE adapter + BiLSTM | 191.6M NER model | 3.70M (1.69%) | 2.5 | 927–952 | 7.8k |
| MoE LoRMA+ + BiLSTM | – | 2.61M (1.19%) | 2.5 | 983 | 7.6k |
| MoE LoRMA+ linear | – | 0.51M (0.23%) | 2.4 | 330–714 | 10–11k |
| GQA adapter + BiLSTM | 111.1M | 3.70M (3.33%) | 1.38 | 771 | 9.7k |
| GQA partial FT top-2 | – | 14.5M | 1.00 | 493 | 11.3k |
| GQA full FT + BiLSTM | – | 82.4M | 2.67 | 416 | 9.0k |
| LSTM partial FT top-1 | 83.9M | 22.6M | 1.03 | 2620 | – |
| LSTM frozen + BiLSTM | – | 3.69M | 0.67 | 2254 | – |
| MoR frozen + BiLSTM | 120.8M | 2.11M | 0.67 | 607 | – |
| GDN adapter + BiLSTM | 144.3M | 3.90M | 2.10 | 1789 | – |

(KD runs: LSTM student inference 4.1–4.3k tokens/s vs 11.4–12.1k for the MoE teacher, peak 1.8 vs 0.97 GB.) PEFT runs use ~half the peak memory of full FT; the BiLSTM head costs ~+2.1M parameters, 25–35% lower throughput and +0.2 GB.

---
## 8. Data, class-imbalance, boundary and error findings (cross-cutting)

- **E1 Boundary problem.** With the Phase-1 A1 BPE (punctuation glued to the previous word) 2828 of 8393 gold entities (33.7%) do not lie on token boundaries. A learned trim-table repair (per label × edge character, majority vote on the training fold) adds +0.12–0.18 macro-F1 in every run (e.g. 0.245 → 0.369 without/with repair on p2_01). Splitting punctuation before BPE fixes alignment (2828 → 60 misaligned) but raised Phase-1 LM loss (3.106 → 3.559) and gave no net gain (+0.017 fold2, −0.016 Codabench).
- **E2 Remaining boundary errors** are ~3:2 "prediction too short" vs "too long"; the most common single difference is a missing trailing "." (48–69 cases per model), mostly abbreviations the gold includes ("Ltd.", "U.P.", "Anr.") but the majority-vote trim removes. A context-aware repair (keep "." after abbreviation-like tokens) is a cheap, untested improvement.
- **E3 Class imbalance.** Class-weighted CE (inv_sqrt) is the most robust single change on a linear head (+0.036 fold2, +0.042 Codabench on MoE); inverse-frequency weights are harmful (GQA 0.417 → 0.261, precision 0.21); oversampling weak labels helps APP (0.25 → 0.29); label smoothing and focal loss hurt.
- **E4 Hard entity types** (all models): APP, RESP, WIT stay at F1 0.1–0.3 (best APP 0.29, RESP 0.20, WIT 0.30); AUTH and PREC ≈ 0.35; easy: DATE 0.90–0.93, COURT 0.78–0.82, STAT 0.65–0.71. Dominant label confusions: A.COUNSEL↔R.COUNSEL (35–67), APP↔RESP (15–41), RESP→AUTH/PREC, PREC↔COURT, WIT→PREC. The role of a party/counsel is usually stated to the right of the mention ("…Appellant", "learned counsel for the respondent"), which is why right-context (BiLSTM) helps these labels most.
- **E5 Context effects.** Removing the causal mask in the pretrained MoE hurt (−0.036 fold2 / −0.031 Codabench): causally pretrained layers are disrupted; adding right context on top of the unchanged causal layers is what helps. Shorter windows help (1024 vs 2048: +0.03, precision 0.54 vs 0.49). Header-copy idea: not yet tested (in `p2s_moe3`).
- **E6 Boilerplate false positives** are worst in the first 5% of each document (cause-title header): precision 0.35 for the linear head, 0.54 for the BiLSTM head on GQA; the BiLSTM head halves header spurious spans (231 → 117).
- **E7 Long documents.** Micro-F1 by document-length thirds is non-monotonic (e.g. GQA 0.646 / 0.517 / 0.580; GDN 0.558 / 0.410 / 0.482; MoE 0.677 / 0.510 / 0.564): length alone does not explain difficulty (with 11–13 documents per group, document content dominates). Recall/precision by region: the first 5% 0.56/0.54, 5–50% 0.59/0.60, last 50% 0.54/0.55 for the MoE base.
- **E8 Validation proxy.** Spearman correlation fold2 vs Codabench = 0.94 over 12 MoE runs; 0.045–0.09 offset; ordering inside a family holds except for near-ties.
- **E9 Fold variance.** Per-label F1 varies a lot across folds (APP 0.10–0.29, CASENO 0.35–0.57), another reason to rank on the two-fold mean.

---
## 9. Ensemble (not yet submitted on Codabench) — builders `build_p2_ensemble.py` (16a, 16b, 16c)

Method: hard span voting over saved prediction files, greedy non-overlap, modes exact|boundary, weights equal|F1, vote threshold tau grid, abbreviation dot-rule; honest estimate computed with a half-split of fold2 (parameters fit on one half, scored on the other). Members need only `prediction.json` (Codabench zips in the Kaggle dataset `phase2`). Soft voting (probability averaging) is not possible for models whose weights were not saved; not built yet.
- 16b (cross-architecture, one-per-model): honest fold2 estimate **0.5071** vs 0.4766 for the best single member (+0.03) — largest benefit from combining different architectures. Codabench result pending (best candidate for the next submission).
- 16a (MoE-only variants) and 16c (MoE + KD student): notebooks built, not yet run by the student.
- Course status: the TA confirmed ensembles are allowed provided all four models meet the requirements.

---
## 10. Honest limitations and disclosures for the report

1. **LLM-use declaration (mandatory) — the student must write this and be precise.** Claude (the assistant) wrote the Phase-2 tooling: data/offset alignment, the training/evaluation harness, the adapter / LoRA / LoRMA / BiLSTM-head / span-head modules, the KD and ensemble code, the notebook builders and the error-analysis code. The Phase-1 models, the choice of models/techniques, the running of every notebook and the reading of results are the student's. **Policy risk to handle honestly:** the course policy says the LLM must not write the whole architecture, do hyper-parameter tuning or draw result conclusions. The sweep setup lists (which LR, epochs, window, loss variants to try) and the interpretation text in these docs were proposed by the assistant. The student should review them, state in the declaration who proposed what, and write the final conclusions in their own words (the findings above are drafts to confirm or rephrase).
2. KD variant B uses unlabelled test text (transductive) — disclose.
3. 35 documents per fold: differences < 0.015 are noise; seed noise 0.003; top-4 setups of the MoE sweep cannot be ranked reliably.
4. Header-copy and several combinations are untested at time of writing (`p2s_moe3` pending); prompt tuning not run; canon-layers variant of the 4th architecture not run; LoRA + BiLSTM on MoE not run.
5. Hyper-parameters such as LR, epochs, adapter dim, rank were set by the student/assistant as untuned placeholders and then swept only on MoE (sweep 5); other architectures keep the placeholder recipe.
6. The Codabench numbers for the MoE sweep winners and the ensemble are pending; this file must be updated when they arrive.

---
## 11. Document index (all in this Project)

| Doc | Content |
|---|---|
| `capstone-overview.md` | brief, deadlines, submission format, phase overview |
| `phase2-lner-data-findings.md` | L-NER data facts |
| `phase2-lner-plan-and-results.md` | brief §8.1–8.5, pipeline design, all 14 MoE runs (tables, per-label, errors), findings F1–F9 |
| `phase2-sweep-results.md` | sweeps 1–4: GQA, MoR, GDN, LSTM (rankings, per-label, errors, findings G/M/D/L, Codabench table X1–X4) |
| `phase2-kd-results.md` | knowledge distillation A/B (findings K1–K5) |
| `phase2-moe-sweep-results.md` | second MoE sweep (findings M1–M9, header bug) |
| `phase2-MASTER-results-for-report.md` | THIS FILE |
| `code/` | `p2_core.py`, `p2_backbone_*.py`, `build_p2_*.py` (notebook builders), `phase2_lner_align.py`, tests |

## 12. Pending experiments (as of this update)

- `p2s_moe3` (stacked winners + header + window512 + oversample + LoRMA, 6 setups, ~5–6 h) — student to run.
- Codabench submission of `p2s_moe2` zips (lr_high_long first, window1024 second).
- Ensembles 16a / 16b / 16c (minutes each) and a possible rebuild with the new sweep winners.
- Optional: soft-voting ensemble, prompt tuning, LoRA + BiLSTM.

---
## 13. UPDATE PROTOCOL (for any future session that continues this work)
The student sends notebook outputs / Codabench scores as they arrive and will ask for the full set of observations at the end. On every new result:
1. Add the numbers to the right table here (§2 leaderboard, §3 architectures, §4 techniques, §5 MoE sweeps, §6 KD, §9 ensembles) and mark pending items done in §10 and §14.
2. Add the new observation to the findings list of the relevant section, with the reasoning; say when a result is within noise (±0.015 on a 35-document fold, seed noise 0.003).
3. Update `phase2-METHODS-detailed.md` if a new method was added or changed, and `project-changelog-phase2.md` (dated entry; mistakes and who proposed what).
4. Keep the per-experiment docs (`phase2-moe-sweep-results.md`, `phase2-kd-results.md`, `phase2-sweep-results.md`) as the detail store; this file stays the summary.
5. Never overwrite history: corrections go in as dated notes.

## 14. UPDATE LOG
- 2026-10-11 11:00 — file created from the project docs and the returned notebooks (14 MoE runs, 4 architecture sweeps, KD A/B, MoE sweep `p2s_moe2`, ensemble 16b estimate).
- PENDING: Codabench scores of the four `p2s_moe2` zips · ensemble 16a/16b/16c results · `p2s_moe3` results · any further submissions.
