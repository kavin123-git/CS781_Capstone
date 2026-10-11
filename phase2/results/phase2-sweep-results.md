# Phase 2 (L-NER) — per-architecture SWEEP results (companion to `phase2-lner-plan-and-results.md`)

Sweep protocol (`build_p2_sweeps.py`): every setup trained on fold1, epoch chosen and scored (strict macro-F1) on fold2; the top 5 re-run fold2→fold1 and ranked by the two-fold mean; top 4 refit on fold1+fold2 → Codabench zips. Shared settings: class weights inv_sqrt (unless stated), window 2048/1024, batch 2×4, AdamW wd 0.01, warmup 6%, head LR 1e-3, fp16, causal attention, exact Phase-1 tokens + learned boundary repair, Viterbi BIO decoding. "bilstm" = 1-layer BiLSTM head (256/direction).

---
## Sweep 1 — GQA 160M (dense decoder, GQA + RoPE, 111.1M params) — `p2s_gqa160`, run 2026-10-07/08, 242.7 min on one T4
Phase-1 checkpoint `phase1_decoder_only_gqa_kvcache_full_layernorm_pre_adamw_rope_ctx2048_160Mw_a1bpe_160Mw_1ep_checkpoint.pt` (Phase-1 Codabench 0.8254). Next-token loss on NER docs: **3.366** nats (MoE: 3.106).

### Ranking (fold2 = fold1→fold2 score; fold1 = fold2→fold1 score)
| Rank | Setup | Fold2 F1 | Fold1 F1 | Mean | P | R | best ep | Trainable (% of 111.1M) | Peak GB | Train s | tok/s | Codabench |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | adapter_bilstm (Houlsby) | 0.4655 | 0.4784 | **0.4719** | 0.497 | 0.465 | 17 | 3.70M (3.33%) | 1.38 | 771 | 9,661 | **0.5200** |
| 2 | lorma_bilstm (LoRMA+) | 0.4552 | 0.4700 | 0.4626 | 0.487 | 0.454 | 18 | 2.61M (2.35%) | 1.41 | 804 | 9,273 | 0.4999 |
| 3 | adapter_pfeiffer_bilstm | 0.4486 | 0.4706 | 0.4596 | 0.475 | 0.452 | 15 | 2.91M (2.62%) | 1.26 | 753 | 9,892 | 0.5113 |
| 4 | lorma_pi_bilstm (LoRMA_π) | 0.4515 | 0.4571 | 0.4543 | 0.490 | 0.451 | 17 | 2.61M (2.35%) | 1.45 | 819 | 9,094 | 0.5071 |
| 5 | lora_bilstm | 0.4449 | 0.4525 | 0.4487 | 0.491 | 0.440 | 20 | 2.75M (2.48%) | 1.43 | 807 | 9,231 | – |
| 6 | partial_ft_top2_bilstm (top 2 of 8 blocks) | 0.4440 | – | – | 0.462 | 0.453 | 12 | 14.5M (13.1%) | **1.00** | 493 | **11,334** | – |
| 7 | bitfit_bilstm | 0.4429 | – | – | 0.475 | 0.442 | 17 | 2.17M (1.96%) | 1.28 | 771 | 9,663 | – |
| 8 | full_ft_bilstm | 0.4173 | – | – | 0.423 | 0.432 | 10 (last) | 82.4M (74.2%) | 2.67 | 416 | 8,963 | – |
| 9 | adapter (linear head) | 0.4072 | – | – | 0.388 | 0.449 | 17 | 1.61M (1.45%) | 1.22 | 502 | 14,844 | – |
| 10 | full_ft (linear head) | 0.3715 | – | – | 0.352 | 0.416 | 10 (last) | 80.4M (72.3%) | 2.53 | 284 | 13,116 | – |
| 11 | lorma (linear head) | 0.3512 | – | – | 0.323 | 0.403 | 17 | 511K (0.46%) | 1.26 | 535 | 13,937 | – |
| 12 | full_ft_bilstm, class weights **inv** | 0.2606 | – | – | 0.210 | 0.369 | 10 | 82.4M | 2.67 | 420 | 8,861 | – |
Refits (fold1+fold2): adapter_bilstm 17 ep (806 s, 3989 test entities), lorma_bilstm 15 ep (744 s, 4152), adapter_pfeiffer_bilstm 16 ep (722 s, 4044), lorma_pi_bilstm 15 ep (756 s, 4106). Test inference ~17 s per model. **Codabench:** adapter_bilstm 0.5200 · Pfeiffer 0.5113 · LoRMA_π 0.5071 · LoRMA+ 0.4999 (Codabench − fold2 = +0.054 / +0.063 / +0.056 / +0.045).

### Per-label fold2 F1
| Setup | APP | RESP | A.CNSL | R.CNSL | JUDGE | WIT | AUTH | COURT | STAT | PREC | DATE | CASENO |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| adapter_bilstm | .259 | .118 | .457 | .541 | .552 | .169 | .323 | .790 | .696 | .288 | .917 | .475 |
| lorma_bilstm | .254 | .096 | .495 | .515 | .507 | .143 | .293 | .791 | .658 | .287 | .912 | .512 |
| adapter_pfeiffer_bilstm | .247 | .112 | .448 | .481 | .539 | .095 | .311 | .787 | .658 | .296 | .913 | .497 |
| lorma_pi_bilstm | .254 | .103 | .451 | .471 | .553 | .151 | .295 | .800 | .638 | .308 | .915 | .479 |
| lora_bilstm | .287 | .105 | .442 | .456 | .528 | .139 | .276 | .793 | .632 | .301 | .908 | .473 |
| partial_ft_top2_bilstm | .236 | .111 | .467 | .448 | .495 | .175 | .302 | .777 | .634 | .301 | .906 | .476 |
| bitfit_bilstm | .259 | .138 | .423 | .429 | .537 | .132 | .305 | .784 | .629 | .291 | .914 | .476 |
| full_ft_bilstm | .236 | .111 | .426 | .358 | .451 | .130 | .293 | .769 | .613 | .284 | .902 | .435 |
| adapter | .142 | .043 | .424 | .478 | .400 | .226 | .305 | .761 | .636 | .261 | .873 | .336 |
| full_ft | .115 | .046 | .412 | .336 | .369 | .129 | .263 | .754 | .585 | .255 | .873 | .322 |
| lorma | .108 | .043 | .366 | .316 | .363 | .166 | .269 | .740 | .510 | .188 | .830 | .317 |
| full_ft_bilstm_inv | .108 | .046 | .149 | .132 | .121 | .157 | .348 | .537 | .456 | .203 | .632 | .236 |
Fold2→fold1 (confirmed setups) per-label F1 is in the run log; notable: APP drops to 0.10–0.14 and RESP rises to 0.13–0.19 on fold1, CASENO 0.48–0.57.

### Errors on fold2 (correct / boundary / boundary+label / label / spurious / missed)
adapter_bilstm 2407/706/105/156/712/923 · lorma_bilstm 2350/725/96/152/712/974 · lora_bilstm 2299/740/122/133/662/1015 · bitfit_bilstm 2323/777/119/133/712/962 · partial_ft_top2_bilstm 2354/770/113/167/861/902 · lorma_pi_bilstm 2348/725/117/137/706/978 · adapter_pfeiffer_bilstm 2376/733/110/158/755/923 · full_ft_bilstm 2317/894/122/160/970/838 · full_ft 2290/1064/263/149/1547/647 · adapter 2388/922/233/177/1496/622 · lorma 2180/1094/313/159/1812/664 · full_ft_bilstm_inv 2150/1471/227/209/3781/342.

### Detailed error analysis (fold2)
- **Boundary error kinds** (adapter_bilstm): prediction too SHORT 428, too LONG 248, shifted 30; 125 differ only by punctuation/space. Same pattern in every setup (too short ≈ 1.5–1.9× too long).
- **Most common missing/extra text**: a trailing **"."** missing (48–57 cases in every BiLSTM setup), then "]" (11–24), ")" (9–13), " Act", " Government", "Review Petition". The trailing "." case is mostly abbreviations that gold includes ("Ltd.", "U.P.", "Anr.") but the learned trim repair removes (it decides per label × character, by majority) → a context-aware repair (do not trim "." after an abbreviation-like token) is a cheap, untested improvement.
- **By region of the document** (adapter_bilstm): header (first 5% of characters, the cause-title boilerplate) precision 0.54 / recall 0.55; 5–50% precision 0.64 / recall 0.60; 50–100% precision 0.54 / recall 0.52. Linear-head full FT: header precision only 0.35 (231 spurious of 791 predictions) → **boilerplate false positives are worst in the header and the BiLSTM head halves them** (231→117).
- **By document length** (thirds of fold2 by characters): short (17–24k chars) micro-F1 0.646, middle (24–30k) 0.517, long (30–114k) 0.580 for adapter_bilstm; same non-monotonic pattern in all setups → length alone does not explain difficulty (content/entity mix of individual documents matters more with 11–13 docs per group).

### GQA findings
- **G1. The BiLSTM head result replicates on a second architecture**: adapter +0.058, LoRMA +0.104, full FT +0.046 fold2 over the linear head (same pattern as the MoE: +0.03 / +0.094 / +0.059).
- **G2. PEFT > full FT on GQA**: every BiLSTM-PEFT setup (0.443–0.466) beats full FT + BiLSTM (0.417) — full FT updates 74% of the weights on 35 documents (memorisation; its best epoch was the last of 10, so it may also be under-scheduled at LR 3e-5). Partial FT of the top 2 blocks gives 0.444 with the lowest memory (1.0 GB) and highest speed.
- **G3. Technique ranking with the BiLSTM head**: adapters (Houlsby) ≥ LoRMA+ ≥ Pfeiffer ≈ LoRMA_π ≥ LoRA ≈ partial FT ≈ BitFit, all within 0.025 — same conclusion as MoE F3: once the head supplies right context, how the backbone is adapted matters little. **LoRMA+ beats LoRA by +0.010 fold2 / +0.014 two-fold mean** here (MoE linear head: LoRA slightly ahead). LoRMA_π ≈ LoRMA+ (−0.008 mean) at the same parameter count. Pfeiffer adapters (after FFN only, 21% fewer params) ≈ Houlsby (−0.012 mean).
- **G4. Stronger class weights hurt badly**: inverse-frequency weights (inv) → 0.261 vs 0.417 with inv_sqrt; 3781 spurious spans, precision 0.21 — over-weighting rare tags makes the model tag far too much.
- **G5. Transfer of Phase-1 representations**: GQA (111M, Phase-1 BERTScore 0.8254) with adapters+BiLSTM reaches 0.4655 fold2 — about the same as the 2× larger MoE (218M, 0.8376) with the same setup (0.4562). Better language-modelling quality in Phase 1 did not translate into better NER here; the dense model is also half the memory (1.38 vs 2.52 GB).
- **G6. Two-fold check**: fold2→fold1 scores are 0.006–0.022 higher than fold1→fold2 for all 5 confirmed setups; the top-2 order is unchanged (adapter_bilstm, lorma_bilstm).

---
## Sweep 2 — MoR 160M (Mixture-of-Recursions decoder, 120.8M params) — `p2s_mor2`, run 2026-10-08, 274.5 min on one T4
Phase-1 checkpoint `..._160Mw_a1bpe_mor160Mw_1ep_checkpoint.pt`. Next-token loss on NER docs: **5.217** nats (GQA 3.366, MoE 3.106) — by far the weakest LM of the family.
**Fix applied before this run** (first attempt `p2s_mor` produced train loss NaN and F1 = 0): MoR's routing thresholds (`tau`) are buffers that the Phase-1 code updates inside forward() in training mode; during fine-tuning they drifted, an fp16 overflow made them NaN, and every later batch was NaN. This run freezes routing (`MOR_FREEZE_ROUTING=True`: same routing rule, thresholds fixed at their Phase-1 values) and skips non-finite batches. Result: **0 skipped batches, no NaN in any setup**. Sweep list as GQA but `full_ft_bilstm_inv` replaced by `frozen_bilstm`; no prefix/prompt (not wired for the recursive forward).

### Ranking
| Rank | Setup | Fold2 F1 | Fold1 F1 | Mean | P | R | best ep | Trainable (% of 120.8M) | Peak GB | Train s |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | lorma_pi_bilstm | 0.3406 | 0.3536 | **0.3471** | 0.369 | 0.338 | 19/19 | 2.42M (2.00%) | 1.65 | 875 |
| 2 | lorma_bilstm (LoRMA+) | 0.3372 | 0.3399 | 0.3386 | 0.365 | 0.335 | 18/14 | 2.42M (2.00%) | 1.59 | 859 |
| 3 | bitfit_bilstm | 0.3308 | 0.3356 | 0.3332 | 0.365 | 0.326 | 20/13 | 2.18M (1.80%) | 1.41 | 823 |
| 4 | lora_bilstm | 0.3306 | 0.3337 | 0.3322 | 0.354 | 0.331 | 19/20 | 2.51M (2.08%) | 1.61 | 872 |
| 5 | adapter_pfeiffer_bilstm | 0.3287 | 0.3317 | 0.3302 | 0.360 | 0.326 | 20/18 | 3.20M (2.65%) | 1.38 | 790 |
| 6 | adapter_bilstm (Houlsby) | 0.3244 | – | – | 0.347 | 0.328 | 18 | 3.70M (3.06%) | 1.51 | 809 |
| 7 | frozen_bilstm (head only) | 0.3173 | – | – | 0.329 | 0.321 | 20 | 2.11M (1.75%) | **0.67** | 607 |
| 8 | partial_ft_top2_bilstm | 0.3147 | – | – | 0.316 | 0.331 | 15 | 14.52M (12.0%) | 1.39 | 572 |
| 9 | full_ft_bilstm | 0.2856 | – | – | 0.266 | 0.324 | 10 (last) | 92.2M (76.3%) | 2.88 | 417 |
| 10 | lorma (linear head) | 0.2292 | – | – | 0.190 | 0.299 | 19 | 0.33M (0.27%) | 1.44 | 574 |
| 11 | adapter (linear head) | 0.2099 | – | – | 0.173 | 0.279 | 16 | 1.61M (1.33%) | 1.35 | 534 |
| 12 | full_ft (linear head) | 0.2061 | – | – | 0.171 | 0.269 | 10 (last) | 90.1M (74.6%) | 2.74 | 290 |
Refits (fold1+fold2) → zips: lorma_pi_bilstm 19 ep (1056 s, 3946 test entities), lorma_bilstm 16 ep (860 s, 4078), bitfit_bilstm 16 ep (823 s, 4040), lora_bilstm 20 ep (1099 s, 3889). Test inference ~17 s. **Codabench: lorma_pi_bilstm 0.4189, bitfit_bilstm 0.4079** (lorma_bilstm and lora_bilstm zips not submitted). Codabench − fold2 = +0.078 / +0.077.

### Per-label fold2 F1
| Setup | APP | RESP | A.CNSL | R.CNSL | JUDGE | WIT | AUTH | COURT | STAT | PREC | DATE | CASENO |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| lorma_pi_bilstm | .034 | .086 | .260 | .290 | .223 | .073 | .325 | .785 | .619 | .244 | .817 | .332 |
| lorma_bilstm | .027 | .089 | .256 | .282 | .247 | .086 | .329 | .786 | .623 | .212 | .801 | .309 |
| bitfit_bilstm | .036 | .127 | .260 | .267 | .210 | .073 | .337 | .770 | .610 | .220 | .783 | .276 |
| lora_bilstm | .040 | .093 | .191 | .297 | .229 | .058 | .324 | .790 | .609 | .218 | .804 | .314 |
| adapter_pfeiffer_bilstm | .028 | .128 | .200 | .256 | .226 | .045 | .332 | .781 | .628 | .227 | .787 | .308 |
| adapter_bilstm | .027 | .079 | .237 | .281 | .189 | .059 | .313 | .772 | .614 | .222 | .785 | .315 |
| frozen_bilstm | .035 | .083 | .225 | .298 | .214 | .014 | .321 | .769 | .608 | .191 | .772 | .278 |
| partial_ft_top2_bilstm | .041 | .048 | .280 | .277 | .172 | .056 | .309 | .748 | .606 | .225 | .759 | .254 |
| full_ft_bilstm | .034 | .047 | .267 | .263 | .067 | .072 | .311 | .709 | .552 | .188 | .716 | .201 |
| lorma | .034 | .022 | .138 | .244 | .089 | .090 | .225 | .659 | .412 | .095 | .593 | .150 |
| adapter | .012 | .030 | .123 | .139 | .076 | .056 | .238 | .656 | .390 | .075 | .587 | .137 |
| full_ft | .020 | .042 | .139 | .158 | .055 | .025 | .233 | .627 | .390 | .087 | .599 | .099 |

### Errors on fold2 (correct / boundary / boundary+label / label / spurious / missed)
lorma_pi_bilstm 2019/828/165/141/753/1176 · lorma_bilstm 2000/852/173/141/785/1192 · bitfit_bilstm 1954/860/170/144/772/1209 · lora_bilstm 1991/846/161/151/815/1195 · adapter_pfeiffer_bilstm 1976/848/182/148/749/1207 · adapter_bilstm 1975/883/180/158/837/1166 · frozen_bilstm 1930/922/184/144/834/1181 · partial_ft_top2_bilstm 1986/984/199/169/1121/1056 · full_ft_bilstm 1952/1154/273/189/1768/866 · lorma 1813/1401/583/212/3407/621 · adapter 1786/1430/701/226/3580/521 · full_ft 1729/1466/670/193/3310/632.
- Header region (first 5% of characters) precision: BiLSTM setups 0.33–0.37 (GQA best 0.54); linear-head setups 0.12–0.14 (≈600 spurious spans in the header alone). Linear-head boundary errors are dominated by **" of India"** and **" and"** being cut off/added (party names like "Union of India", "X and Y"); BiLSTM setups go back to the usual trailing "." / "]" pattern.

### MoR findings
- **M1. Routing must be frozen for fine-tuning.** Routing buffers that update during training are a Phase-1 training mechanism, not a model parameter; with 35 NER documents and fp16 they diverged to NaN. Freezing them (inference-time behaviour) made all 12 setups train stably. Reportable as an engineering lesson for adaptive-computation models.
- **M2. Weak LM → weak NER, but less than expected.** Sanity loss 5.22 nats (1.86 nats worse than GQA) and best mean F1 0.347 vs GQA 0.472 (−0.125). Still slightly above the GDN encoder (0.336) whose LM loss is much better (3.51) → LM loss does not predict NER quality across architecture types.
- **M3. The backbone barely adapts.** frozen_bilstm (head only, 1.75% params, 0.67 GB) gets 0.317, only 0.024 below the best; all 6 BiLSTM-PEFT setups lie within 0.016 (0.324–0.341) — differences between techniques are at noise level (fold-to-fold differences are of similar size, 0.003–0.013).
- **M4. BiLSTM head again the largest effect**: adapter +0.115, LoRMA+ +0.108, full FT +0.080 over the linear head. Linear-head setups have precision 0.17–0.19 with 3300–3600 spurious spans; the BiLSTM head cuts spurious spans by ~4× (3407 → 785 for LoRMA).
- **M5. LoRMA_π is the best MoR setup** (first architecture where it wins), LoRMA+ second — both LoRMA variants beat LoRA (+0.015 / +0.006 mean) at the same parameter count. BitFit (bias only) is 3rd — competitive at 1.8% params.
- **M6. Full FT is worst again** (0.286 with BiLSTM, 0.206 linear; best epoch = last of 10) — same as GQA G2 / MoE.
- **M7. Under-training signal**: best epoch 18–20 of 20 for most PEFT setups → a longer schedule might add a little, but the gap to GQA/MoE is far too large to close.

---
## Sweep 3 — GDN encoder (encoder of the Gated-DeltaNet encoder-decoder, 144.3M params total) — `p2s_gdn`, run 2026-10-08, 583.3 min (T4; part of the run on CPU after the GPU quota ran out, then restarted on GPU with CONFIRM_TOP_M=5 → cached results reused)
Phase-1 checkpoint `..._160Mw_a1bpe_gdnencdec160Mw_1ep_checkpoint.pt`. Encoder-decoder sanity loss (encoder reads 768 tokens, decoder predicts next 256): **3.513** nats. Only the encoder (6 bidirectional GDN blocks, forward + backward mixer each) is used; window 1024 / stride 512 (GDN is ~2× slower per token than attention at this length on T4). The encoder is **bidirectional by construction**.

### Ranking
| Rank | Setup | Fold2 F1 | Fold1 F1 | Mean | P | R | best ep | Trainable (% of 144.3M) | Peak GB | Train s |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | adapter_bilstm (Houlsby) | 0.3341 | 0.3386 | **0.3364** | 0.340 | 0.338 | 18/13 | 3.90M (2.70%) | 2.10 | 1789 |
| 2 | lorma_bilstm (LoRMA+) | 0.3025 | 0.3218 | 0.3121 | 0.293 | 0.325 | 15/20 | 3.00M (2.08%) | 2.45 | 2033 |
| 3 | adapter (linear head) | 0.2547 | 0.2803 | 0.2675 | 0.224 | 0.306 | 19/16 | 1.80M (1.25%) | 1.98 | 1493 |
| 4 | full_ft_bilstm | 0.2415 | 0.2645 | 0.2530 | 0.208 | 0.299 | 9/10 | 84.9M (58.9%) | 3.56 | 1031 |
| 5 | lorma_pi (linear head) | 0.2436 | 0.2435 | 0.2435 | 0.215 | 0.287 | 17/14 | 0.90M (0.63%) | 2.46 | 1906 |
| 6 | lorma (LoRMA+, linear) | 0.2342 | – | – | 0.215 | 0.261 | 19 | 0.90M (0.63%) | 2.34 | 1675 |
| 7 | adapter_pfeiffer (linear) | 0.2320 | – | – | 0.189 | 0.310 | 19 | 0.61M (0.43%) | 1.89 | 1632 |
| 8 | lora (linear) | 0.2292 | – | – | 0.202 | 0.271 | 18 | 1.00M (0.69%) | 2.23 | 1678 |
| 9 | full_ft (linear) | 0.2284 | – | – | 0.198 | 0.281 | 10 (last) | 82.8M (57.4%) | 3.45 | 885 |
| 10 | bitfit (linear) | 0.2163 | – | – | 0.173 | 0.300 | 20 | 0.05M (0.04%) | 2.15 | 1587 |
| 11 | partial_ft_top2 (linear) | 0.2054 | – | – | 0.166 | 0.283 | 14 | 17.4M (12.1%) | 1.29 | 817 |
| 12 | frozen (linear head only) | 0.1607 | – | – | 0.147 | 0.211 | 20 | 0.02M (0.01%) | 0.58 | 814 |
Refits → zips: adapter_bilstm 16 ep (2129 s, 4362 test entities), lorma_bilstm 18 ep (2686 s, 4228), adapter 18 ep (1864 s, 6447), full_ft_bilstm 10 ep (1546 s, 5383). Test inference ~25–30 s. **Codabench: adapter_bilstm 0.3946** (+0.060 over fold2); the other three zips were not submitted.

### Per-label fold2 F1
| Setup | APP | RESP | A.CNSL | R.CNSL | JUDGE | WIT | AUTH | COURT | STAT | PREC | DATE | CASENO |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| adapter_bilstm | .114 | .105 | .251 | .283 | .188 | .025 | .310 | .761 | .574 | .221 | .828 | .348 |
| lorma_bilstm | .074 | .086 | .159 | .193 | .120 | .093 | .309 | .771 | .589 | .158 | .773 | .304 |
| adapter | .046 | .019 | .143 | .171 | .110 | .028 | .274 | .686 | .474 | .137 | .729 | .239 |
| full_ft_bilstm | .030 | .047 | .123 | .107 | .064 | .052 | .279 | .651 | .493 | .154 | .720 | .176 |
| lorma_pi | .049 | .063 | .165 | .167 | .083 | .021 | .256 | .697 | .464 | .114 | .688 | .157 |
| lorma | .029 | .044 | .103 | .129 | .081 | .021 | .235 | .736 | .469 | .104 | .673 | .187 |
| adapter_pfeiffer | .032 | .011 | .106 | .194 | .075 | .027 | .281 | .647 | .451 | .104 | .695 | .161 |
| lora | .029 | .050 | .197 | .121 | .063 | .020 | .247 | .701 | .453 | .095 | .653 | .122 |
| full_ft | .020 | .037 | .132 | .126 | .044 | .010 | .278 | .684 | .478 | .091 | .736 | .105 |
| bitfit | .039 | .021 | .113 | .080 | .040 | .030 | .276 | .609 | .471 | .078 | .691 | .146 |
| partial_ft_top2 | .014 | .010 | .075 | .146 | .018 | .067 | .246 | .599 | .438 | .055 | .646 | .151 |
| frozen | .000 | .000 | .039 | .052 | .000 | .000 | .183 | .605 | .332 | .021 | .632 | .065 |
Fold2→fold1 adapter_bilstm: CASENO 0.546 (vs 0.348 on fold2), DATE 0.844, JUDGE only 0.122 — large per-label fold variance.

### Errors on fold2 (correct / boundary / boundary+label / label / spurious / missed)
adapter_bilstm 2024/831/193/162/998/1095 · lorma_bilstm 2000/954/284/195/1499/899 · adapter 1921/1167/483/225/2587/688 · full_ft_bilstm 1908/1255/475/205/2973/628 · lorma_pi 1807/1204/475/164/2477/848 · lorma 1722/1123/456/143/2070/951 · adapter_pfeiffer 1960/1365/585/211/3590/486 · lora 1740/1327/508/162/2623/773 · full_ft 1867/1214/659/178/2960/656 · bitfit 1961/1273/694/186/4212/495 · partial_ft_top2 1846/1358/787/225/4721/433 · frozen 1551/1429/866/85/3737/824.
- adapter_bilstm boundary kinds: too short 481, too long 309, shifted 41, punctuation/space only 90. Header precision 0.39 (linear-head setups 0.14–0.17). Doc-length thirds micro-F1 0.558 / 0.410 / 0.482 (same non-monotonic pattern as GQA).

### GDN findings
- **D1. A bidirectional backbone still needs the BiLSTM head**: +0.079 (adapter), +0.068 (LoRMA+ vs linear), +0.013 (full FT). So the BiLSTM gain on the causal decoders is **not only "adding right context"** — the GDN encoder already sees both sides, yet the head still gives the largest jump. The head adds sequential smoothing over tag decisions and a task-specific layer on top of features that were trained for a different objective (feeding a decoder through cross-mixing, not being read token-by-token).
- **D2. Weakest backbone of the sweeps** despite a good LM loss (3.51): frozen linear probe 0.161 (MoR not tested linear-frozen; GQA/MoE much higher with any head). The encoder's states were only ever trained to be consumed by the decoder, so they are less directly usable for token labelling than a decoder's last hidden states.
- **D3. Technique ranking is less flat than on the decoders**: adapter_bilstm leads lorma_bilstm by 0.024 (two-fold mean) — the only sweep where a clear winner stands out. Among linear-head PEFT: Houlsby adapters 0.255 > LoRMA_π 0.244 ≈ LoRMA+ 0.234 ≈ Pfeiffer 0.232 ≈ LoRA 0.229 > BitFit 0.216.
- **D4. Full FT and partial FT are again worst/near-worst** (0.228 / 0.205 linear; partial FT and BitFit have the most spurious spans, 4200–4700).
- **D5. Cost**: ~2× the training time of the decoders per setup with half the window (GDN recurrent scan on T4), 2.1–3.6 GB peak.

### Cross-architecture summary so far (best setup per family, two-fold mean fold F1)
| Family | Phase-1 NER-doc LM loss | Best setup | Mean F1 |
|---|---|---|---|
| MoE high-court (218M) | 3.106 | adapter_bilstm (run 13, Codabench 0.5335) | fold2 0.456 (single fold only) |
| GQA 160M (111M) | 3.366 | adapter_bilstm | 0.472 |
| MoR 160M (121M) | 5.217 | lorma_pi_bilstm | 0.347 |
| GDN enc-dec encoder (144M) | 3.513 (enc-dec) | adapter_bilstm | 0.336 |
Common to all: BiLSTM head = largest effect; PEFT ≥ full FT; class weights inv_sqrt; trailing-"." abbreviation boundary errors.

---
## Sweep 4 — LSTM 84M (3-layer LSTM, 768-d embeddings, 1536 hidden; 83.9M params) — `p2s_lstm84`, 2026-10-08/09, T4; stages 1-2 took 9.7 h (hit Kaggle's 12 h limit during the refits), refits re-run in a second session (3.5 h)
Phase-1 checkpoint `phase1_lightweight_lstm84M_full_checkpoint.pt` (Phase-1 Codabench 0.8237). Next-token loss on NER docs **4.018** nats. Window 2048/1024 as the decoders (the LSTM does not need it: shorter windows would be ~3× faster). The 3-layer LSTM is split into three single-layer LSTMs with copied weights so that adapters can sit between layers and the top layer can be fine-tuned alone. Sweep list: no LoRA/LoRMA/prefix (they need linear projections inside attention; the LSTM has none).

### Ranking
| Rank | Setup | Fold2 F1 | Fold1 F1 | Mean | P | R | best ep | Trainable (% of 83.9M) | Peak GB | Train s |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | partial_ft_top1_bilstm (top LSTM layer + head) | 0.4285 | 0.4425 | **0.4355** | 0.423 | 0.453 | 18/16 | 22.6M (26.9%) | 1.03 | 2620 |
| 2 | adapter_bilstm (Houlsby) | 0.4090 | 0.4484 | 0.4287 | 0.389 | 0.444 | 18/19 | 4.28M (5.1%) | 0.98 | 2943 |
| 3 | adapter_pfeiffer_bilstm | 0.4099 | 0.4346 | 0.4223 | 0.377 | 0.457 | 19/19 | 3.89M (4.6%) | 0.70 | 2290 |
| 4 | frozen_bilstm (head only) | 0.4053 | 0.4373 | 0.4213 | 0.375 | 0.454 | 20/19 | 3.69M (4.4%) | **0.67** | 2254 |
| 5 | bitfit_bilstm | 0.4043 | 0.4311 | 0.4177 | 0.376 | 0.453 | 17/19 | 3.72M (4.4%) | 1.02 | 3328 |
| 6 | full_ft_bilstm | 0.3998 | – | – | 0.382 | 0.436 | 15 | 86.3M (103%*) | 2.01 | 2505 |
| 7 | adapter (linear head) | 0.3419 | – | – | 0.292 | 0.432 | 14 | 0.63M (0.75%) | 0.80 | 2546 |
| 8 | full_ft (linear head) | 0.3345 | – | – | 0.285 | 0.421 | 14 | 82.7M (98.6%) | 1.87 | 2304 |
*the percentage exceeds 100 because the BiLSTM head adds 2.5M parameters on top of the 83.9M backbone.
Refits (fold1+fold2) → zips: partial_ft_top1_bilstm 17 ep (2986 s, 4461 test entities), adapter_bilstm 18 ep (3782 s, 5022), adapter_pfeiffer_bilstm 19 ep (2644 s, 5364), frozen_bilstm 20 ep (2719 s, 5137). Test inference ~51 s. **Codabench: partial_ft_top1_bilstm 0.5008, adapter_bilstm 0.5008, adapter_pfeiffer_bilstm 0.4921, frozen_bilstm 0.4909** (Codabench − fold2 = +0.072 / +0.092 / +0.082 / +0.086).

### Per-label fold2 F1
| Setup | APP | RESP | A.CNSL | R.CNSL | JUDGE | WIT | AUTH | COURT | STAT | PREC | DATE | CASENO |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| partial_ft_top1_bilstm | .159 | .122 | .402 | .511 | .447 | .207 | .331 | .751 | .588 | .276 | .900 | .449 |
| adapter_bilstm | .117 | .112 | .416 | .477 | .399 | .168 | .323 | .726 | .635 | .259 | .879 | .397 |
| adapter_pfeiffer_bilstm | .123 | .101 | .449 | .453 | .373 | .233 | .356 | .724 | .592 | .256 | .869 | .392 |
| frozen_bilstm | .135 | .126 | .389 | .467 | .382 | .219 | .347 | .711 | .574 | .252 | .872 | .389 |
| bitfit_bilstm | .113 | .091 | .434 | .455 | .381 | .194 | .345 | .722 | .587 | .255 | .872 | .403 |
| full_ft_bilstm | .160 | .105 | .366 | .464 | .359 | .118 | .328 | .748 | .617 | .239 | .885 | .408 |
| adapter | .130 | .076 | .442 | .413 | .155 | .106 | .287 | .722 | .529 | .207 | .812 | .225 |
| full_ft | .117 | .069 | .324 | .460 | .201 | .067 | .294 | .668 | .527 | .213 | .821 | .253 |

### Errors on fold2 (correct / boundary / boundary+label / label / spurious / missed)
partial_ft_top1_bilstm 2331/790/119/233/1105/826 · adapter_bilstm 2352/871/172/260/1352/676 · adapter_pfeiffer_bilstm 2410/887/204/247/1652/602 · frozen_bilstm 2376/854/178/249/1679/692 · bitfit_bilstm 2384/833/181/275/1678/641 · full_ft_bilstm 2320/862/197/261/1314/689 · adapter 2365/1024/357/354/3074/306 · full_ft 2292/1036/346/309/2576/420.

### LSTM findings
- **L1. The smallest model is the third-best architecture**: mean 0.436, behind GQA 0.472 (and MoE) but far above MoR 0.347 and GDN 0.336, with 84M parameters and peak memory ~1 GB. Phase-1 NER-document LM loss is the worst of the working models (4.02 vs 3.1–3.5) yet NER is good → LM loss and NER quality are not tied across architecture types (see also M2).
- **L2. Partial fine-tuning wins here** (top LSTM layer only, 27% of parameters) — the only sweep where it comes first. Explanation: the top recurrent layer sits directly under the head and is cheap to retrain; the lower layers carry general lexical/syntactic features. It also has the best precision (0.423) and the fewest spurious spans (1105 vs 1350–1680).
- **L3. The backbone barely matters again**: frozen backbone + BiLSTM head is 0.421, only 0.014 below the best; the five BiLSTM setups span 0.418–0.436. PEFT ≈ full FT + BiLSTM (0.400 fold2; best epoch 15 of 15 earlier cap) — full FT is not better than a frozen backbone.
- **L4. BiLSTM head gain is smaller than on the decoders** (+0.067 adapter, +0.065 full FT; +0.08–0.115 on decoders): the LSTM backbone is causal too, so the head adds right-hand context, but the backbone's own states are already good local features. The linear-head setups over-predict (3074 / 2576 spurious spans, precision 0.29).
- **L5. Weak labels**: APP ≤ .16, RESP ≤ .13, WIT ≤ .23 (same as every model); R.COUNSEL (.51), JUDGE (.45) and CASENO (.45–.57 on fold1) are mid-range, COURT .75 and DATE .90 near GQA.
- **L6. Cost**: LSTM epochs are ~2.5× slower than GQA's (147 s vs ~40 s for adapter+BiLSTM on a T4) because tokens must be processed sequentially (2048 steps × 3 layers, backward repeats them) and a 1536-wide step is too small to fill the GPU; PEFT does not shorten this, since activations still flow through all layers. The sweep took 9.7 h vs 4 h for GQA. Shorter windows with larger batches would help.
- **L7. Over-prediction at test time**: refits produce 4461–5364 entities vs ~4000 expected (GQA 3989–4152; MoR 3889–4078) → precision is the LSTM's weak side.

### Cross-architecture summary (best two-fold mean)
| Family | NER-doc LM loss | Best setup | Mean F1 |
|---|---|---|---|
| MoE high-court (218M) | 3.106 | adapter_bilstm (run 13, Codabench 0.5335) | fold2 0.456 (single fold only) |
| GQA 160M (111M) | 3.366 | adapter_bilstm | 0.472 |
| LSTM 84M | 4.018 | partial_ft_top1_bilstm | 0.436 |
| MoR 160M (121M) | 5.217 | lorma_pi_bilstm | 0.347 |
| GDN enc-dec encoder (144M) | 3.513 | adapter_bilstm | 0.336 |

### Codabench results of the sweep winners (added 2026-10-10) and what they show
| Family | Params | Submitted | Codabench | fold2 (same setup) | Codabench − fold2 |
|---|---|---|---|---|---|
| MoE high-court (p2_13 adapter_bilstm) | 218.7M (111M active) | yes | **0.5335** | 0.456 | +0.077 |
| GQA 160M adapter_bilstm | 111.1M | yes | 0.5200 | 0.4655 | +0.055 |
| LSTM 84M partial_ft_top1_bilstm / adapter_bilstm | 83.9M | yes | 0.5008 / 0.5008 | 0.4285 / 0.409 | +0.072 / +0.092 |
| MoR 160M lorma_pi_bilstm | 120.8M | yes | 0.4189 | 0.3406 | +0.078 |
| GDN encoder adapter_bilstm | 144.3M | yes | 0.3946 | 0.3341 | +0.060 |
- **X1. The ordering of the architectures on Codabench is MoE > GQA > LSTM >> MoR > GDN**, the same as on the validation folds; the sweep winners' fold2 scores predict the Codabench ordering within a family except for near-ties (GQA: LoRMA+ and LoRMA_π swap; LSTM: partial FT and adapters tie at 0.5008 although fold2 differed by 0.02).
- **X2. The 84M LSTM is within 0.033 of the 218M MoE teacher and 0.019 of the 111M GQA model** on the test set → a small, fully sequential model captures most of what the larger Transformers give on this task; the representations, not the size, are limited by 35 training documents. It leaves a small head-room for distillation (teacher − student = 0.033).
- **X3. Phase-1 quality (BERTScore on Phase 1: MoE 0.8376, GQA 0.8254, LSTM 0.8237, MoR 0.7954, GDN 0.8010) tracks the L-NER Codabench order** (0.5335 > 0.5200 > 0.5008 > 0.4189 / 0.3946) apart from MoR vs GDN, which swap. So the Phase-1 representations do transfer, with the weak MoR/GDN models transferring worst (MoR's LM loss on NER text is 5.2 nats).
- **X4. Codabench scores are 0.045–0.09 above fold2 for every setup** (test documents are easier/more typical, and the final models see twice the data); the offset is roughly constant within a family, so fold2 is a usable selection signal.
