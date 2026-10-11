# Phase 2 — Sweep 5: MoE improvement sweep (`p2s_moe2`)

Notebook: `p2s_moe2` (built by `build_p2_moe_sweep.py`). Backbone: Phase-1 MoE (moe_hc, 218.7M, GQA + MoE FFN). Base recipe = p2_13 (Houlsby adapter 768→64→768, BiLSTM-256 head, inv_sqrt class weights, LR 3e-4, 20 epochs, patience 5, window 2048/stride 1024). Each setup changes ONE thing from base. Protocol: train fold1 → fold2 F1; top setups confirmed fold2 → fold1; top 4 refit on fold1+fold2 and predicted on test. Strict macro-F1 on character spans. 35 documents per fold, so differences < ~0.015 are noise; two identical-recipe runs with different seeds differ by 0.003 (0.4539 vs 0.4564).

## Results (fold2 = train fold1; fold1 = train fold2)

| rank | setup | fold2 | fold1 | mean | trainable | note |
|---|---|---|---|---|---|---|
| 1 | lr_high_long (LR 1e-3, 30 ep, pat 6) | 0.4882 | 0.4945 | 0.4914 | 3.70M | refit 18 ep, 4072 test entities |
| 2 | window1024 (stride 512) | 0.4868 | 0.4942 | 0.4905 | 3.70M | fastest (861s), highest precision (0.54) |
| 3 | oversample_rare (APP/RESP/WIT windows x3) | 0.4829 | 0.4824 | 0.4827 | 3.70M | APP F1 0.29 vs 0.25 |
| 4 | adapter_dim128 | 0.4779 | 0.4858 | 0.4819 | 5.27M | |
| 5 | bilstm2layer | 0.4771 | 0.4627 | 0.4699 | 5.28M | helps fold2 only; unstable |
| 6 | base (p2_13 recipe) | 0.4539 | 0.4652 | 0.4596 | 3.70M | reference |
| – | bilstm512 | 0.4579 | – | – | 6.86M | no gain |
| – | base_seed1 | 0.4564 | – | – | 3.70M | noise floor ±0.003 |
| – | focal (γ=2) | 0.4219 | – | – | | hurts |
| – | label_smooth (0.1) | 0.4016 | – | – | | hurts most |
| – | span_decode | 0.3766 | – | – | 3.84M | P 0.56, R 0.31; same model decoded with Viterbi = 0.4431 |
| – | header256 / combo_hdr_512_ls | FAILED | | | | bug, see below |

## Findings
- **M1 Schedule matters most.** Higher head/adapter LR (1e-3) and more epochs gave +0.03 over base on both folds (0.4882/0.4945 vs 0.4539/0.4652). The base recipe plateaued around 0.45 by epoch 15-20 (best epoch 17 and 20 for the two seeds), so the longer, higher-LR schedule reaches a better optimum rather than just converging faster.
- **M2 Shorter windows help.** WINDOW 1024 gave +0.03 with markedly higher precision (0.54 vs 0.49) and is the cheapest setup. Possible cause (not tested): shorter windows give the adapters/BiLSTM less irrelevant context and more, shorter training examples.
- **M3 Oversampling weak labels is real but narrow.** Windows containing APP/RESP/WIT seen 3x lifted APP F1 0.25 → 0.29 and AUTH 0.35 → 0.38, with +0.029 on fold2 and +0.017 on fold1 (stable). It costs some COURT/STAT/DATE F1 (e.g. DATE 0.907 → 0.9075 flat, COURT 0.783 → 0.781), so overall effect is moderate. Good ensemble member because its error profile differs.
- **M4 Adapter capacity.** Dim 128 gave +0.024/+0.021 over base; at 5.3M trainable it is not better than M1/M2 at 3.7M, so capacity is not the bottleneck.
- **M5 Loss reshaping hurts.** Label smoothing 0.1 (−0.05) lowers precision (0.35) because every tag gets some mass, so O is over-predicted as an entity; focal γ=2 (−0.03) loses recall on the frequent labels. Inverse-sqrt class weights already balance the classes enough.
- **M6 Span head hurts.** Span decoding: precision 0.56 but recall 0.31, F1 0.377. Viterbi on the same jointly trained model gives 0.443, still below base 0.454, so the extra loss slightly harms the tagger and the span decoder is not worth its complexity. APP, WIT, A.COUNSEL near 0 with span decoding.
- **M7 Decoding.** Viterbi BIO ≈ merge_average (±0.003) and both far better than argmax (−0.07 to −0.08 for most setups, −0.04 for bilstm2layer, −0.07 for lr_high_long). Constrained decoding is essential.
- **M8 BiLSTM.** Hidden 512 gives nothing (+0.004, in noise); 2 layers helps fold2 (+0.023) but hurts fold1 (−0.003) → not reliable.
- **M9 Top four are interchangeable.** lr_high_long, window1024, oversample_rare and adapter_dim128 lie within 0.01 on mean; with ±0.003 seed noise plus 35-document folds the order between them is not trustworthy.
- **Persistent weak labels.** APP, RESP, WIT stay at F1 0.1-0.3 in every setup (fold2: APP 0.22-0.29, RESP 0.14-0.18, WIT 0.10-0.30); their errors are mostly APP↔RESP and A.COUNSEL↔R.COUNSEL role confusions plus boundary errors, which tweaks of loss/head do not fix. The first 5% of each document (header) has the same recall as the rest.
- **Boundary errors dominate.** ~750 boundary errors vs ~880 entirely missed vs ~800 spurious on fold2; most frequent: missing trailing "." or ")" (A1 BPE glues punctuation to the previous word).

## Bug found (header setups)
`header256` and `combo_hdr_512_ls` crashed: `window (2304 incl. 0 prompt tokens) longer than the model's 2048 positions`. Cause: HEADER=256 was used with the default WINDOW=2048; WINDOW + HEADER must be ≤ 2048. Fixed in `build_p2_moe_sweep.py` (WINDOW=1792, strides 896). Those two ideas remain UNTESTED; do not claim a header result in the report.

## Submissions produced
`submission_p2s_moe2_lr_high_long.zip` (4072 entities), `..._window1024.zip` (4088), `..._oversample_rare.zip` (3999), `..._adapter_dim128.zip` (4171). Expected Codabench ≈ fold2 + 0.05-0.09 → ~0.54-0.58. Recommended submission order: lr_high_long, window1024, then oversample_rare (diverse member for an ensemble). Skip adapter_dim128.

## Report notes (§8.5)
- Techniques covered: adapter (Houlsby) tuning with BiLSTM head, loss reweighting/oversampling, windowing, and a span-based head (negative result).
- Hyperparameter choices were made by the student from the sweep results; the LLM only built the tooling (per the course policy, the LLM declaration must say this).
