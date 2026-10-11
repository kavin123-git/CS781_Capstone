# Phase 2 (L-NER) — DETAILED METHODS of everything we built and tried (for the report's Methods section)

Companion to `phase2-MASTER-results-for-report.md` (results) and the per-experiment docs. This file describes WHAT each component does and WHY, in enough detail to be condensed later. Everything here is read from the code (`p2_core.py`, `lner_align.py`, `p2_backbone_*.py`, `build_p2_*.py`) as it ran. Wording is a draft by the assistant for the student to check and rewrite (course LLM-use policy).

Contents: 1 Task and data · 2 Pipeline overview · 3 Tokenization and character alignment · 4 BIO tagging, windows, merging · 5 Boundary repair (the learned trim table) · 6 Decoding · 7 Backbones for the four Phase-1 architectures · 8 Prediction heads · 9 Adaptation techniques (full FT, partial FT, BitFit, LoRA, LoRMA, Houlsby/Pfeiffer adapters, prefix, prompt) · 10 Training procedure and losses · 11 Experimental protocol and the sweep driver · 12 Knowledge distillation · 13 Ensembling · 14 Evaluation and error analysis · 15 Step-by-step walkthrough of the 0.5335 model (p2_13) · 16 Architecture diagram · 17 Engineering issues met and how they were handled · 18 Code map.

---
## 1. Task and data
- **Task (L-NER):** given the full text of an Indian court judgment, output all entity spans `[start, end, LABEL]` (character offsets, end exclusive) for 12 labels: APP (appellant/petitioner), RESP (respondent), A.COUNSEL (counsel for appellant), R.COUNSEL (counsel for respondent), JUDGE, WIT (witness), AUTH (authority), COURT, STAT (statute), PREC (precedent), DATE, CASENO.
- **Data:** fold1 (35 judgments, 4,120 entities), fold2 (35, 4,273), test fold3 (35, unlabelled; one document, id 1305957, is empty and must still appear with `[]`). Documents are long: median ≈ 4.5k words (≈ 5.2k tokens), max ≈ 20k words, 114k characters. Text has no newlines. Gold spans never overlap or nest, so a flat tag sequence can represent every gold entity.
- **Label counts** are very unbalanced (fold1/fold2): DATE 813/762, STAT 525/735, PREC 624/559 vs A.COUNSEL 70/125, WIT 67/120, R.COUNSEL 87/97. The metric is macro-F1, so a rare label weighs as much as DATE.
- **Metric (strict macro-F1):** a prediction counts only if (start, end, label) all match a gold triple. Per label: precision, recall, F1 from TP/FP/FN, then the unweighted mean of the 12 F1's. Duplicate (start, end, label) triples in a document make the official evaluator fail, so predictions are de-duplicated.
- **Why this is hard:** only 70 labelled documents, exact-boundary scoring, ~38% of entities end before attached punctuation (",", ".", ")") while the Phase-1 tokenizer glues punctuation to the previous word, and several labels are distinguished only by role words that appear to the right of the mention.

## 2. Pipeline overview
1. Encode each document with the Phase-1 A1 BPE tokenizer and keep, for every token, its character span.
2. Convert gold character spans to a BIO tag per token.
3. Cut long documents into overlapping windows of up to 2048 tokens (or 1024/512 in the sweeps).
4. Feed each window through a Phase-1 backbone (one of five: MoE, GQA, MoR, GDN encoder, LSTM) whose language-model output layer is removed; take the final hidden states.
5. Put a prediction head on top (linear, or BiLSTM + linear; optionally a span scorer) → per-token tag logits (25 tags).
6. Merge window predictions into one logit per token, decode a BIO-valid tag sequence with Viterbi, convert tags back to character spans.
7. Repair span edges at character level with a table learned from the training fold (boundary repair), de-duplicate, write `prediction.json`.
Adaptation of the backbone (which weights train) is the "technique"; the head is always trained.

## 3. Tokenization and character alignment (`lner_align.py`)
- **Tokenizer:** the Phase-1 "A1 BPE": 40,000-token vocabulary, byte-level symbols, whitespace-separated words, each word ending with an end-of-word marker `</w>`; punctuation is NOT split off, so "Ltd.," is part of one word. We do not change it. The file `a1bpe_tokenizer_160Mw_1ep_full_v40000_160000000w_seed42_minf1.json` is used for all models.
- **Offsets:** `encode_with_offsets` re-runs the tokenizer's own word-tokenisation and records, for every token, the (start_char, end_char) it covers (handling multi-byte UTF-8 characters split across tokens). `check_encode_identical` asserts on every document that the produced ids equal `BPETokenizer.encode(text)` exactly (it passed on all documents), and that spans are monotone and inside the text.
- **Input modes:** `none` = exact Phase-1 tokens (used for all submissions); `punct` splits every punctuation character into its own piece before BPE (tested once, run p2_05); `punct_digit` additionally separates letters from digits (not used).
- **Alignment problem measured:** with exact tokens, 2,828 of 8,393 gold entities (33.7%) start or end inside a token (e.g. gold "Ltd" but the token is "Ltd.</w>"); with `punct` only 60 do. This motivates boundary repair (§5).
- **Phase-1 sanity check:** before any training, each notebook computes the next-token loss of the loaded Phase-1 model on a few NER documents (MoE 3.106 nats, GQA 3.366, GDN enc-dec 3.513, LSTM 4.018, MoR 5.217; random weights would give ≈ 10.6). It proves the checkpoint loaded correctly and is later used to compare Phase-1 quality with NER quality.

## 4. BIO tagging, windows, merging
- **Tags:** 25 = O + (B, I) for each of 12 labels. B_label = 1 + 2·index, I_label = 2 + 2·index. A token is inside an entity if its character span overlaps the entity; the first overlapping token gets B, the following I. Tokens with an empty character span (a lone end marker) get the ignore label −100.
- **Windows:** documents are cut into windows of `WINDOW` tokens with a stride (`TRAIN_STRIDE`, `INFER_STRIDE`): default 2048/1024; the sweeps also use 1024/512 and 512/256. The last window is flush with the document end, so no tokens are dropped. A window is one training example; fold1 gives ≈ 182 windows per epoch at 2048/1024.
- **Merging at inference (`MERGE`):** each token is predicted in several overlapping windows. "context" (default) keeps, for each token, the window that gives it the most context: for a causal backbone the largest position in the window (most left context), for a bidirectional model the position furthest from both window edges. "average" averages the logits of all windows (tested in the decoding ablation, ±0.003 vs "context").
- **Header option (`HEADER`=H, `p2s_moe3`):** every window that does not start the document is prefixed with the document's first min(H, start) tokens (court, parties, case number), so each window knows which case it belongs to. Header positions carry no label and are stripped from the output. Needs WINDOW + H ≤ 2048 (the model's positions). Not yet tested on real data (first attempt crashed on this constraint).
- **Oversampling (`OVERSAMPLE`=k, `OVERSAMPLE_TAGS`):** every training window that contains a token of a weak label (APP, RESP, WIT in the sweep) is repeated k−1 extra times per epoch.

## 5. Boundary repair — the learned trim table (`learn_trim_table`, `repair_span`)
**Problem.** After decoding, a predicted span covers whole tokens. Because punctuation is glued to words, a span ending on the token "Ltd.," ends after the comma, but the gold usually excludes the comma. Without repair, such predictions are wrong under strict matching. Measured effect: +0.12 to +0.18 macro-F1 in every run (e.g. 0.245 → 0.369 on p2_01).

**What is learned.** Only from TRAINING documents, never from the fold being scored:
1. For each training document, build the "oracle" prediction: tag tokens from the gold spans and convert them back to character spans (`gold_to_token_tags` → `token_tags_to_spans`). This is the best a perfect tagger could do with whole tokens.
2. Match each oracle span to the single gold entity of the same label it contains.
3. Look at the characters at the end of the oracle span that lie beyond the gold end, going backwards while the character is trailing punctuation (set `, . ; : ) ] } ' " ’ ” ` - /`). Each time such a character is excluded by the gold, count `excluded[(label, char)] += 1`; when the character is part of the gold entity, count `kept[(label, char)] += 1` and stop. The same is done for leading punctuation (set `( [ { ' " ‘ “ ` - : . /`) at the start.
4. The rule is a majority vote per (label, character): `trim[(label, char)] = excluded > kept`.
Example: for COURT, a trailing "," is excluded by the gold nearly always, so the table says "trim ',' after COURT"; for A.COUNSEL after an abbreviation "." (e.g. "Adv.") the gold often keeps the period, so the table says "keep".

**How it is applied.** `repair_span` strips trailing/leading characters while the table says "trim" and the span stays non-empty; applied to every predicted span after decoding (`finalize_doc_entities`), then empty/invalid spans are dropped and exact duplicates removed. For the fold1→fold2 experiments the table is learned on fold1; for the final refit on fold1+fold2.

**Known limitation (error analysis).** The vote is per (label, character), not per context, so it cannot tell "Ltd." (keep) from "Ltd.," + sentence period. The most frequent remaining boundary error is a missing trailing "." after abbreviations ("Ltd.", "U.P.", "Anr.": 48–69 cases per model). A context-aware repair (keep "." after abbreviation-like tokens) was identified but not implemented in the models; the ensemble notebook has a separate abbreviation-period rule (§13).

## 6. Decoding (`p2_core.py`)
- **Viterbi under BIO constraints (default).** Log-softmax of the merged logits; dynamic programme over the 25 tags with a hard transition constraint "I-X may only follow B-X or I-X" and "a sequence cannot start with an I tag"; no learned transition scores. It always returns a valid tag sequence. It is far better than argmax (−0.04 to −0.08 when argmax is used instead), because independent per-token argmax produces broken spans (I tags without B, label flips inside an entity).
- **Argmax** (ablation only): independent best tag per token, then lenient conversion (an I tag that does not continue the same label starts a new entity).
- **Span decoding (`DECODE="span"`, ablation):** a `SpanScorer` head scores every candidate token span (length up to 32 tokens) as "no entity" or one of 12 labels (§8). Candidates with probability ≥ 0.5 are taken greedily by probability, skipping overlaps.
- **Finalisation:** token spans → character spans via the stored offsets; boundary repair; de-duplication; empty document → `[]`.

## 7. Backbones for the four Phase-1 architectures
All Phase-1 model code is used unchanged; a thin wrapper exposes the hidden states that feed the head and removes the vocabulary projection (the final LayerNorm is kept where the model has one). Windows longer than the model's positions are refused (assertion).
- **MoE decoder (main model, 218.7M parameters, ≈ 111M active per token):** decoder-only transformer with grouped-query attention, 8 blocks, d=768, mixture-of-experts feed-forward, Muon-trained, final LayerNorm, continued pre-training on high-court text. Routing is kept fixed during fine-tuning (`MOE_BIAS_RATE_FT = 0.0`). Wrapper `DecoderBackbone`. Can optionally remove the causal mask (ablation p2_04) or add prefix keys/values.
- **GQA 160M (111.1M):** same code as MoE with a dense feed-forward; `DecoderBackbone`.
- **MoR 160M (Mixture-of-Recursions, 120.8M):** decoder with recursion depth chosen by routing. Wrapper `ForwardBackbone` calls the model's own forward with the vocabulary layer replaced by the identity. **Routing must be frozen**: the Phase-1 code updates routing thresholds (`tau`) inside forward() during training; fine-tuning on 35 documents under fp16 made them diverge to NaN (first run: loss NaN, F1 0). The fix freezes `tau` at the Phase-1 values (same routing rule) and skips non-finite batches; all later runs trained stably.
- **GDN encoder-decoder (144.3M):** Gated-DeltaNet encoder-decoder; only the **encoder** (6 bidirectional blocks, each with a forward and a backward GDN mixer, plus a final norm) is used; decoder and output layer are dropped (`GDNEncoderBackbone`). Window 1024/512 because GDN's recurrent scan is ~2× slower per token on a T4. The encoder was pre-trained only to feed a decoder.
- **LSTM 84M (lightweight, 83.9M):** embeddings 768-d, 3 LSTM layers of 1536 hidden units. Wrapper `LSTMBackbone` splits the 3-layer LSTM into three single-layer LSTMs with the weights copied exactly, so adapters can be attached between layers and the top layer fine-tuned alone. The output of the top layer (1536-d) feeds the head. LoRA/LoRMA/prefix do not apply (no attention projections).
- **Canon-layers model:** wrapper exists (`ForwardBackbone`) but no results are recorded.

## 8. Prediction heads
- **Linear head:** dropout(0.1) → Linear(d, 25) applied to each token's hidden state.
- **BiLSTM head (the single most effective component):** dropout → bidirectional LSTM (1 layer, 256 units per direction, padding removed with packed sequences so the backward direction never reads pad positions; run in fp32) → Linear(512, 25). Rationale: the backbones are causal (except GDN), so a token's state knows only what is to its left, yet many labels are decided by words to the right ("X … Appellant", "Mr. Y, learned counsel for the respondent", "Z (PW-1)"). The BiLSTM gives every token context on both sides inside the window without touching the pretrained layers (removing the causal mask inside the MoE hurt: −0.036 fold2). It also adds sequential smoothing of neighbouring tag decisions. Costs ≈ +2.1M parameters, 25–35% lower throughput, +0.2 GB. Variants tried: hidden 512 (no gain), 2 layers (helps fold2 only).
- **Span head (`SPAN_DIM`=128, sweep ablation):** for each start token i and length k (< 32): score = MLP(start-projection(h_i) + end-projection(h_{i+k}) + length-embedding(k)) → 13 classes (none + 12 labels). Trained jointly with the token tagger: cross-entropy over all gold spans plus the hardest negatives (20× as many, at least 256), weight `SPAN_WEIGHT`. Result: precision 0.56 but recall 0.31; worse than the token tagger (0.377 vs 0.454).

## 9. Adaptation techniques
Number of trainable parameters is reported by each notebook; "head" parameters (linear, BiLSTM, span) always train.
- **Full fine-tuning:** all backbone weights train (LR 3e-5). 59–86% of parameters; highest memory (5.4 GB on MoE); memorises 35 documents (train loss → 0.001–0.003, held-out F1 flat after ~5–7 epochs).
- **Partial fine-tuning:** only the top k blocks (k=2 for transformers/MoR/GDN; k=1 for the LSTM = its top layer) plus the final norm train. "Frozen" = k=0: the whole Phase-1 model is frozen and only the head (and final norm) learns — the control that shows how much the backbone adapts.
- **BitFit:** only bias vectors train (all `.bias` of Linear/LayerNorm and LSTM `bias_ih/bias_hh`); 0.1–2% of parameters.
- **LoRA (Hu et al. 2022):** for selected weight matrices W0 (attention q, k, v, out projections in every block) add a trainable low-rank update: y = W0 x + b + (α/r)·B A x, A (r×in) random, B (out×r) zero, so training starts exactly at the Phase-1 function. r=16, α=32. The Phase-1 weights stay frozen.
- **LoRMA (Bihany, Patel, Modi, ACL Findings 2025; compulsory technique):** a **multiplicative** low-rank update, W' = (BA)·W0 instead of W0 + BA, i.e. the update acts on the output of W0 (the matrices A, B are on the output side: A is r×out, B is out×r). Two variants, both implemented:
  - **LoRMA+ ("plus", additive rank inflation):** W' = ((α/r)·B A + I) W0. Starts exactly at W0 (B=0). Computed as y0 = W0 x; y = y0 + (α/r)·B(A y0), so the out×out matrix is never formed (same cost as LoRA).
  - **LoRMA_π ("pi", permutation-based inflation):** W' = I_π(BA) W0, where I_π rotates row i of BA cyclically by i positions, inflating the rank of the update beyond r. Initialised so I_π(BA)=I (B's first column all ones, A has a single 1 at [0,0]). Forms the out×out matrix every step (O(d²(r+batch))), so it is the more expensive variant.
  Same targets, rank and alpha as LoRA: LoRMA trains 22% fewer parameters than LoRA on the MoE (511K vs 658K) because A is r×out rather than r×in (smaller for the grouped-query k/v projections).
- **Houlsby adapters (Houlsby et al. 2019):** small bottleneck MLPs inserted into the frozen network. Each adapter takes the output h of a sub-layer, projects it down to a small dimension, applies a non-linearity and projects back up, then adds it to the input: h' = h + W_up·GELU(W_down·h). W_up is initialised to zero, so the network starts as the unchanged Phase-1 model. In the original Houlsby design adapters sit after **both** the attention sub-layer and the feed-forward sub-layer of every block; here: bottleneck 64 (128 in one sweep setup), d=768, so one adapter = 2·768·64 + 64 + 768 ≈ 99K parameters; MoE has 8 blocks × 2 = 16 adapters = 1.59M parameters. They are attached with forward hooks on `attn` and `ff` modules, so the Phase-1 code is untouched. Adapters need roughly twice the epochs of full fine-tuning (best epochs 16–20 of 20) and halve peak memory.
- **Pfeiffer adapters (Pfeiffer et al. 2021):** one adapter per block, after the feed-forward sub-layer only (≈ 21% fewer parameters). For the LSTM: Houlsby = one adapter after each of the 3 layers, Pfeiffer = after the top layer only. For the GDN encoder: Houlsby adapters after the forward mixer, backward mixer and feed-forward of each block.
- **Prefix tuning (Li & Liang 2021), MoE only:** 20 learnable key and value vectors per layer (per KV head) are prepended to every attention layer's keys/values (never rotated; always visible). Implemented directly (no MLP re-parameterisation). Failed (F1 0.09): train loss stuck at ≈ 0.47, entities predicted everywhere. Not available for MoR/GDN/LSTM wrappers.
- **Prompt tuning (Lester et al. 2021):** code exists (20 learned input embeddings prepended, initialised from random vocabulary embeddings) but no run was done.
- **Knowledge distillation:** §12.

## 10. Training procedure and losses (`train_ner`)
- **Optimizer:** AdamW (β=(0.9, 0.98), weight decay 0.01), two learning-rate groups: head 1e-3 (`HEAD_LR`) and everything else (`LR`: 3e-5 full FT, 3e-4 PEFT, 1e-3 BitFit, `lr_high_long` 1e-3). Linear warm-up for 6% of steps, then linear decay to 0. Gradient clipping 1.0. fp16 autocast with a gradient scaler. Batch 2 windows × 4 accumulation steps (effective 8); windows of 1024 use 4 × 2.
- **Loss:** token-level cross-entropy over the 25 tags, ignoring −100 positions. **Class weights `inv_sqrt`:** weight of tag t = sqrt(total_tokens / count_t), normalised so that O has weight 1 (rare B/I tags get larger weights). `inv` (total/count) was tried and was harmful (over-tagging, precision 0.21). Optional **label smoothing** (0.1) and **focal modulation** (γ=2; loss × (1−p_t)^γ) were tried in the MoE sweep and hurt.
- **Epoch selection:** after every epoch the model is scored (strict macro-F1, full decoding pipeline) on the held-out fold; the trainable state of the best epoch is kept; early stop after `PATIENCE` epochs without improvement. Non-finite batches are skipped (never back-propagated).
- **What is stored:** only the trainable parameters (adapter/LoRMA/LoRA matrices, head, and any updated routing buffers) — a few MB per model — plus the learned trim table.
- **Cost bookkeeping:** every run records trainable parameters (and % of the Phase-1 model), peak GPU memory, training seconds, tokens/second and inference time.

## 11. Experimental protocol and the sweep driver
- **Cross-validation protocol:** (1) train on fold1, score on fold2 after each epoch, pick the best epoch (this is the "fold2 F1"); (2) for the best setups, train on fold2 and score on fold1 (confirmation; the two-fold mean is more reliable than one 35-document fold); (3) **refit** on fold1+fold2 for the chosen number of epochs (best epoch of step 1) and predict the test set (fold3) → `prediction.json` zipped as the Codabench submission. Boundary-repair tables are learned on the training folds of each step.
- **Why not more folds:** only two labelled folds exist; differences below ≈ 0.015 are noise (seed noise 0.003; fold-to-fold differences of the same setup 0.003–0.02).
- **Sweep notebooks (`build_p2_sweeps.py`):** one Run-All notebook per architecture. Stage 1 runs every setup fold1→fold2 and saves its full report (per-label F1, error breakdown, boundary-error kinds, header-vs-body and document-length analysis, parameters, memory, time). Stage 2 re-runs the best setups fold2→fold1 and ranks by the mean. Stage 3 refits the top K on fold1+fold2 and writes `submission_<sweep>_<setup>.zip`. Results of every finished setup are uploaded to a model bucket immediately, so a stopped session resumes by Run All (finished setups are skipped). A setup that fails (e.g. window overflow) is logged and the sweep continues.
- **First MoE series (p2_01–p2_14)** (`build_p2_notebooks.py`): hand-chosen runs: 01 full FT; 02 LoRMA; 03 LoRA; 04 full FT without causal mask; 05 full FT with punctuation split; 06 full FT + class weights; 07/08 LoRMA/LoRA + class weights; 09 Houlsby adapters + class weights; 10 BitFit; 11 prefix tuning; 12/13/14 full FT / adapters / LoRMA + BiLSTM head + class weights.
- **Architecture sweeps:** GQA (12 setups), MoR (12), GDN encoder (12), LSTM (8); **MoE sweep 2** (`p2s_moe2`, 13 setups, one change at a time from the p2_13 recipe, with seed-repeat and base reference runs and a decoding ablation); **MoE sweep 3** (`p2s_moe3`, 6 setups built on the stacked winners; pending).

## 12. Knowledge distillation (`build_p2_kd.py`)
- **Roles:** teacher = p2_13 (MoE + Houlsby + BiLSTM, 191.6M parameters as counted, fold2 0.456, Codabench 0.5335); student = Phase-1 LSTM 84M with top layer + BiLSTM head trained (22.6M trainable; the best LSTM setup, no-KD Codabench 0.5008).
- **Loss:** (1−α)·CE(gold tags) + α·T²·KL(softmax(teacher/T) ‖ softmax(student/T)) with α=0.5, T=2, computed per token on the 25-tag distributions, 20 epochs. The teacher's weights are frozen and it runs forward on every student window (`teacher_fn`).
- **For the fold1→fold2 experiment** the teacher is p2_13's fold1-only model (never saw fold2 labels); **for the final model** it is p2_13's fold1+fold2 refit, so the held-out estimate simulates the real situation.
- **Variant A (labelled only):** KD only on the labelled training windows (the CE term uses gold tags; the KL term uses the teacher on the same tokens).
- **Variant B (+ unlabelled text):** additionally, "teacher-only" windows are built from texts without labels — fold2 text during model selection and the test text for the final model; there the CE term is skipped (all labels ignored) and every real token receives only the KL term. This is semi-supervised / transductive distillation: no test labels are used, but the test inputs are. 241 teacher-only windows per epoch were added to the 182 labelled ones.
- **Reported:** F1 vs the no-KD student and the teacher, the share of teacher F1 retained, per-label results, teacher–student span agreement, parameters, memory and speed.

## 13. Ensembling (`build_p2_ensemble.py`; hard span voting on prediction files)
The TA stated ensembles are allowed provided all four architectures meet the requirements. The notebooks need no model weights: fold2 predictions of the fold1-trained models (from the bucket) are used to tune the ensemble, and the test predictions are the `prediction.json` files of the already-submitted refits (Kaggle dataset `phase2` of zips).
- **Voting:** each member's predicted spans are counted per document; a span is kept if its (weighted) vote share ≥ τ (τ grid 0.15–0.80). Two modes: *exact* votes on (start, end, label); *boundary* votes on (start, end) and takes the most-voted label. Weights: equal, or proportional to the member's fold2 F1. Selected spans are ordered by votes (ties: longer, earlier) and accepted greedily if they do not overlap an already-accepted span (gold never overlaps).
- **Abbreviation-period rule (learned on fold1 only):** per (label, last word) and per (label, word shape) it counts whether gold entities include a final "." or stop before one, and then adds the period to member predictions where the rule says so, before voting.
- **Search and honest estimate:** the grid (member set × mode × weights × τ × dot-rule) is scored on fold2; to avoid reporting an optimistic number, the best configuration is chosen on one half of the fold2 documents and scored on the other half (and vice versa); this "honest" F1 is the number to trust. Member groups tried: top-N by fold2 F1, MoE-only (16a), one-per-model / two-per-model / no-LSTM-family (16b), MoE + KD student (16c).
- **Limitation:** hard voting cannot combine probabilities; soft voting (average the per-token probabilities of the models and Viterbi-decode) needs the saved weights and was not built.

## 14. Evaluation and error analysis (`strict_scores`, `error_breakdown`, `detailed_analysis`)
- **Strict scores** replicate the official evaluator's arithmetic (per label TP/FP/FN, macro-F1, macro-P, macro-R) and are used for epoch selection and every table.
- **Error breakdown** classifies every prediction: *correct*; *label error* (same span, wrong label); *boundary error* (overlaps a gold span of the same label with different edges); *boundary + label error*; *spurious* (overlaps no gold span); and gold entities *missed entirely*.
- **Boundary-error kinds:** prediction too long / too short / shifted, how many differ only by punctuation or whitespace, and the most frequent differing strings (e.g. a missing trailing ".").
- **Where errors occur:** by region of the document (first 5% = cause-title header, 5–50%, 50–100%), and by document length (shortest/middle/longest third, micro-F1); label-confusion pairs (e.g. A.COUNSEL→R.COUNSEL).
- **Decoding ablation:** the same trained model decoded with Viterbi, argmax, average merging and (if present) span decoding.
- **Phase-1 vs Phase-2:** Phase-1 BERTScore and the Phase-1 next-token loss on NER text are compared with L-NER scores to study transfer of the Phase-1 representations.

## 15. Step-by-step walkthrough of the 0.5335 model (p2_13: MoE + Houlsby adapters + BiLSTM head + class weights)
1. **Load.** The Phase-1 MoE checkpoint (218.7M parameters, high-court continuation) is loaded; the vocabulary projection is dropped, the final LayerNorm kept; every Phase-1 weight is frozen. Sanity check: next-token loss 3.106 nats on NER text.
2. **Tokenize.** Each judgment → A1 BPE token ids + character offsets (ids identical to the Phase-1 encoder).
3. **Label.** Gold character spans → BIO tag per token (25 tags); 2,828 gold edges fall inside tokens (repaired later).
4. **Windows.** Each document is cut into 2048-token windows, stride 1024 (182 windows per epoch from fold1).
5. **Adapters.** 16 Houlsby adapters (768→64→768, zero-initialised up-projection) are hooked after the attention and feed-forward of each of the 8 blocks. At the start the network is exactly the Phase-1 MoE.
6. **Head.** The final hidden states (768-d) pass dropout → BiLSTM (256 per direction) → Linear(512→25). Trainable: adapters 1.59M + BiLSTM 2.10M + linear 0.01M = 3.70M parameters (1.69% of the model); peak GPU memory 2.5 GB.
7. **Train.** Class-weighted cross-entropy (inv_sqrt), AdamW, head LR 1e-3, adapter LR 3e-4, 20 epochs max, patience 5, fp16. After each epoch fold2 is scored with the full pipeline; best epoch 16 (fold2 F1 0.456; P 0.492, R 0.464).
8. **Predict.** Each fold3 document: overlapping windows → logits → keep for each token the window with most left context → Viterbi (valid BIO sequence) → character spans.
9. **Repair.** Learned trim table removes trailing/leading punctuation that gold usually excludes; de-duplicate; empty doc → `[]`.
10. **Refit and submit.** Retrain from the Phase-1 weights on fold1+fold2 for 16 epochs with the trim table learned on both folds → predict fold3 → `prediction.json` → Codabench **0.5335**.
Why it works: the frozen Phase-1 features already contain most of what is needed; the BiLSTM supplies right-hand context and a flexible output layer; adapters adapt the features at 1.7% of the parameter cost; class weights make rare labels count; Viterbi + trim repair fix span structure and edges.

## 16. Architecture diagram (text version; a picture was generated earlier in the session)
```
character text ──► A1 BPE (40k, whitespace words) ──► token ids + char offsets
                                         │
                       sliding windows (2048 / stride 1024) [+ optional header tokens]
                                         ▼
  ┌──────────────── Phase-1 backbone (frozen; MoE shown) ────────────────┐
  │ embedding                                                            │
  │ 8 × [ GQA attention ─► ⟨Houlsby adapter 768→64→768⟩ ─►               │
  │       MoE feed-forward ─► ⟨Houlsby adapter⟩ ]   (LoRA/LoRMA instead: │
  │                                  low-rank updates on q,k,v,out)      │
  │ final LayerNorm   (vocabulary projection removed)                    │
  └───────────────────────────────┬──────────────────────────────────────┘
                                  ▼  hidden states (768-d per token)
                     dropout ─► BiLSTM(256 ×2 directions) ─► Linear(512 → 25 tags)
                                  ▼
          merge overlapping windows ─► Viterbi (BIO-valid) ─► token spans
                                  ▼
        learned boundary repair (trim table) ─► de-duplicate ─► prediction.json
```
(⟨ ⟩ = trainable; everything else in the backbone is frozen. Head and adapters ≈ 3.7M trainable parameters.)

## 17. Engineering issues met and how they were handled (useful for the "challenges" paragraph)
- **Tokenizer location:** the tokenizer file lives only in the old model repo; notebooks fall back to it. Hugging Face tokens are used only through Kaggle secrets.
- **MoR NaN:** routing thresholds updated during training → fp16 overflow → NaN; fixed by freezing routing and skipping non-finite batches.
- **Window + header overflow:** HEADER=256 with WINDOW=2048 exceeded the model's 2048 positions (two sweep setups crashed after hours); fixed (WINDOW=1792 / 1024) and the settings are now validated before training starts. The quick test did not catch it because it uses a 512 window.
- **Kaggle runtime limits:** the LSTM sweep exceeded 12 h; sweeps are resumable (results cached in the bucket) and `MAX_SWEEP_HOURS` stops new setups early.
- **Ensemble file handling:** Codabench truncates long zip names (~35 characters) → prefix matching; Kaggle auto-unzips uploaded zips into folders named after the zip → names are taken from the folder; both layouts accepted.
- **Models not saved:** several early models' weights were not saved, which is why the ensemble works on prediction files only.
- **Submission budget:** document says 40 total / 10 per day, page shows 80 / 20; 40 assumed.

## 18. Code map (all in the Project under `claude/code/`)
| File | Content |
|---|---|
| `phase2_lner_align.py` (local `lner_align.py`) | offsets, BIO, trim table, repair, strict scoring, error breakdown, windows |
| `p2_core.py` | NER model, heads (linear/BiLSTM/span), LoRA, LoRMA, adapters, techniques, training, KD loss, decoding, error analysis |
| `p2_backbone_decoder.py` | GQA/MoE wrapper (prefix, bidirectional option, adapter sites) |
| `p2_backbone_generic.py` | MoR/canon, LSTM and GDN-encoder wrappers |
| `build_p2_notebooks.py` | notebook builder for p2_01–p2_14 and MoR routing freeze |
| `build_p2_sweeps.py` | per-architecture sweep notebooks |
| `build_p2_moe_sweep.py`, `build_p2_moe_sweep2.py` | MoE sweeps 2 and 3 |
| `build_p2_kd.py` | KD notebooks 15a/15b |
| `build_p2_ensemble.py` | ensemble notebooks 16a/16b/16c |
