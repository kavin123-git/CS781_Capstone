# Phase 2 (L-NER) — Knowledge-distillation experiment (p2_15a / p2_15b), results of 2026-10-10

**Setup (`build_p2_kd.py`).** Teacher: p2_13 (Phase-1 MoE high-court + Houlsby adapters + BiLSTM head; fold2 0.4562, Codabench 0.5335; 191.6M params in the NER model as counted by the notebook, 111M active per token of the 218.7M MoE). Student: Phase-1 LSTM 84M (83.9M; 86.3M with the head), top LSTM layer + BiLSTM head trained (22.6M trainable) — the best LSTM setup of the sweep (no-KD fold2 0.4285 / fold1 0.4425, Codabench 0.5008). Loss = 0.5·CE(gold) + 0.5·T²·KL(teacher‖student), T = 2, 20 epochs, epoch chosen on fold2, refit on fold1+fold2 with the teacher's fold1+fold2 refit.
- **A (labelled only)**: student trains only on labelled windows (182/epoch).
- **B (+ unlabelled text)**: additionally 241 teacher-only windows per epoch from the UNLABELLED text of fold2 (during model selection) and of the test set (final refit); the teacher's soft tags are the only supervision there; no test annotations are used, but the test text is. (Uses the test inputs → transductive; must be disclosed in the report.)

| Run | Fold2 F1 | P | R | vs no-KD student 0.4285 | share of teacher F1 retained | best ep | train time | test entities (refit) | **Codabench** | Codabench − fold2 |
|---|---|---|---|---|---|---|---|---|---|---|
| teacher p2_13 (fold1 model) | 0.4562 | 0.4916 | 0.4644 | – | 100% | – | – | – | 0.5335 | +0.077 |
| student, no KD (LSTM partial_ft_top1_bilstm) | 0.4285 | 0.423 | 0.453 | 0 | 93.9% | 18 | 2620 s | 4461 | 0.5008 | +0.072 |
| **A** KD, labelled only | 0.4245 | 0.4295 | 0.4435 | −0.004 | 93.1% | 20 (last) | 2958 s (118 min total) | 4342 | **0.5109** | +0.086 |
| **B** KD + unlabelled | **0.4745** | 0.5157 | 0.4756 | **+0.046** | 104.0% | 17 | 5537 s (170 min total) | 4122 | **0.5458** | +0.071 |

Per-label fold2 F1 (A / B): APP .209/.276, RESP .090/.183, A.COUNSEL .426/.464, R.COUNSEL .495/.500, JUDGE .408/.515, WIT .135/.114, AUTH .374/.414, COURT .760/.780, STAT .610/.691, PREC .242/.336, DATE .905/.928, CASENO .440/.495.
Errors fold2 (A / B): correct 2316/2500, spurious 993/768, boundary 759/637, label 225/167, boundary+label 138/82, missed 858/875. Teacher–student identical spans: A 2760 of 4431 student / 4326 teacher spans; B 3313 of 4154 / 4326 (B's student copies the teacher markedly more closely).

**Efficiency (T4).** Student has 2.2× fewer parameters than the teacher (86.3M vs 191.6M NER-model params; 111M active in the teacher), but is NOT faster or lighter at inference: 4.1–4.3k tokens/s vs 11.4–12.1k for the teacher (the LSTM is sequential), peak memory 1.80 GB vs 0.97 GB (LSTM activations over a 2048-token window, 22.6M trainable parameters). So distillation compressed the parameter count, but on this hardware the compressed model is ~2.8× slower and ~1.9× heavier in peak memory. Training time of the KD runs is 1.1–2.1× the no-KD run (teacher forward on every window; B has 2.3× the windows).

**Findings.**
- **K1. KD on the labelled data alone gives a small gain on Codabench and none on fold2**: fold2 −0.004 (within the ±0.015 noise of 35 documents), Codabench +0.010 (0.5109 vs 0.5008). The student retains 93% of the teacher's fold2 F1 and 95.8% of its Codabench score (0.5109 / 0.5335).
- **K2. The big gain of B comes from extra input text, not from softer labels**: B differs from A only by teacher-labelled windows of documents the student has not seen labelled (fold2 text; test text for the final model). Codabench: **0.5458 — a new best single model, +0.0123 above its own teacher (0.5335) and +0.045 above the no-KD student**. It improves every label on fold2 except WIT, mainly via precision (0.516 vs 0.43; spurious spans 993 → 768) — the student learns the teacher's *consistent* behaviour on the unseen text and averages out the teacher's noise (self-training / ensembling effect). The fold2 estimate (0.4745, 104% of teacher) was a fair simulation of the test-time situation: Codabench confirmed the direction and the size (+0.071 over fold2, in line with the other models' +0.05–0.09).
- **K3. Codabench−fold2 is not a constant**: A +0.086, B +0.071, teacher +0.077, no-KD student +0.072 — differences of ±0.01 are noise.
- **K4. Efficiency conclusion for §8.5**: KD preserved ≥ 93% of the teacher's F1 with 2.2× fewer parameters, but it did not give faster inference or lower memory on a GPU because of the recurrent architecture; a Transformer student (e.g. GQA 111M) would be the efficiency-oriented choice. In absolute terms B is both smaller (86M vs 192M) and more accurate than the teacher on the test set.
- **K5. Caveat for the report**: B uses the test documents' text (never labels) for pseudo-labelled training (transductive); disclose it. A is the clean KD result.
