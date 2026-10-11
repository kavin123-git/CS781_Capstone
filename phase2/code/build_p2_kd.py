"""Builds the Phase-2 knowledge-distillation notebook: teacher = MoE (Houlsby adapters + BiLSTM head, p2_13, Codabench 0.5335),
student = the Phase-1 LSTM (84M, top LSTM layer + BiLSTM head fine-tuned, the best LSTM setup of the sweep)."""
import nbformat as nbf
import sys; sys.path.insert(0, "/home/claude/p2")
from build_p2_sweeps import *

MOE, LSTM = FAMILIES["moe_hc"], FAMILIES["lstm"]

KD_NAME = "p2_15_kd_moe13_to_lstm"

KD_SETTINGS = '''import json
# =====================================================================================================
# KNOWLEDGE DISTILLATION SETTINGS -- every value is YOUR choice (untuned placeholders)
# =====================================================================================================
RUN_NAME = "''' + KD_NAME + '''"
TOKENIZER_NAME = "a1bpe_tokenizer_160Mw_1ep_full_v40000_160000000w_seed42_minf1.json"

# TEACHER: the MoE (218.7M params, 111M active) with Houlsby adapters + BiLSTM head = run p2_13 (Codabench 0.5335).
#   For the fold1 -> fold2 experiment the teacher is p2_13's fold1-only model (it never saw fold2 labels);
#   for the final model the teacher is p2_13's fold1+fold2 refit.
TEACHER_RUN = "p2_13_moe_hc_adapter_bilstm_clsw"
TEACHER_CHECKPOINT = "''' + MOE["checkpoint"] + '''"
# STUDENT: the Phase-1 LSTM (83.9M params), top LSTM layer + new BiLSTM head trained = the best LSTM setup of the sweep
#   (partial_ft_top1_bilstm: fold2 0.4285, fold1 0.4425, Codabench 0.5008 WITHOUT distillation).
STUDENT_CHECKPOINT = "''' + LSTM["checkpoint"] + '''"
BASELINE_NO_KD = dict(fold2=0.4285, fold1=0.4425, codabench=0.5008)      # same recipe, hard labels only (sweep p2s_lstm84)
TEACHER_REF = dict(fold2=0.4562, codabench=0.5335)

PEFT_TARGETS = []
NORM_TYPE = "layernorm"; NORM_POSITION = "pre"
CONTEXT_LEN = 2048
MOE_BIAS_RATE_FT = 0.0
INPUT_MODE = "none"
ATTENTION = "causal"

# ---- distillation -------------------------------------------------------------------------------------
KD_ALPHA = 0.5               # loss = (1 - KD_ALPHA) * cross-entropy(gold tags) + KD_ALPHA * T^2 * KL(teacher || student)
KD_TEMPERATURE = 2.0         # softens both distributions; the KL is computed on the softened ones
KD_TRANSFER_UNLABELLED = True
# True  = ALSO distil on documents without gold labels: the teacher labels (softly) the text of the other fold and of the
#         test documents, and the student learns only from the teacher there (the test TEXT is used, never any test label).
#         This is where distillation can add information the gold labels do not carry.
# False = classic distillation: soft labels only on the documents that also have gold labels.
UNLABELLED_STRIDE = 2048     # unlabelled documents are cut into non-overlapping windows (half the cost of the labelled ones)

''' + BASE_DEFAULTS + '''
TEACHER_C = dict(BASE, TECHNIQUE="adapter", ADAPTER_TYPE="houlsby", ADAPTER_DIM=64, HEAD_TYPE="bilstm", LR=3e-4, EPOCHS=20, PATIENCE=5)
STUDENT_C = dict(BASE, TECHNIQUE="partial_ft", PARTIAL_TOP_BLOCKS=1, HEAD_TYPE="bilstm", LR=3e-4, EPOCHS=20, PATIENCE=5,
                 KD_ALPHA=KD_ALPHA, KD_TEMPERATURE=KD_TEMPERATURE)
QUICK_TEST = False           # True = tiny end-to-end check (2 docs per fold, 1 epoch, 512-token windows)
if QUICK_TEST:
    STUDENT_C.update(WINDOW=512, TRAIN_STRIDE=512, INFER_STRIDE=256, EPOCHS=1, PATIENCE=0)
    TEACHER_C.update(WINDOW=512, TRAIN_STRIDE=512, INFER_STRIDE=256)
    UNLABELLED_STRIDE = 512
    RUN_NAME += "_quicktest"
print(json.dumps(dict(student=STUDENT_C, teacher=TEACHER_C), default=str)[:600], "...")
'''

# the MoE (teacher) build functions get their own names so that they can live next to the LSTM (student) ones
TEACHER_BUILD = (MOE["build_src"].replace("PHASE1_STATE", "TEACHER_PHASE1_STATE").replace("build_phase1_lm", "build_teacher_lm")
                 .replace("make_backbone", "make_teacher_backbone").replace("phase1_sanity", "teacher_sanity"))
STUDENT_BUILD = LSTM["build_src"]

KD_MODELS = '''t0 = time.time()
TAG_NAMES = bio_tag_names(LABELS)
DOCS = {f: prepare_docs(bpe_tokenizer, TEXTS[f], GOLDS[f], LABELS, INPUT_MODE) for f in TEXTS}
TEST_DOCS = prepare_docs(bpe_tokenizer, TEST_TEXTS, None, LABELS, INPUT_MODE)
check_encode_identical(bpe_tokenizer, {**TEXTS["fold1"], **TEXTS["fold2"], **TEST_TEXTS})
print("token ids identical to BPETokenizer.encode on every document")
GOLD3 = {f: {d: [e[:3] for e in GOLDS[f][d]] for d in GOLDS[f]} for f in GOLDS}


def trim_tables(folds):
    tx, gd, sp = {}, {}, {}
    for f in folds:
        tx.update(TEXTS[f]); gd.update(GOLDS[f]); sp.update({d: DOCS[f][d]["spans"] for d in DOCS[f]})
    return learn_trim_table(tx, gd, sp, LABELS)


TRIM = {"fold1": trim_tables(["fold1"]), "fold2": trim_tables(["fold2"]), "both": trim_tables(["fold1", "fold2"])}
N_TOK = {f: sum(len(d["ids"]) for d in DOCS[f].values()) for f in DOCS}
print(len(TAG_NAMES), "tags;", N_TOK, "tokens;", round(time.time() - t0), "s")


def is_causal(C):
    return C["ATTENTION"] != "bidirectional" and C["HEAD_TYPE"] != "bilstm"


def load_phase1_state(name):
    ck = torch.load(fetch_model_file(name), map_location="cpu", mmap=True)
    return ck["model"] if "model" in ck else ck


def build_ner_from(build_lm, make_bb, C):
    lm = build_lm()
    bb = make_bb(lm, C)
    ner = NERModel(bb, bb.d_model, len(TAG_NAMES), C["HEAD_TYPE"], C["HEAD_DROPOUT"], C["HEAD_NORM"], C["HEAD_LSTM_HIDDEN"])
    info = apply_technique(ner, C)
    return ner.to(device), info


# ---------------- TEACHER ----------------
t0 = time.time()
TEACHER_PHASE1_STATE = load_phase1_state(TEACHER_CHECKPOINT)
teacher, teacher_info = build_ner_from(build_teacher_lm, make_teacher_backbone, TEACHER_C)
cv_state = fetch_model_file(TEACHER_RUN + "_cvmodel_state.pt")
load_trainable_state(teacher, torch.load(cv_state, map_location="cpu"))
teacher.eval()
for p in teacher.parameters():
    p.requires_grad_(False)
TEACHER_PARAMS = sum(p.numel() for p in teacher.parameters())
tp, t_secs = predict_docs(teacher, DOCS["fold2"], LABELS, TEACHER_C, device, is_causal(TEACHER_C), *TRIM["fold1"])
TS = strict_scores(GOLD3["fold2"], tp, LABELS)
TEACHER_FOLD2 = dict(macro_F1=round(TS["macro_F1"], 4), macro_P=round(TS["macro_P"], 4), macro_R=round(TS["macro_R"], 4),
                     inference_seconds=round(t_secs, 1), tokens_per_sec=round(N_TOK["fold2"] / t_secs, 1),
                     params=TEACHER_PARAMS, peak_mem_gb=round(torch.cuda.max_memory_allocated() / 1e9, 3) if device.type == "cuda" else None)
print("TEACHER (fold1 model) on fold2:", TEACHER_FOLD2, "| expected about", TEACHER_REF["fold2"], f"[{time.time()-t0:.0f}s]")
assert QUICK_TEST or TS["macro_F1"] > 0.40, "the teacher did not load correctly (its fold2 F1 should be about 0.456)"
if device.type == "cuda":
    torch.cuda.reset_peak_memory_stats()


def teacher_fn(ids, mask):
    with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=(device.type == "cuda")):
        return teacher(ids, mask)


# ---------------- STUDENT ----------------
STUDENT_PHASE1_STATE = load_phase1_state(STUDENT_CHECKPOINT)
PHASE1_STATE = STUDENT_PHASE1_STATE
_lm = build_phase1_lm()
STUDENT_PHASE1_PARAMS = sum(p.numel() for p in _lm.parameters())
print(f"student Phase-1 LSTM: {STUDENT_PHASE1_PARAMS:,} params; next-token loss on NER text {lm_sanity_loss(_lm, DOCS['fold1'], device):.3f} nats (expect about 4.0)")
del _lm; gc.collect()


def unlabelled(docs, prefix):
    """Documents the student sees ONLY through the teacher (no gold tags): tags are all IGNORE, flagged `unlabeled`."""
    return {prefix + ":" + k: dict(text=d["text"], ids=d["ids"], spans=d["spans"], tags=[IGNORE] * len(d["ids"]),
                                   unlabeled=True, train_stride=UNLABELLED_STRIDE) for k, d in docs.items()}
'''

KD_CV = '''# ---- distil on fold1 (+ unlabelled fold2 and test text), choose the epoch on fold2 ---------------------------------
student, student_info = build_ner_from(build_phase1_lm, make_backbone, STUDENT_C)
train_cv = dict(DOCS["fold1"])
if KD_TRANSFER_UNLABELLED:
    train_cv.update(unlabelled(DOCS["fold2"], "fold2"))      # fold2 TEXT only: its labels are used only to score
    train_cv.update(unlabelled(TEST_DOCS, "test"))
n_lab = len(window_items(DOCS["fold1"], STUDENT_C["WINDOW"], STUDENT_C["TRAIN_STRIDE"]))
n_all = len(window_items(train_cv, STUDENT_C["WINDOW"], STUDENT_C["TRAIN_STRIDE"]))
print(f"student windows/epoch: {n_lab} labelled + {n_all - n_lab} teacher-only")


def eval_fn(model):
    preds, _ = predict_docs(model, DOCS["fold2"], LABELS, STUDENT_C, device, is_causal(STUDENT_C), *TRIM["fold1"])
    s = strict_scores(GOLD3["fold2"], preds, LABELS)
    return s["macro_F1"], dict(heldout_P=round(s["macro_P"], 4), heldout_R=round(s["macro_R"], 4))


RESULTS = dict(run=RUN_NAME, student_phase1=STUDENT_CHECKPOINT, teacher=TEACHER_RUN, kd=dict(alpha=KD_ALPHA, temperature=KD_TEMPERATURE,
               transfer_unlabelled=KD_TRANSFER_UNLABELLED), student_settings=STUDENT_C, teacher_fold2=TEACHER_FOLD2,
               baseline_no_kd=BASELINE_NO_KD, technique_info=student_info,
               gpu=torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu")
cv = train_ner(student, train_cv, STUDENT_C, device, eval_fn=eval_fn, teacher=teacher_fn)
RESULTS["cv"] = {k: v for k, v in cv.items() if k != "best_state"}
load_trainable_state(student, cv["best_state"])
torch.save(cv["best_state"], f"{WORK}/{RUN_NAME}_cvmodel_state.pt")
'''

KD_REPORT = '''# ---- held-out report on fold2 (for the write-up) -------------------------------------------------------------------
preds, infer_s = predict_docs(student, DOCS["fold2"], LABELS, STUDENT_C, device, is_causal(STUDENT_C), *TRIM["fold1"])
S = strict_scores(GOLD3["fold2"], preds, LABELS)
json.dump(preds, open(f"{WORK}/{RUN_NAME}_heldout_fold2_pred.json", "w"))
H = dict(macro_F1=round(S["macro_F1"], 4), macro_P=round(S["macro_P"], 4), macro_R=round(S["macro_R"], 4),
         per_label={l: {k: (round(v, 4) if isinstance(v, float) else v) for k, v in d.items()} for l, d in S["per_label"].items()},
         errors=error_breakdown(GOLD3["fold2"], preds), label_confusions=confusion_pairs(GOLD3["fold2"], preds),
         analysis=detailed_analysis(GOLD3["fold2"], preds, TEXTS["fold2"]),
         inference_seconds=round(infer_s, 1), tokens_per_sec=round(N_TOK["fold2"] / infer_s, 1))
# how much of the teacher's output does the student reproduce, and how much of the student's gold-correct output is shared
gold_set = {d: set(map(tuple, GOLD3["fold2"][d])) for d in GOLD3["fold2"]}
sp = {d: set(map(tuple, preds[d])) for d in preds}; tpp = {d: set(map(tuple, tp[d])) for d in tp}
both = sum(len(sp[d] & tpp[d]) for d in sp)
H["teacher_student_agreement"] = dict(student_spans=sum(len(v) for v in sp.values()), teacher_spans=sum(len(v) for v in tpp.values()),
                                       identical_spans=both)
RESULTS["heldout"] = H
print("STUDENT (distilled) on fold2: strict macro-F1", H["macro_F1"], f"(P {H['macro_P']}, R {H['macro_R']})")
print("  same recipe WITHOUT distillation:", BASELINE_NO_KD["fold2"], "| teacher:", TEACHER_FOLD2["macro_F1"])
print("  change from distillation: %+.4f" % (H["macro_F1"] - BASELINE_NO_KD["fold2"]),
      "| share of the teacher's F1 retained: %.1f%%" % (100 * H["macro_F1"] / TEACHER_FOLD2["macro_F1"]))
print("  per-label P/R/F1:")
for l, d in H["per_label"].items():
    print(f"    {l:10s} {d['P']:.3f} {d['R']:.3f} {d['F1']:.3f}  (tp {d['tp']}, fp {d['fp']}, fn {d['fn']})")
print("  errors:", H["errors"])
print("  top label confusions:", dict(list(H["label_confusions"].items())[:10]))
print("  boundary error kinds:", H["analysis"]["boundary_error_kinds"])
print("  by region of the document:", H["analysis"]["by_region"])
print("  by document length:", H["analysis"]["by_doc_length"])
print("  teacher/student agreement on fold2 spans:", H["teacher_student_agreement"])
print("  epochs:", " | ".join(f"{h['epoch']}: loss {h['train_loss']}, F1 {h.get('heldout_macroF1')}" for h in cv["history"]))
EFF = dict(
    teacher=dict(params=TEACHER_PARAMS, params_active_note="111M of 218.7M active per token (MoE)", inference_tokens_per_sec=TEACHER_FOLD2["tokens_per_sec"],
                 peak_mem_gb=TEACHER_FOLD2["peak_mem_gb"]),
    student=dict(params=sum(p.numel() for p in student.parameters()), inference_tokens_per_sec=H["tokens_per_sec"],
                 peak_mem_gb=cv["peak_mem_gb"], trainable=student_info["params_trainable"], train_seconds=cv["train_seconds"]))
RESULTS["efficiency"] = EFF
print("EFFICIENCY:", json.dumps(EFF))
'''

KD_FINAL = '''# ---- final student: teacher = p2_13's fold1+fold2 refit; train on both folds' gold + teacher-only test text --------
load_trainable_state(teacher, torch.load(fetch_model_file(TEACHER_RUN + "_final_state.pt"), map_location="cpu"))
teacher.eval()
del student; gc.collect(); torch.cuda.empty_cache() if device.type == "cuda" else None
student, _ = build_ner_from(build_phase1_lm, make_backbone, STUDENT_C)
train_all = {**DOCS["fold1"], **DOCS["fold2"]}
if KD_TRANSFER_UNLABELLED:
    train_all.update(unlabelled(TEST_DOCS, "test"))
epochs = cv["best_epoch"]
fit = train_ner(student, train_all, dict(STUDENT_C, EPOCHS=epochs, PATIENCE=0), device, eval_fn=None, teacher=teacher_fn)
RESULTS["refit"] = {k: v for k, v in fit.items() if k != "best_state"}
torch.save(fit["best_state"], f"{WORK}/{RUN_NAME}_final_state.pt")
test_pred, test_s = predict_docs(student, TEST_DOCS, LABELS, STUDENT_C, device, is_causal(STUDENT_C), *TRIM["both"])


def validate_prediction(pred, texts, labels):
    assert set(pred) == set(texts), f"id mismatch: missing {len(set(texts)-set(pred))}, extra {len(set(pred)-set(texts))}"
    for did, ents in pred.items():
        seen = set()
        for x in ents:
            s, e, l = x
            assert type(s) is int and type(e) is int and 0 <= s < e <= len(texts[did]) and l in labels, (did, x)
            assert (s, e, l) not in seen, ("duplicate", did, x); seen.add((s, e, l))


validate_prediction(test_pred, TEST_TEXTS, LABELS)
zip_path = f"{WORK}/submission_{RUN_NAME}.zip"
with open(f"{WORK}/prediction.json", "w", encoding="utf-8") as f:
    json.dump(test_pred, f)
with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
    z.write(f"{WORK}/prediction.json", arcname="prediction.json")
with zipfile.ZipFile(zip_path) as z:
    assert z.namelist() == ["prediction.json"]
RESULTS["test"] = dict(docs=len(test_pred), entities=sum(len(v) for v in test_pred.values()), inference_seconds=round(test_s, 1),
                       per_label=dict(Counter(e[2] for v in test_pred.values() for e in v)))
print(f"wrote {zip_path}: {RESULTS['test']['entities']} entities over {len(test_pred)} documents (refit {epochs} epochs, {fit['train_seconds']:.0f}s)")
'''

KD_SAVE = '''RESULTS["notebook_minutes"] = round((time.time() - NB_START) / 60, 1)
res_path = f"{WORK}/{RUN_NAME}_results.json"
json.dump(RESULTS, open(res_path, "w"), indent=1, default=str)
for p in (res_path, f"{WORK}/{RUN_NAME}_heldout_fold2_pred.json", f"{WORK}/{RUN_NAME}_cvmodel_state.pt", f"{WORK}/{RUN_NAME}_final_state.pt"):
    upload_model_file(p)
print("=" * 100)
print(f"RESULT {RUN_NAME}: distilled student fold2 strict macro-F1 {RESULTS['heldout']['macro_F1']} | no-KD student {BASELINE_NO_KD['fold2']} | "
      f"teacher {TEACHER_FOLD2['macro_F1']} | student {EFF['student']['params']:,} params vs teacher {EFF['teacher']['params']:,} | "
      f"{RESULTS['notebook_minutes']} min")
print("per-label F1:", {l: d["F1"] for l, d in RESULTS["heldout"]["per_label"].items()})
print("=" * 100)
print("Submit:", f"submission_{RUN_NAME}.zip", "| send me this notebook (with outputs) or", os.path.basename(res_path))
SUMMARY_JSON = json.dumps(RESULTS, default=str)
print("SUMMARY_JSON_START"); print(SUMMARY_JSON); print("SUMMARY_JSON_END")
'''


def build_kd():
    cells = []
    md = lambda s: cells.append(nbf.v4.new_markdown_cell(s))
    code = lambda s: cells.append(nbf.v4.new_code_cell(s))
    md(f"""# CS-781 Phase 2 (L-NER) — `{KD_NAME}`: knowledge distillation, MoE teacher → LSTM student

**Teacher:** the Phase-1 MoE (218.7M params, 111M active) with Houlsby adapters + BiLSTM head — your run p2_13 (Codabench 0.5335).
**Student:** the Phase-1 LSTM (83.9M params), top LSTM layer + a new BiLSTM head trained — the best LSTM setup of the sweep, which reached 0.4285 / 0.4425 (fold2 / fold1) and 0.5008 on Codabench **without** distillation.

Loss = (1 − α) · cross-entropy(gold tags) + α · T² · KL(teacher ‖ student), computed on temperature-softened tag distributions.

What the notebook does (one Run All):
1. loads the data, tokenizer, both Phase-1 checkpoints and p2_13's saved adapter/head weights; **checks the teacher reproduces its fold2 score** (≈ 0.456);
2. trains the student on fold1 with the teacher (+ optionally teacher-only windows from the unlabelled text of fold2 and the test set — see `KD_TRANSFER_UNLABELLED`), scoring fold2 after every epoch;
3. reports fold2 F1 against the same student without distillation and against the teacher, per-label results, errors, teacher/student agreement, size / memory / speed;
4. retrains the student on fold1+fold2 with the teacher's fold1+fold2 refit and writes `submission_{KD_NAME}.zip`.

**Kaggle:** GPU T4, secrets `HF` and `HF_WRITE`. Expected ≈ 2.5–3.5 h (the teacher runs forward on every student window; the LSTM itself is the slow part). Needs the files `{{TEACHER_RUN}}_cvmodel_state.pt` and `_final_state.pt` in the bucket (written by p2_13).""")
    md("## 1. Settings")
    code(KD_SETTINGS)
    md("## 2. Setup: login, bucket")
    code(SETUP_CELL)
    md("## 3. Phase-2 data")
    code(DATA_CELL)
    md("## 4. Your Phase-1 BPE tokenizer (class unchanged), loaded from storage")
    code(BPE_SRC + TOK_LOAD)
    md("## 5. Character offsets, BIO tags, span decoding, boundary repair, strict scorer")
    code(ALIGN_SRC)
    md(f"## 6. TEACHER model code: {MOE['source_nb']}")
    code(MOE["pre"] + "\n" + MOE["model_src"])
    md("## 7. Teacher backbone wrapper")
    code(MOE["backbone_src"] + "\n\n" + TEACHER_BUILD)
    md(f"## 8. STUDENT model code: {LSTM['source_nb']}")
    code(LSTM["pre"] + "\n" + LSTM["model_src"])
    md("## 9. Student backbone wrapper")
    code(LSTM["backbone_src"] + "\n\n" + STUDENT_BUILD)
    md("## 10. NER engine: head, adaptation methods, training (with distillation), inference")
    code(CORE_SRC + "\n\nEVALUATOR_SRC = " + repr(EVAL_SRC))
    md("## 11. Tokenize, load the teacher (and verify it), load the student")
    code(KD_MODELS)
    md("## 12. Distil on fold1, select the epoch on fold2")
    code(KD_CV)
    md("## 13. Held-out report, comparison with the no-distillation student and the teacher, efficiency")
    code(KD_REPORT)
    md("## 14. Final student on fold1+fold2, test predictions, submission zip")
    code(KD_FINAL)
    md("## 15. Save and print the result block")
    code(KD_SAVE)
    nb = nbf.v4.new_notebook()
    nb["cells"] = cells
    nb["metadata"] = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                      "language_info": {"name": "python"}}
    nbf.validate(nb)
    path = OUT + KD_NAME + ".ipynb"
    nbf.write(nb, path)
    return path




def build_kd_variants():
    """Two notebooks: (a) distil on the labelled training folds only, (b) also on teacher-labelled unlabelled text (fold2 + test)."""
    import re
    base = build_kd()
    out = []
    for new, flag, note in [("p2_15a_kd_moe13_to_lstm_labelled_only", False,
                             "**Variant A:** the student sees only the labelled training folds (fold1, then fold1+fold2); the test text is never used for training."),
                            ("p2_15b_kd_moe13_to_lstm_plus_unlabelled", True,
                             "**Variant B:** additionally, the teacher labels the UNLABELLED text of fold2 (during model selection) and of the test set (final fit); no test annotations are used, but the test text is.")]:
        nb = nbf.read(base, as_version=4)
        for c in nb["cells"]:
            c["source"] = c["source"].replace(KD_NAME, new).replace("KD_TRANSFER_UNLABELLED = True", f"KD_TRANSFER_UNLABELLED = {flag}")
        nb["cells"][0]["source"] = nb["cells"][0]["source"].replace("\n\n**Teacher:**", f"\n\n{note}\n\n**Teacher:**", 1)
        assert any(f"KD_TRANSFER_UNLABELLED = {flag}" in c["source"] for c in nb["cells"])
        nbf.validate(nb)
        path = OUT + new + ".ipynb"
        nbf.write(nb, path)
        out.append(path)
    import os; os.remove(base)
    return out


if __name__ == "__main__":
    print(build_kd_variants())
