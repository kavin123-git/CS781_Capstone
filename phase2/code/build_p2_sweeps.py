"""Builds Phase-2 SWEEP notebooks: one notebook per Phase-1 architecture that tries a list of setups on fold1->fold2,
optionally re-checks the best few on fold2->fold1, and writes submission zips only for the top K."""
import nbformat as nbf
exec(open("/home/claude/p2/build_p2_notebooks.py").read().split("if __name__")[0])

# defaults shared by every setup (each SWEEP entry overrides some of these)
BASE_DEFAULTS = '''BASE = dict(
    TECHNIQUE="full_ft", RANK=16, ALPHA=32, LORMA_VARIANT="plus", LORA_DROPOUT=0.0, PEFT_TARGETS=PEFT_TARGETS,
    PARTIAL_TOP_BLOCKS=2, ADAPTER_TYPE="houlsby", ADAPTER_DIM=64, PROMPT_LEN=20, PREFIX_LEN=20,
    ATTENTION="causal", HEAD_TYPE="linear", HEAD_LSTM_HIDDEN=256, HEAD_DROPOUT=0.1, HEAD_NORM=False,
    CLASS_WEIGHTING="inv_sqrt",          # the only ablation that helped on both fold2 and Codabench for the MoE (p2_06)
    DECODE="viterbi", MERGE="context", BOUNDARY_REPAIR=True, INPUT_MODE=INPUT_MODE,
    WINDOW=2048, TRAIN_STRIDE=1024, INFER_STRIDE=1024, BATCH=2, ACCUM=4, EPOCHS=10, PATIENCE=3,
    LR=3e-5, HEAD_LR=1e-3, WEIGHT_DECAY=0.01, WARMUP_FRAC=0.06, GRAD_CLIP=1.0, AMP="fp16", SEED=0, INFER_BATCH=4,
    KD_ALPHA=0.0, KD_TEMPERATURE=2.0)
'''

DEFAULT_SWEEP = '''SWEEP = [   # (name, settings that differ from BASE). Order = run order; edit freely. All values are YOUR choices.
    # BiLSTM head (best head on the MoE: +0.05-0.09 Codabench for every technique) x techniques
    ("full_ft_bilstm",         dict(TECHNIQUE="full_ft", HEAD_TYPE="bilstm", LR=3e-5, EPOCHS=10, PATIENCE=3)),
    ("adapter_bilstm",         dict(TECHNIQUE="adapter", HEAD_TYPE="bilstm", LR=3e-4, EPOCHS=20, PATIENCE=5)),
    ("lorma_bilstm",           dict(TECHNIQUE="lorma", HEAD_TYPE="bilstm", LR=3e-4, EPOCHS=20, PATIENCE=5)),
    ("lora_bilstm",            dict(TECHNIQUE="lora", HEAD_TYPE="bilstm", LR=3e-4, EPOCHS=20, PATIENCE=5)),
    ("bitfit_bilstm",          dict(TECHNIQUE="bitfit", HEAD_TYPE="bilstm", LR=1e-3, EPOCHS=20, PATIENCE=5)),
    ("partial_ft_top2_bilstm", dict(TECHNIQUE="partial_ft", PARTIAL_TOP_BLOCKS=2, HEAD_TYPE="bilstm", LR=1e-4, EPOCHS=15, PATIENCE=4)),
    ("lorma_pi_bilstm",        dict(TECHNIQUE="lorma", LORMA_VARIANT="pi", HEAD_TYPE="bilstm", LR=3e-4, EPOCHS=20, PATIENCE=5)),
    ("adapter_pfeiffer_bilstm",dict(TECHNIQUE="adapter", ADAPTER_TYPE="pfeiffer", HEAD_TYPE="bilstm", LR=3e-4, EPOCHS=20, PATIENCE=5)),
    # linear-head references (show the head effect on this architecture)
    ("full_ft",                dict(TECHNIQUE="full_ft", LR=3e-5, EPOCHS=10, PATIENCE=3)),
    ("adapter",                dict(TECHNIQUE="adapter", LR=3e-4, EPOCHS=20, PATIENCE=5)),
    ("lorma",                  dict(TECHNIQUE="lorma", LR=3e-4, EPOCHS=20, PATIENCE=5)),
    ("full_ft_bilstm_clsw_inv",dict(TECHNIQUE="full_ft", HEAD_TYPE="bilstm", CLASS_WEIGHTING="inv", LR=3e-5, EPOCHS=10, PATIENCE=3)),
]
'''


FAMILY_SWEEPS = {
    "lstm": ('''SWEEP = [   # (name, settings that differ from BASE). LoRA / LoRMA / prefix do not apply to an LSTM (no linear layers in the recurrence).
    ("full_ft_bilstm",          dict(TECHNIQUE="full_ft", HEAD_TYPE="bilstm", LR=1e-4, EPOCHS=15, PATIENCE=4)),
    ("adapter_bilstm",          dict(TECHNIQUE="adapter", HEAD_TYPE="bilstm", LR=3e-4, EPOCHS=20, PATIENCE=5)),     # adapters after each LSTM layer
    ("partial_ft_top1_bilstm",  dict(TECHNIQUE="partial_ft", PARTIAL_TOP_BLOCKS=1, HEAD_TYPE="bilstm", LR=3e-4, EPOCHS=20, PATIENCE=5)),
    ("bitfit_bilstm",           dict(TECHNIQUE="bitfit", HEAD_TYPE="bilstm", LR=1e-3, EPOCHS=20, PATIENCE=5)),
    ("adapter_pfeiffer_bilstm", dict(TECHNIQUE="adapter", ADAPTER_TYPE="pfeiffer", HEAD_TYPE="bilstm", LR=3e-4, EPOCHS=20, PATIENCE=5)),
    ("frozen_bilstm",           dict(TECHNIQUE="partial_ft", PARTIAL_TOP_BLOCKS=0, HEAD_TYPE="bilstm", LR=1e-3, EPOCHS=20, PATIENCE=5)),   # Phase-1 LSTM frozen: only the head learns
    ("full_ft",                 dict(TECHNIQUE="full_ft", LR=1e-4, EPOCHS=15, PATIENCE=4)),
    ("adapter",                 dict(TECHNIQUE="adapter", LR=3e-4, EPOCHS=20, PATIENCE=5)),
]
''', ""),
    "gdn": ('''SWEEP = [   # (name, settings that differ from BASE). The GDN encoder is already bidirectional, so linear and BiLSTM heads are both tried.
    ("full_ft",                 dict(TECHNIQUE="full_ft", LR=3e-5, EPOCHS=10, PATIENCE=3)),
    ("adapter",                 dict(TECHNIQUE="adapter", LR=3e-4, EPOCHS=20, PATIENCE=5)),
    ("lorma",                   dict(TECHNIQUE="lorma", LR=3e-4, EPOCHS=20, PATIENCE=5)),
    ("lora",                    dict(TECHNIQUE="lora", LR=3e-4, EPOCHS=20, PATIENCE=5)),
    ("bitfit",                  dict(TECHNIQUE="bitfit", LR=1e-3, EPOCHS=20, PATIENCE=5)),
    ("partial_ft_top2",         dict(TECHNIQUE="partial_ft", PARTIAL_TOP_BLOCKS=2, LR=1e-4, EPOCHS=15, PATIENCE=4)),
    ("full_ft_bilstm",          dict(TECHNIQUE="full_ft", HEAD_TYPE="bilstm", LR=3e-5, EPOCHS=10, PATIENCE=3)),
    ("adapter_bilstm",          dict(TECHNIQUE="adapter", HEAD_TYPE="bilstm", LR=3e-4, EPOCHS=20, PATIENCE=5)),
    ("lorma_bilstm",            dict(TECHNIQUE="lorma", HEAD_TYPE="bilstm", LR=3e-4, EPOCHS=20, PATIENCE=5)),
    ("lorma_pi",                dict(TECHNIQUE="lorma", LORMA_VARIANT="pi", LR=3e-4, EPOCHS=20, PATIENCE=5)),
    ("adapter_pfeiffer",        dict(TECHNIQUE="adapter", ADAPTER_TYPE="pfeiffer", LR=3e-4, EPOCHS=20, PATIENCE=5)),
    ("frozen",                  dict(TECHNIQUE="partial_ft", PARTIAL_TOP_BLOCKS=0, LR=1e-3, EPOCHS=20, PATIENCE=5)),   # Phase-1 encoder frozen: only the head (+ final norm) learns
]
''', "BASE.update(WINDOW=1024, TRAIN_STRIDE=512, INFER_STRIDE=512)   # the Phase-1 encoder read prompts of up to 1024 tokens\n"),
}
_DEC_SWEEP = DEFAULT_SWEEP.replace('("full_ft_bilstm_clsw_inv",dict(TECHNIQUE="full_ft", HEAD_TYPE="bilstm", CLASS_WEIGHTING="inv", LR=3e-5, EPOCHS=10, PATIENCE=3)),',
                                   '("frozen_bilstm",          dict(TECHNIQUE="partial_ft", PARTIAL_TOP_BLOCKS=0, HEAD_TYPE="bilstm", LR=1e-3, EPOCHS=20, PATIENCE=5)),   # Phase-1 model frozen: only the head learns')
FAMILY_SWEEPS["mor"] = (_DEC_SWEEP, "")
FAMILY_SWEEPS["canon"] = (_DEC_SWEEP, "")


def sweep_settings(name, fam, family_key):
    return f'''import json
# =====================================================================================================
# SWEEP SETTINGS -- every value is YOUR choice (untuned placeholders)
# =====================================================================================================
SWEEP_NAME = "{name}"
MODEL_KEY = "{family_key}"
PHASE1_CHECKPOINT = "{fam['checkpoint']}"
TOKENIZER_NAME = "a1bpe_tokenizer_160Mw_1ep_full_v40000_160000000w_seed42_minf1.json"
PEFT_TARGETS = {fam['default_targets']}
{fam['extra_settings']}INPUT_MODE = "none"            # tokenization is shared by all setups in this notebook

{BASE_DEFAULTS}{FAMILY_SWEEPS.get(family_key, ("", ""))[1]}
{FAMILY_SWEEPS.get(family_key, (DEFAULT_SWEEP, ""))[0]}
TOP_K_SUBMIT = 4              # retrain the best K setups on fold1+fold2 and write a submission zip for each
CONFIRM_TOP_M = 5             # re-run the best M setups the other way round (train fold2 -> score fold1) and rank by the
                              # MEAN of both folds (more reliable than one 35-document fold); 0 = rank on fold2 only
MAX_SWEEP_HOURS = None        # None = no limit (every setup runs). A number = stop STARTING new setups after that many hours
QUICK_TEST = False            # True = tiny end-to-end check (2 docs per fold, 1 epoch, 512-token windows, first 2 setups)

if QUICK_TEST:
    SWEEP = SWEEP[:2]
    BASE.update(WINDOW=512, TRAIN_STRIDE=512, INFER_STRIDE=256)
    SWEEP = [(n, dict(o, EPOCHS=1, PATIENCE=0)) for n, o in SWEEP]
    SWEEP_NAME += "_quicktest"
    CONFIRM_TOP_M = min(CONFIRM_TOP_M, 1)
ATTENTION = BASE["ATTENTION"]
print(f"{{SWEEP_NAME}}: {{len(SWEEP)}} setups:", [n for n, _ in SWEEP])
'''


SWEEP_PREP = '''t0 = time.time()
TAG_NAMES = bio_tag_names(LABELS)
DOCS = {f: prepare_docs(bpe_tokenizer, TEXTS[f], GOLDS[f], LABELS, INPUT_MODE) for f in TEXTS}
TEST_DOCS = prepare_docs(bpe_tokenizer, TEST_TEXTS, None, LABELS, INPUT_MODE)
if INPUT_MODE == "none":
    check_encode_identical(bpe_tokenizer, {**TEXTS["fold1"], **TEXTS["fold2"], **TEST_TEXTS})
    print("token ids identical to BPETokenizer.encode on every document")
GOLD3 = {f: {d: [e[:3] for e in GOLDS[f][d]] for d in GOLDS[f]} for f in GOLDS}


def trim_tables(folds):
    tx, gd, sp = {}, {}, {}
    for f in folds:
        tx.update(TEXTS[f]); gd.update(GOLDS[f]); sp.update({d: DOCS[f][d]["spans"] for d in DOCS[f]})
    return learn_trim_table(tx, gd, sp, LABELS)


TRIM = {"fold1": trim_tables(["fold1"]), "fold2": trim_tables(["fold2"]), "both": trim_tables(["fold1", "fold2"])}
mis = sum(len(gold_to_token_tags(DOCS[f][d]["spans"], GOLDS[f][d], LABELS)[1]) for f in DOCS for d in DOCS[f])
print(f"{len(TAG_NAMES)} tags; {mis} gold entities not on token boundaries; {time.time()-t0:.0f}s")
'''

SWEEP_LOAD = '''t0 = time.time()
_ck = torch.load(fetch_model_file(PHASE1_CHECKPOINT), map_location="cpu", mmap=True)
PHASE1_STATE = _ck["model"] if "model" in _ck else _ck
del _ck
_lm = build_phase1_lm()
PHASE1_PARAMS = sum(p.numel() for p in _lm.parameters())
SANITY_LOSS = phase1_sanity(_lm, DOCS["fold1"])
print(f"Phase-1 model {MODEL_KEY}: {PHASE1_PARAMS:,} parameters; next-token loss on NER documents = {SANITY_LOSS:.3f} nats "
      f"(far below {math.log(VOCAB):.1f} = trained weights loaded; about {math.log(VOCAB):.1f} = random weights) [{time.time()-t0:.0f}s]")
assert SANITY_LOSS < 8.0, "the Phase-1 weights do not seem to be loaded correctly"
del _lm; gc.collect(); torch.cuda.empty_cache() if device.type == "cuda" else None


def build_ner(C):
    lm = build_phase1_lm()
    bb = make_backbone(lm, C)
    ner = NERModel(bb, bb.d_model, len(TAG_NAMES), C["HEAD_TYPE"], C["HEAD_DROPOUT"], C["HEAD_NORM"], C["HEAD_LSTM_HIDDEN"],
                   C.get("HEAD_LSTM_LAYERS", 1), C.get("SPAN_DIM", 0), C.get("SPAN_MAX_LEN", 32))
    info = apply_technique(ner, C)
    return ner.to(device), info


def is_causal(C):
    return C["ATTENTION"] != "bidirectional" and C["HEAD_TYPE"] != "bilstm"


def free(*objs):
    for o in objs:
        del o
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
'''

SWEEP_RUN = '''# ---- one setup, one direction: train on `tr`, choose the epoch and report on `ev` --------------------------------
OTHER = {"fold1": "fold2", "fold2": "fold1"}
os.makedirs(f"{WORK}/sweep", exist_ok=True)


def result_name(name, tr):
    return f"{SWEEP_NAME}__{name}__{tr}to{OTHER[tr]}.json"


def load_cached_result(name, tr):
    fn = result_name(name, tr)
    for path in (f"{WORK}/sweep/{fn}",):
        if os.path.exists(path):
            return json.load(open(path))
    try:
        p = fetch_model_file(fn, f"{WORK}/sweep/{fn}")
        print(f"  (found finished result {fn} in storage -- not re-running it)")
        return json.load(open(p))
    except Exception:
        return None


def print_setup_report(res):
    """Everything the write-up needs about one run, printed so the notebook output alone is enough."""
    print(f"  [{res['setup']} {res['train_fold']}->{res['eval_fold']}] settings: " + json.dumps(res["overrides"], default=str))
    print("  epochs: " + " | ".join(f"{h['epoch']}: loss {h['train_loss']}, F1 {h.get('heldout_macroF1')}" for h in res["history"]))
    print(f"  {'label':10s} {'P':>6s} {'R':>6s} {'F1':>6s} {'tp':>5s} {'fp':>5s} {'fn':>5s}")
    for l, d in res["per_label"].items():
        print(f"  {l:10s} {d['P']:6.3f} {d['R']:6.3f} {d['F1']:6.3f} {d['tp']:5d} {d['fp']:5d} {d['fn']:5d}")
    print("  errors:", res["errors"])
    print("  top label confusions:", dict(list(res["label_confusions"].items())[:10]))
    a = res["analysis"]
    print("  boundary error kinds:", a["boundary_error_kinds"])
    print("  most common boundary differences (extra/missing text):", a["boundary_diff_top"][:10])
    print("  by region of the document:", a["by_region"])
    print("  by document length:", a["by_doc_length"])
    ti = res["technique_info"]
    print(f"  params: trainable {ti['params_trainable']:,} / total {ti['params_total']:,} ({ti['trainable_pct_of_backbone']}% of Phase-1 "
          f"{res['phase1_params']:,}) | train {res['train_seconds']}s, {res['tokens_per_sec']} tok/s, peak {res['peak_mem_gb']} GB | "
          f"inference {res['inference_seconds']}s, {res['inference_tokens_per_sec']} tok/s")


def run_setup(name, overrides, tr):
    cached = load_cached_result(name, tr)
    if cached is not None:
        return cached
    ev = OTHER[tr]
    C = dict(BASE, **overrides)
    print(f"\\n=== {name}  [{tr} -> {ev}]  " + ", ".join(f"{k}={v}" for k, v in overrides.items()))
    t0 = time.time()
    ner, info = build_ner(C)
    causal = is_causal(C)

    def eval_fn(model):
        preds, _ = predict_docs(model, DOCS[ev], LABELS, C, device, causal, *TRIM[tr])
        s = strict_scores(GOLD3[ev], preds, LABELS)
        return s["macro_F1"], dict(heldout_P=round(s["macro_P"], 4), heldout_R=round(s["macro_R"], 4))

    cv = train_ner(ner, DOCS[tr], C, device, eval_fn=eval_fn)
    load_trainable_state(ner, cv["best_state"])
    preds, infer_s = predict_docs(ner, DOCS[ev], LABELS, C, device, causal, *TRIM[tr])
    S = strict_scores(GOLD3[ev], preds, LABELS)
    n_tok = sum(len(d["ids"]) for d in DOCS[ev].values())
    res = dict(sweep=SWEEP_NAME, setup=name, model=MODEL_KEY, train_fold=tr, eval_fold=ev, overrides=overrides,
               settings={k: v for k, v in C.items()}, technique_info=info, phase1_params=PHASE1_PARAMS,
               best_epoch=cv["best_epoch"], macro_F1=round(S["macro_F1"], 4), macro_P=round(S["macro_P"], 4),
               macro_R=round(S["macro_R"], 4),
               per_label_F1={l: round(d["F1"], 4) for l, d in S["per_label"].items()},
               per_label={l: {k: (round(v, 4) if isinstance(v, float) else v) for k, v in d.items()} for l, d in S["per_label"].items()},
               errors=error_breakdown(GOLD3[ev], preds), label_confusions=confusion_pairs(GOLD3[ev], preds),
               history=cv["history"], train_seconds=cv["train_seconds"], peak_mem_gb=cv["peak_mem_gb"],
               tokens_per_sec=cv["tokens_per_sec"], inference_seconds=round(infer_s, 1),
               inference_tokens_per_sec=round(n_tok / max(infer_s, 1e-9), 1), wall_seconds=round(time.time() - t0, 1),
               analysis=detailed_analysis(GOLD3[ev], preds, TEXTS[ev]))
    pf = f"{WORK}/sweep/{SWEEP_NAME}__{name}__{tr}to{ev}_pred.json"
    json.dump(preds, open(pf, "w"))
    upload_model_file(pf)
    fn = result_name(name, tr)
    json.dump(res, open(f"{WORK}/sweep/{fn}", "w"), indent=1, default=str)
    upload_model_file(f"{WORK}/sweep/{fn}")
    print(f"  -> {name} [{tr}->{ev}] strict macro-F1 {res['macro_F1']} (P {res['macro_P']}, R {res['macro_R']}), best epoch "
          f"{res['best_epoch']}, trainable {info['params_trainable']:,} ({info['trainable_pct_of_backbone']}%), "
          f"peak {res['peak_mem_gb']} GB, {res['wall_seconds']:.0f}s")
    print_setup_report(res)
    free(ner)
    return res


# ---- stage 1: every setup, fold1 -> fold2 -------------------------------------------------------------------------
STAGE1 = {}
for name, overrides in SWEEP:
    if MAX_SWEEP_HOURS is not None and (time.time() - NB_START) / 3600 > MAX_SWEEP_HOURS:
        print(f"MAX_SWEEP_HOURS reached: skipping {name} and the rest")
        break
    STAGE1[name] = run_setup(name, overrides, "fold1")
'''

SWEEP_CONFIRM = '''# ---- stage 2: re-check the best CONFIRM_TOP_M setups the other way round (fold2 -> fold1) -------------------------
order = sorted(STAGE1, key=lambda n: -STAGE1[n]["macro_F1"])
STAGE2 = {}
for name in order[:CONFIRM_TOP_M]:
    STAGE2[name] = run_setup(name, dict(SWEEP)[name], "fold2")

RANK = []
for name in order:
    f2 = STAGE1[name]["macro_F1"]
    f1 = STAGE2[name]["macro_F1"] if name in STAGE2 else None
    mean = (f1 + f2) / 2 if f1 is not None else None
    RANK.append(dict(setup=name, fold2_F1=f2, fold1_F1=f1, mean_F1=mean,
                     best_epochs=[STAGE1[name]["best_epoch"]] + ([STAGE2[name]["best_epoch"]] if name in STAGE2 else [])))
# confirmed setups are ranked by the two-fold mean and come first; the rest follow by fold2 F1
RANK.sort(key=lambda r: (r["mean_F1"] is None, -(r["mean_F1"] if r["mean_F1"] is not None else r["fold2_F1"])))

print(f"\\n{'rank':4s} {'setup':22s} {'fold2 F1':>9s} {'fold1 F1':>9s} {'mean':>7s} {'trainable':>12s} {'peak GB':>8s} {'train s':>8s}")
for i, r in enumerate(RANK, 1):
    s = STAGE1[r["setup"]]
    print(f"{i:<4d} {r['setup']:22s} {r['fold2_F1']:9.4f} {('%.4f' % r['fold1_F1']) if r['fold1_F1'] is not None else '-':>9s} "
          f"{('%.4f' % r['mean_F1']) if r['mean_F1'] is not None else '-':>7s} {s['technique_info']['params_trainable']:>12,} "
          f"{str(s['peak_mem_gb']):>8s} {s['train_seconds']:>8.0f}")
print("\\nper-label fold2 F1:")
for r in RANK:
    print(f"  {r['setup']:22s}", {l: v for l, v in STAGE1[r['setup']]['per_label_F1'].items()})
'''

SWEEP_FINAL = '''# ---- stage 3: retrain the top K on fold1+fold2 and write their submission zips -----------------------------------
def validate_prediction(pred, texts, labels):
    assert set(pred) == set(texts), f"id mismatch: missing {len(set(texts)-set(pred))}, extra {len(set(pred)-set(texts))}"
    for did, ents in pred.items():
        seen = set()
        for x in ents:
            s, e, l = x
            assert type(s) is int and type(e) is int and 0 <= s < e <= len(texts[did]) and l in labels, (did, x)
            assert (s, e, l) not in seen, ("duplicate", did, x); seen.add((s, e, l))


both = {**DOCS["fold1"], **DOCS["fold2"]}
FINAL = []
for r in RANK[:TOP_K_SUBMIT]:
    name = r["setup"]
    C = dict(BASE, **dict(SWEEP)[name])
    epochs = max(1, round(sum(r["best_epochs"]) / len(r["best_epochs"])))
    print(f"\\n=== refit {name} on fold1+fold2 for {epochs} epochs")
    tag = f"{SWEEP_NAME}_{name}"
    ner, info = build_ner(C)
    try:                                   # resume: a refit finished in an earlier (timed-out) session is reused
        load_trainable_state(ner, torch.load(fetch_model_file(f"{tag}_final_state.pt"), map_location="cpu"))
        fit = dict(best_state=None, train_seconds=0.0)
        print(f"  (found refitted state {tag}_final_state.pt in storage -- not retraining)")
    except Exception:
        fit = train_ner(ner, both, dict(C, EPOCHS=epochs, PATIENCE=0), device, eval_fn=None)
    pred, test_s = predict_docs(ner, TEST_DOCS, LABELS, C, device, is_causal(C), *TRIM["both"])
    validate_prediction(pred, TEST_TEXTS, LABELS)
    tag = f"{SWEEP_NAME}_{name}"
    with open(f"{WORK}/prediction.json", "w", encoding="utf-8") as f:
        json.dump(pred, f)
    zp = f"{WORK}/submission_{tag}.zip"
    with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED) as z:
        z.write(f"{WORK}/prediction.json", arcname="prediction.json")
    with zipfile.ZipFile(zp) as z:
        assert z.namelist() == ["prediction.json"]
    if fit["best_state"] is not None:
        torch.save(fit["best_state"], f"{WORK}/{tag}_final_state.pt")
        upload_model_file(f"{WORK}/{tag}_final_state.pt")
    FINAL.append(dict(setup=name, zip=os.path.basename(zp), refit_epochs=epochs, refit_seconds=fit["train_seconds"],
                      test_entities=sum(len(v) for v in pred.values()), test_inference_seconds=round(test_s, 1)))
    print(f"  wrote {zp}: {FINAL[-1]['test_entities']} entities")
    del ner, fit
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
'''

SWEEP_SAVE = '''SUMMARY = dict(sweep=SWEEP_NAME, model=MODEL_KEY, phase1_checkpoint=PHASE1_CHECKPOINT, phase1_params=PHASE1_PARAMS,
               sanity_lm_loss=round(SANITY_LOSS, 4), base=BASE, ranking=RANK, final=FINAL,
               stage1={k: {kk: v[kk] for kk in ("macro_F1", "macro_P", "macro_R", "best_epoch", "per_label_F1", "errors",
                                                 "train_seconds", "peak_mem_gb", "technique_info")} for k, v in STAGE1.items()},
               stage2={k: {kk: v[kk] for kk in ("macro_F1", "best_epoch", "per_label_F1")} for k, v in STAGE2.items()},
               notebook_minutes=round((time.time() - NB_START) / 60, 1),
               gpu=torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu")
sp = f"{WORK}/{SWEEP_NAME}_summary.json"
json.dump(SUMMARY, open(sp, "w"), indent=1, default=str)
upload_model_file(sp)
print("=" * 100)
print(f"SWEEP {SWEEP_NAME} ({MODEL_KEY}, {PHASE1_PARAMS:,} params, sanity loss {SANITY_LOSS:.3f}) -- {SUMMARY['notebook_minutes']} min")
for i, r in enumerate(RANK, 1):
    print(f"  {i}. {r['setup']}: fold2 {r['fold2_F1']}" + (f", fold1 {r['fold1_F1']}, mean {round(r['mean_F1'], 4)}" if r['mean_F1'] is not None else ""))
print("Submission zips (top", TOP_K_SUBMIT, "):", [f["zip"] for f in FINAL])
print("\\nFull table (fold1->fold2):")
print(f"{'setup':22s} {'F1':>6s} {'P':>6s} {'R':>6s} {'ep':>3s} {'trainable':>12s} {'%':>8s} {'GB':>6s} {'train s':>8s} {'tok/s':>8s} "
      f"{'correct':>7s} {'bound':>6s} {'b+lab':>6s} {'label':>6s} {'spur':>6s} {'missed':>6s}")
for n, v in STAGE1.items():
    e, ti = v["errors"], v["technique_info"]
    print(f"{n:22s} {v['macro_F1']:6.4f} {v['macro_P']:6.4f} {v['macro_R']:6.4f} {v['best_epoch']:3d} {ti['params_trainable']:>12,} "
          f"{ti['trainable_pct_of_backbone']:>8} {str(v['peak_mem_gb']):>6s} {v['train_seconds']:>8.0f} {v['tokens_per_sec']:>8.0f} "
          f"{e.get('correct',0):>7d} {e.get('boundary_error',0):>6d} {e.get('boundary_and_label_error',0):>6d} {e.get('label_error',0):>6d} "
          f"{e.get('spurious',0):>6d} {e.get('missed_entirely',0):>6d}")
print("\\nPer-label F1 (fold1->fold2):")
print(f"{'setup':22s} " + " ".join(f"{l[:9]:>9s}" for l in LABELS))
for n, v in STAGE1.items():
    print(f"{n:22s} " + " ".join(f"{v['per_label_F1'][l]:9.3f}" for l in LABELS))
if STAGE2:
    print("\\nPer-label F1 (fold2->fold1, confirmed setups):")
    for n, v in STAGE2.items():
        print(f"{n:22s} " + " ".join(f"{v['per_label_F1'][l]:9.3f}" for l in LABELS))
print("\\nRefits:", FINAL)
print("\\nSUMMARY_JSON_START")
print(json.dumps(SUMMARY, default=str))
print("SUMMARY_JSON_END")
print("=" * 100)
print("Send me this notebook (with outputs) or", os.path.basename(sp))
'''


def build_sweep(name, family_key):
    fam = FAMILIES[family_key]
    cells = []
    md = lambda s: cells.append(nbf.v4.new_markdown_cell(s))
    code = lambda s: cells.append(nbf.v4.new_code_cell(s))
    md(f"""# CS-781 Phase 2 (L-NER) — SWEEP `{name}`

**Phase-1 model:** {fam['title']}

One Run All tries every setup in `SWEEP` (Settings) on this model and submits only the best 4:
1. **Stage 1** — each setup: train on fold1, pick the epoch and score strict macro-F1 on fold2 (per-label, errors, parameters, memory, time are all recorded).
2. **Stage 2** — the best `CONFIRM_TOP_M` setups are re-run the other way round (fold2 → fold1); they are ranked by the mean of both folds.
3. **Stage 3** — the top `TOP_K_SUBMIT` setups are retrained on fold1+fold2 and each gets `submission_{name}_<setup>.zip`.

Every finished setup's result is uploaded to the bucket straight away, so if the session stops, **Run All again resumes** (finished setups are skipped). `MAX_SWEEP_HOURS` (default: no limit) can stop new setups early.

**Kaggle:** GPU T4 (one GPU is used), secrets `HF` and `HF_WRITE`. Each setup takes ~6–15 min on this model (estimate). All settings are your choices; the values are untuned placeholders. Class weights (`inv_sqrt`) are on by default because they were the only change that helped the MoE on both fold2 and Codabench.""")
    md("## 1. Settings: the setups to try")
    code(sweep_settings(name, fam, family_key))
    md("## 2. Setup: login, bucket")
    code(SETUP_CELL)
    md("## 3. Phase-2 data")
    code(DATA_CELL)
    md("## 4. Your Phase-1 BPE tokenizer (class unchanged), loaded from storage")
    code(BPE_SRC + TOK_LOAD)
    md("## 5. Character offsets, BIO tags, span decoding, boundary repair, strict scorer")
    code(ALIGN_SRC)
    md(f"## 6. Phase-1 model code (copied unchanged from {fam['source_nb']})")
    code(fam["pre"] + "\n" + fam["model_src"])
    md("## 7. NER backbone wrapper + loading the Phase-1 model")
    code(fam["backbone_src"] + "\n\n" + fam["build_src"])
    md("## 8. NER engine: head, adaptation methods, training, inference")
    code(CORE_SRC + "\n\nEVALUATOR_SRC = " + repr(EVAL_SRC))
    md("## 9. Tokenize the documents and learn the boundary-repair tables (one per training fold)")
    code(SWEEP_PREP)
    md("## 10. Load the Phase-1 checkpoint and check it")
    code(SWEEP_LOAD)
    md("## 11. Stage 1: every setup, fold1 → fold2")
    code(SWEEP_RUN)
    md("## 12. Stage 2: re-check the best setups fold2 → fold1, and the ranking table")
    code(SWEEP_CONFIRM)
    md("## 13. Stage 3: retrain the top K on both folds, write submission zips")
    code(SWEEP_FINAL)
    md("## 14. Save the summary to the bucket")
    code(SWEEP_SAVE)
    nb = nbf.v4.new_notebook()
    nb["cells"] = cells
    nb["metadata"] = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                      "language_info": {"name": "python"}}
    nbf.validate(nb)
    path = OUT + name + ".ipynb"
    nbf.write(nb, path)
    return path


if __name__ == "__main__":
    for n, f in [("p2s_gqa160", "gqa160"), ("p2s_lstm84", "lstm"), ("p2s_gdn", "gdn"), ("p2s_mor2", "mor"), ("p2s_canon", "canon")]:
        print(build_sweep(n, f))
