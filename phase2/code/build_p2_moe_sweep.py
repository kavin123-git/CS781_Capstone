"""Builds the MoE improvement sweep notebook (p2s_moe2): ~13 setups on the best Phase-1 model, top 4 submitted."""
import sys
sys.path.insert(0, "/home/claude/p2")
import build_p2_sweeps as B

MOE_SWEEP = '''SWEEP = [   # (name, settings that differ from BASE).  BASE = the p2_13 recipe: Houlsby adapter + BiLSTM(256) head + inv_sqrt class weights.
    # ---- references: the two setups below are NEVER submitted; they measure the noise floor of a 35-document fold ----------
    ("base_adapter_bilstm", dict(TECHNIQUE="adapter", HEAD_TYPE="bilstm", LR=3e-4, EPOCHS=20, PATIENCE=5)),
    ("base_seed1",          dict(TECHNIQUE="adapter", HEAD_TYPE="bilstm", LR=3e-4, EPOCHS=20, PATIENCE=5, SEED=1)),
    # ---- loss / imbalance ------------------------------------------------------------------------------------------------
    ("label_smooth",        dict(TECHNIQUE="adapter", HEAD_TYPE="bilstm", LR=3e-4, EPOCHS=20, PATIENCE=5, LABEL_SMOOTHING=0.1)),
    ("focal",               dict(TECHNIQUE="adapter", HEAD_TYPE="bilstm", LR=3e-4, EPOCHS=20, PATIENCE=5, FOCAL_GAMMA=2.0)),
    ("oversample_rare",     dict(TECHNIQUE="adapter", HEAD_TYPE="bilstm", LR=3e-4, EPOCHS=20, PATIENCE=5, OVERSAMPLE=3,
                                 OVERSAMPLE_LABELS=["APP", "RESP", "WIT"])),        # windows with a weak label are seen 3x per epoch
    # ---- context ---------------------------------------------------------------------------------------------------------
    ("header256",           dict(TECHNIQUE="adapter", HEAD_TYPE="bilstm", LR=3e-4, EPOCHS=20, PATIENCE=5, HEADER=256,
                                 WINDOW=1792, TRAIN_STRIDE=896, INFER_STRIDE=896)),   # first 256 tokens (court, parties) prepended to every later window; WINDOW+HEADER must be <= 2048
    ("window1024",          dict(TECHNIQUE="adapter", HEAD_TYPE="bilstm", LR=3e-4, EPOCHS=20, PATIENCE=5, WINDOW=1024,
                                 TRAIN_STRIDE=512, INFER_STRIDE=512, BATCH=4, ACCUM=2)),
    # ---- prediction head -------------------------------------------------------------------------------------------------
    ("bilstm512",           dict(TECHNIQUE="adapter", HEAD_TYPE="bilstm", LR=3e-4, EPOCHS=20, PATIENCE=5, HEAD_LSTM_HIDDEN=512)),
    ("bilstm2layer",        dict(TECHNIQUE="adapter", HEAD_TYPE="bilstm", LR=3e-4, EPOCHS=20, PATIENCE=5, HEAD_LSTM_LAYERS=2)),
    ("span_decode",         dict(TECHNIQUE="adapter", HEAD_TYPE="bilstm", LR=3e-4, EPOCHS=20, PATIENCE=5, SPAN_DIM=128,
                                 SPAN_WEIGHT=1.0, DECODE="span")),     # span classifier trained jointly with the token tagger, decoded as spans
    # ---- adaptation capacity / schedule ----------------------------------------------------------------------------------
    ("adapter_dim128",      dict(TECHNIQUE="adapter", HEAD_TYPE="bilstm", LR=3e-4, EPOCHS=20, PATIENCE=5, ADAPTER_DIM=128)),
    ("lr_high_long",        dict(TECHNIQUE="adapter", HEAD_TYPE="bilstm", LR=1e-3, EPOCHS=30, PATIENCE=6)),
    # ---- a speculative combination of the ideas most likely to stack -----------------------------------------------------
    ("combo_hdr_512_ls",    dict(TECHNIQUE="adapter", HEAD_TYPE="bilstm", LR=3e-4, EPOCHS=20, PATIENCE=5, HEADER=256,
                                 WINDOW=1792, TRAIN_STRIDE=896, INFER_STRIDE=896, HEAD_LSTM_HIDDEN=512, LABEL_SMOOTHING=0.1)),
]
NO_SUBMIT = {"base_adapter_bilstm", "base_seed1"}     # reference runs: ranked and reported, but never refitted / zipped
'''

B.FAMILY_SWEEPS["moe_hc"] = (MOE_SWEEP, "")

_orig_settings = B.sweep_settings


def settings(name, fam, family_key):
    s = _orig_settings(name, fam, family_key)
    s = s.replace("CONFIRM_TOP_M = 5 ", "CONFIRM_TOP_M = 5 ", 1)
    s = s.replace("MAX_SWEEP_HOURS = None        # None = no limit (every setup runs). A number = stop STARTING new setups after that many hours",
                  "MAX_SWEEP_HOURS = 7.5         # stop STARTING new stage-1 setups after this many hours (Kaggle sessions end at 12 h; Run All resumes). None = no limit")
    # run order is SWEEP order; the quick test must also exercise the new code paths
    s = s.replace("    SWEEP = SWEEP[:2]\n", "    SWEEP = [x for x in SWEEP if x[0] in ('base_adapter_bilstm', 'header256', 'span_decode', 'oversample_rare', 'focal')][:5]\n")
    return s


B.sweep_settings = settings

# ---- run-time pieces -------------------------------------------------------------------------------------------------
MAKE_C = '''

def make_C(overrides):
    """BASE + overrides, with OVERSAMPLE_LABELS (names) turned into BIO tag ids."""
    C = dict(BASE, **overrides)
    C["OVERSAMPLE_TAGS"] = [t for l in C.get("OVERSAMPLE_LABELS", []) for t in (1 + 2 * LABELS.index(l), 2 + 2 * LABELS.index(l))]
    return C


def alt_decodes(C):
    """Other decoders applied to the SAME trained model (a decoding ablation that costs only inference)."""
    alts = {"argmax": dict(DECODE="argmax"), "merge_average": dict(MERGE="average")}
    if C.get("SPAN_DIM", 0):
        alts["viterbi" if C["DECODE"] == "span" else "span"] = dict(DECODE="viterbi" if C["DECODE"] == "span" else "span")
    return {k: v for k, v in alts.items() if all(C.get(kk) != vv for kk, vv in v.items())}
'''
B.SWEEP_LOAD = B.SWEEP_LOAD + MAKE_C

run = B.SWEEP_RUN
assert run.count("C = dict(BASE, **overrides)") == 1
run = run.replace("C = dict(BASE, **overrides)", "C = make_C(overrides)")
old = "    res = dict(sweep=SWEEP_NAME, setup=name,"
assert run.count(old) == 1
run = run.replace(old, '''    ALT = {}
    if tr == "fold1":                      # decoding ablation on the trained model (fold2 only)
        for an, patch in alt_decodes(C).items():
            try:
                ap, _ = predict_docs(ner, DOCS[ev], LABELS, dict(C, **patch), device, causal, *TRIM[tr])
                ALT[an] = round(strict_scores(GOLD3[ev], ap, LABELS)["macro_F1"], 4)
            except Exception as ex:
                ALT[an] = f"failed: {type(ex).__name__}: {ex}"
        print("  decoding ablation on this model (strict macro-F1 on", ev + "):", {"main(" + C["DECODE"] + ")": round(S["macro_F1"], 4), **ALT})
''' + old)
assert run.count("analysis=detailed_analysis(GOLD3[ev], preds, TEXTS[ev]))") == 1
run = run.replace("analysis=detailed_analysis(GOLD3[ev], preds, TEXTS[ev]))", "analysis=detailed_analysis(GOLD3[ev], preds, TEXTS[ev]), decode_alt=ALT)")
old_loop = '''    STAGE1[name] = run_setup(name, overrides, "fold1")
'''
assert run.count(old_loop) == 1
run = run.replace(old_loop, '''    try:
        STAGE1[name] = run_setup(name, overrides, "fold1")
    except Exception as ex:                # one failing setup (e.g. out of memory) must not stop the sweep
        import traceback; traceback.print_exc()
        print(f"!!! setup {name} FAILED: {type(ex).__name__}: {ex} -- continuing with the next setup")
        FAILED[name] = f"{type(ex).__name__}: {ex}"
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
''').replace("STAGE1 = {}\n", "STAGE1, FAILED = {}, {}\n")
B.SWEEP_RUN = run

conf = B.SWEEP_CONFIRM
old = '''for name in order[:CONFIRM_TOP_M]:
    STAGE2[name] = run_setup(name, dict(SWEEP)[name], "fold2")
'''
assert conf.count(old) == 1
conf = conf.replace(old, '''CONFIRM = [n for n in order if n not in NO_SUBMIT][:CONFIRM_TOP_M] + [n for n in ["base_adapter_bilstm"] if n in STAGE1]
for name in CONFIRM:                       # the best eligible setups, plus the base recipe for a two-fold reference
    try:
        STAGE2[name] = run_setup(name, dict(SWEEP)[name], "fold2")
    except Exception as ex:
        import traceback; traceback.print_exc()
        print(f"!!! stage 2 of {name} FAILED: {ex}"); gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
''')
conf = conf.replace('''print("\\nper-label fold2 F1:")''', '''print("\\nfailed setups:", FAILED)
print("\\ndecoding ablation (fold2 F1 of the same trained model, other decoders):")
for n in STAGE1:
    print(f"  {n:22s} main {STAGE1[n]['macro_F1']}  alt {STAGE1[n].get('decode_alt')}")
print("\\nper-label fold2 F1:")''')
B.SWEEP_CONFIRM = conf

fin = B.SWEEP_FINAL
for a, b in [("for r in RANK[:TOP_K_SUBMIT]:", "ELIGIBLE = [r for r in RANK if r['setup'] not in NO_SUBMIT]\nfor r in ELIGIBLE[:TOP_K_SUBMIT]:"),
             ("C = dict(BASE, **dict(SWEEP)[name])", "C = make_C(dict(SWEEP)[name])")]:
    assert fin.count(a) == 1, a
    fin = fin.replace(a, b)
B.SWEEP_FINAL = fin

sv = B.SWEEP_SAVE
assert sv.count("final=FINAL,") == 1
sv = sv.replace("final=FINAL,", "final=FINAL, failed=FAILED, decode_alt={k: v.get('decode_alt') for k, v in STAGE1.items()},")
B.SWEEP_SAVE = sv

# the text of the sweep markdown says "Each setup takes ~6-15 min"; correct it by patching the cell source after build
if __name__ == "__main__":
    path = B.build_sweep("p2s_moe2", "moe_hc")
    import nbformat as nbf
    nb = nbf.read(path, as_version=4)
    for c in nb["cells"]:
        if c["cell_type"] == "markdown" and "Each setup takes" in c["source"]:
            c["source"] = c["source"].replace("Each setup takes ~6–15 min on this model (estimate).",
                "Each setup takes ~25–40 min on this model (estimate): about 13 setups in stage 1, 5-6 re-runs in stage 2, 4 refits in stage 3, i.e. 9–11 h in total -- Run All again resumes if the 12 h session limit hits. `base_adapter_bilstm` and `base_seed1` are reference runs (never submitted).")
            c["source"] = c["source"].replace("submits only the best 4", "submits only the best 4 non-reference setups")
    nbf.write(nb, path)
    print(path)
