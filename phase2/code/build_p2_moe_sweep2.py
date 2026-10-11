"""Follow-up MoE sweep (p2s_moe3): stack the winners of p2s_moe2 and test the header idea that crashed there."""
import sys
sys.path.insert(0, "/home/claude/p2")
import build_p2_sweeps as B
import build_p2_moe_sweep as M      # applies the shared run/confirm/final patches (make_C, decode ablation, FAILED dict, ...)

MOE_SWEEP2 = '''SWEEP = [   # (name, settings that differ from BASE).  Every setup except `stack` changes ONE thing from `stack`.
    # `stack` = the two best single changes of the first MoE sweep put together (higher adapter LR + 30 epochs, window 1024).
    ("stack",             dict(TECHNIQUE="adapter", HEAD_TYPE="bilstm", LR=1e-3, EPOCHS=30, PATIENCE=6,
                               WINDOW=1024, TRAIN_STRIDE=512, INFER_STRIDE=512, BATCH=4, ACCUM=2)),
    # ---- document context: does telling every window which case it belongs to help?  (WINDOW + HEADER must be <= 2048) ----
    ("stack_header256",   dict(TECHNIQUE="adapter", HEAD_TYPE="bilstm", LR=1e-3, EPOCHS=30, PATIENCE=6,
                               WINDOW=1024, TRAIN_STRIDE=512, INFER_STRIDE=512, BATCH=4, ACCUM=2, HEADER=256)),
    ("stack_header512",   dict(TECHNIQUE="adapter", HEAD_TYPE="bilstm", LR=1e-3, EPOCHS=30, PATIENCE=6,
                               WINDOW=1024, TRAIN_STRIDE=512, INFER_STRIDE=512, BATCH=4, ACCUM=2, HEADER=512)),
    # ---- shorter windows still (closer to "sentence-level" context) -------------------------------------------------------
    ("stack_window512",   dict(TECHNIQUE="adapter", HEAD_TYPE="bilstm", LR=1e-3, EPOCHS=30, PATIENCE=6,
                               WINDOW=512, TRAIN_STRIDE=256, INFER_STRIDE=256, BATCH=8, ACCUM=1)),
    # ---- weak-label oversampling on top of the stack ----------------------------------------------------------------------
    ("stack_oversample",  dict(TECHNIQUE="adapter", HEAD_TYPE="bilstm", LR=1e-3, EPOCHS=30, PATIENCE=6,
                               WINDOW=1024, TRAIN_STRIDE=512, INFER_STRIDE=512, BATCH=4, ACCUM=2,
                               OVERSAMPLE=3, OVERSAMPLE_LABELS=["APP", "RESP", "WIT"])),
    # ---- the compulsory LoRMA technique with the same schedule and window (it was still improving at epoch 20 earlier) -----
    ("stack_lorma",       dict(TECHNIQUE="lorma", HEAD_TYPE="bilstm", LR=1e-3, EPOCHS=30, PATIENCE=6,
                               WINDOW=1024, TRAIN_STRIDE=512, INFER_STRIDE=512, BATCH=4, ACCUM=2)),
]
NO_SUBMIT = set()
'''
B.FAMILY_SWEEPS["moe_hc"] = (MOE_SWEEP2, "")

_prev = B.sweep_settings          # = M.settings (wraps the original)


def settings2(name, fam, family_key):
    s = _prev(name, fam, family_key)
    a = "    SWEEP = [x for x in SWEEP if x[0] in ('base_adapter_bilstm', 'header256', 'span_decode', 'oversample_rare', 'focal')][:5]\n"
    assert s.count(a) == 1
    s = s.replace(a, "    SWEEP = [x for x in SWEEP if x[0] in ('stack_header512', 'stack_window512', 'stack_oversample', 'stack_lorma')]\n")
    a = "TOP_K_SUBMIT = 4 "
    assert s.count(a) == 1
    s = s.replace(a, "TOP_K_SUBMIT = 3 ")
    a = "CONFIRM_TOP_M = 5 "
    assert s.count(a) == 1
    s = s.replace(a, "CONFIRM_TOP_M = 4 ")
    a = 'ATTENTION = BASE["ATTENTION"]\n'
    assert s.count(a) == 1
    s = s.replace(a, a + '''for _n, _o in SWEEP:       # fail now, not hours into the run
    _c = dict(BASE, **_o)
    assert _c["WINDOW"] + _c.get("HEADER", 0) <= 2048, f"{_n}: WINDOW + HEADER must be <= 2048 (the model has 2048 positions)"
''')
    s = s.replace("MAX_SWEEP_HOURS = 7.5 ", "MAX_SWEEP_HOURS = 6.0 ")
    return s


B.sweep_settings = settings2

# the reference setup `stack` must always get its fold2 -> fold1 run (every header / window / LoRMA claim is a comparison with it), without running twice
_old = 'CONFIRM = [n for n in order if n not in NO_SUBMIT][:CONFIRM_TOP_M] + [n for n in ["base_adapter_bilstm"] if n in STAGE1]'
assert B.SWEEP_CONFIRM.count(_old) == 1
B.SWEEP_CONFIRM = B.SWEEP_CONFIRM.replace(_old, 'CONFIRM = [n for n in order if n not in NO_SUBMIT][:CONFIRM_TOP_M]\nCONFIRM += [n for n in ["stack"] if n in STAGE1 and n not in CONFIRM]')

if __name__ == "__main__":
    path = B.build_sweep("p2s_moe3", "moe_hc")
    import nbformat as nbf
    nb = nbf.read(path, as_version=4)
    for c in nb["cells"]:
        if c["cell_type"] == "markdown" and "Each setup takes" in c["source"]:
            c["source"] = c["source"].replace("Each setup takes ~6–15 min on this model (estimate).",
                "FOLLOW-UP to `p2s_moe2`: six setups, all built on `stack` (higher LR + 30 epochs + window 1024, the two winners of the first sweep). "
                "It tests the document-header idea that crashed in the first sweep (my window bug), a shorter window, oversampling of the weak labels, and LoRMA. "
                "Each setup takes ~20–35 min (estimate): 6 in stage 1, 4 re-runs in stage 2, 3 refits in stage 3, about 5–6 h in total -- Run All again resumes if the session limit hits.")
            c["source"] = c["source"].replace("the best 4", "the best 3").replace("best 4", "best 3")
    nbf.write(nb, path)
    print(path)
