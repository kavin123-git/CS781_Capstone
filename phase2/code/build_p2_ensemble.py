"""Builds the Phase-2 span-voting ENSEMBLE notebook (CPU only): combines the predictions of the models already trained
(MoE runs, GQA / LSTM / MoR / GDN sweep winners). Fold2 predictions come from the bucket; test predictions from the
submission zips (attached as a Kaggle dataset)."""
import nbformat as nbf, sys
sys.path.insert(0, "/home/claude/p2")
from build_p2_sweeps import *

ENS_NAME = "p2_16_span_vote_ensemble"

ENS_SETTINGS = '''import json
# =====================================================================================================
# ENSEMBLE SETTINGS -- every value is YOUR choice
# =====================================================================================================
RUN_NAME = "''' + ENS_NAME + '''"
QUICK_TEST = False
# (key, fold2 predictions of the fold1-trained model in the bucket, name of the submission zip of the fold1+fold2 refit WITHOUT ".zip"
#  and without a leading "submission_")  -- a member without its zip is used for the fold2 analysis only.
MEMBERS = [
    ("moe13_adapter_bilstm",   "p2_13_moe_hc_adapter_bilstm_clsw_heldout_fold2_pred.json",   "p2_13_moe_hc_adapter_bilstm_clsw"),
    ("moe14_lorma_bilstm",     "p2_14_moe_hc_lorma_bilstm_clsw_heldout_fold2_pred.json",     "p2_14_moe_hc_lorma_bilstm_clsw"),
    ("moe12_fullft_bilstm",    "p2_12_moe_hc_full_ft_bilstm_clsw_heldout_fold2_pred.json",   "p2_12_moe_hc_full_ft_bilstm_clsw"),
    ("moe09_adapter",          "p2_09_moe_hc_adapter_clsw_heldout_fold2_pred.json",          "p2_09_moe_hc_adapter_clsw"),
    ("moe06_fullft",           "p2_06_moe_hc_full_ft_clsw_heldout_fold2_pred.json",          "p2_06_moe_hc_full_ft_clsw"),
    ("gqa_adapter_bilstm",     "p2s_gqa160__adapter_bilstm__fold1tofold2_pred.json",         "p2s_gqa160_adapter_bilstm"),
    ("gqa_pfeiffer_bilstm",    "p2s_gqa160__adapter_pfeiffer_bilstm__fold1tofold2_pred.json", "p2s_gqa160_adapter_pfeiffer_bilstm"),
    ("gqa_lorma_bilstm",       "p2s_gqa160__lorma_bilstm__fold1tofold2_pred.json",           "p2s_gqa160_lorma_bilstm"),
    ("gqa_lorma_pi_bilstm",    "p2s_gqa160__lorma_pi_bilstm__fold1tofold2_pred.json",        "p2s_gqa160_lorma_pi_bilstm"),
    ("lstm_partial_top1",      "p2s_lstm84__partial_ft_top1_bilstm__fold1tofold2_pred.json", "p2s_lstm84_partial_ft_top1_bilstm"),
    ("lstm_adapter_bilstm",    "p2s_lstm84__adapter_bilstm__fold1tofold2_pred.json",         "p2s_lstm84_adapter_bilstm"),
    ("lstm_pfeiffer_bilstm",   "p2s_lstm84__adapter_pfeiffer_bilstm__fold1tofold2_pred.json", "p2s_lstm84_adapter_pfeiffer_bilstm"),
    ("lstm_frozen_bilstm",     "p2s_lstm84__frozen_bilstm__fold1tofold2_pred.json",          "p2s_lstm84_frozen_bilstm"),
    ("kd15b_unlabelled",       "p2_15b_kd_moe13_to_lstm_plus_unlabelled_heldout_fold2_pred.json", "p2_15b_kd_moe13_to_lstm_plus_unlabelled"),
    ("kd15a_labelled",         "p2_15a_kd_moe13_to_lstm_labelled_only_heldout_fold2_pred.json",   "p2_15a_kd_moe13_to_lstm_labelled_only"),
    ("mor_lorma_pi_bilstm",    "p2s_mor2__lorma_pi_bilstm__fold1tofold2_pred.json",          "p2s_mor2_lorma_pi_bilstm"),
    ("mor_bitfit_bilstm",      "p2s_mor2__bitfit_bilstm__fold1tofold2_pred.json",            "p2s_mor2_bitfit_bilstm"),
    ("gdn_adapter_bilstm",     "p2s_gdn__adapter_bilstm__fold1tofold2_pred.json",            "p2s_gdn_adapter_bilstm"),
]
# member groups tried besides "the best N by fold2 F1" (keys from MEMBERS)
GROUPS = {
    "moe_only":      ["moe13_adapter_bilstm", "moe14_lorma_bilstm", "moe12_fullft_bilstm", "moe09_adapter", "moe06_fullft"],
    "one_per_model": ["moe13_adapter_bilstm", "gqa_adapter_bilstm", "lstm_partial_top1", "mor_lorma_pi_bilstm", "gdn_adapter_bilstm"],
    "bilstm_top":    ["moe13_adapter_bilstm", "moe14_lorma_bilstm", "moe12_fullft_bilstm", "gqa_adapter_bilstm", "lstm_partial_top1", "lstm_adapter_bilstm"],
}
TOP_N_LIST = [3, 5, 8, 12, 99]                      # "the best N members by fold2 F1" (99 = all)
TAUS = [round(0.15 + 0.05 * i, 2) for i in range(14)]   # a span is kept when its (weighted) vote share >= tau
MODES = ["exact", "boundary"]                        # exact = vote on (start, end, label); boundary = vote on (start, end), label = most voted
WEIGHTS = ["equal", "f1"]                            # every member counts 1, or in proportion to its fold2 F1
DOT_RULES = [False, True]                            # normalise abbreviation periods ("Ltd" vs "Ltd.") in every member before voting
INPUT_MODE = "none"
'''

ENS_LOAD = '''# ---- fold2 predictions of every member (fold1-trained models) --------------------------------------------------------
GOLD2 = {d: [tuple(e[:3]) for e in GOLDS["fold2"][d]] for d in GOLDS["fold2"]}
FOLD2 = {}
for key, fname, zname in MEMBERS:
    try:
        p = json.load(open(fetch_model_file(fname)))
        assert set(p) >= set(GOLD2), "missing documents"
        FOLD2[key] = {d: [tuple(x[:3]) for x in p[d]] for d in GOLD2}
    except Exception as ex:
        print(f"  skipped {key}: {type(ex).__name__}: {str(ex)[:100]}")
SCORES = {k: strict_scores(GOLD2, p, LABELS) for k, p in FOLD2.items()}
print(f"{len(FOLD2)} members loaded; single-model strict macro-F1 on fold2:")
for k in sorted(SCORES, key=lambda k: -SCORES[k]["macro_F1"]):
    s = SCORES[k]
    print(f"  {k:24s} F1 {s['macro_F1']:.4f}  P {s['macro_P']:.4f}  R {s['macro_R']:.4f}  predicted spans {sum(len(v) for v in FOLD2[k].values())}")
BEST_SINGLE = max(SCORES, key=lambda k: SCORES[k]["macro_F1"])
print("best single model:", BEST_SINGLE, round(SCORES[BEST_SINGLE]["macro_F1"], 4))
'''

ENS_CORE = '''import re
from bisect import bisect_left
from collections import defaultdict, Counter


def learn_dot_rule(texts, golds):
    """For every gold entity count whether the entity INCLUDES a final period or stops just before one, per
    (label, last word) and per (label, word shape). Learned on training documents only."""
    inc, exc = Counter(), Counter()
    for did, ents in golds.items():
        t = texts[did]
        for e in ents:
            s, en, lab = e[0], e[1], e[2]
            body = t[s:en]
            if body.endswith("."):
                m = re.search(r"([A-Za-z]+)\\.$", body); kind = inc
                word = m.group(1) if m else ""
            elif t[en:en + 1] == ".":
                m = re.search(r"([A-Za-z]+)$", body); kind = exc
                word = m.group(1) if m else ""
            else:
                continue
            if not word:
                continue
            kind[(lab, word.lower())] += 1
            kind[(lab, "shape:" + ("short" if len(word) <= 3 else "long") + ("cap" if word[0].isupper() else "low"))] += 1
    return inc, exc


def apply_dot_rule(texts, preds, rule):
    inc, exc = rule
    out = {}
    for did, ents in preds.items():
        t, new = texts[did], set()
        for (s, e, lab) in ents:
            if not t[e - 1:e] == "." and t[e:e + 1] == ".":
                m = re.search(r"([A-Za-z]+)$", t[s:e])
                if m:
                    w = m.group(1)
                    k1, k2 = (lab, w.lower()), (lab, "shape:" + ("short" if len(w) <= 3 else "long") + ("cap" if w[0].isupper() else "low"))
                    if inc[k1] + exc[k1] >= 2:
                        go = inc[k1] > exc[k1]
                    elif inc[k2] + exc[k2] >= 3:
                        go = inc[k2] > exc[k2]
                    else:
                        go = False
                    if go:
                        e += 1
            new.add((s, e, lab))
        out[did] = sorted(new)
    return out


def vote(pred_list, weights, tau, mode, doc_ids):
    W = float(sum(weights))
    out = {}
    for d in doc_ids:
        cnt = defaultdict(float)
        for p, w in zip(pred_list, weights):
            for (s, e, l) in set(p.get(d, [])):
                cnt[(s, e, l)] += w
        if mode == "exact":
            cands = [(v, k) for k, v in cnt.items() if v >= tau * W - 1e-9]
        else:
            by = defaultdict(dict)
            for (s, e, l), v in cnt.items():
                by[(s, e)][l] = v
            cands = []
            for (s, e), dl in by.items():
                sup = sum(dl.values())
                lab = max(dl, key=lambda x: (dl[x], x))
                if sup >= tau * W - 1e-9:
                    cands.append((dl[lab] + 1e-3 * sup, (s, e, lab)))
        cands.sort(key=lambda t: (-t[0], -(t[1][1] - t[1][0]), t[1][0], t[1][2]))
        starts, ends, res = [], [], []                 # accepted intervals, non-overlapping (the gold has no overlaps)
        for v, (s, e, l) in cands:
            if any(not (e <= ts or s >= te) for ts, te in zip(starts, ends)):
                continue
            starts.append(s); ends.append(e); res.append((s, e, l))
        out[d] = sorted(res)
    return out
'''

ENS_GRID = '''# ---- search the ensemble settings on fold2 ----------------------------------------------------------------------------
DOT1 = learn_dot_rule(TEXTS["fold1"], GOLDS["fold1"])                    # learned on fold1 only (fold2 is the evaluation fold)
FOLD2_NORM = {k: apply_dot_rule(TEXTS["fold2"], p, DOT1) for k, p in FOLD2.items()}
print("abbreviation-period normalisation on single models (fold2 F1 before -> after):")
for k in sorted(SCORES, key=lambda k: -SCORES[k]["macro_F1"])[:6]:
    print(f"  {k:24s} {SCORES[k]['macro_F1']:.4f} -> {strict_scores(GOLD2, FOLD2_NORM[k], LABELS)['macro_F1']:.4f}")

order = sorted(FOLD2, key=lambda k: -SCORES[k]["macro_F1"])
SETS = {f"top{n}": order[:n] for n in TOP_N_LIST if n <= len(order) or n == 99}
SETS = {("all" if n.endswith("99") else n): v for n, v in SETS.items()}
for g, ks in GROUPS.items():
    ks = [k for k in ks if k in FOLD2]
    if len(ks) >= 2:
        SETS[g] = ks
DOC_IDS = sorted(GOLD2)
HALF = {0: DOC_IDS[0::2], 1: DOC_IDS[1::2]}


def member_weights(keys, wmode):
    return [1.0] * len(keys) if wmode == "equal" else [SCORES[k]["macro_F1"] for k in keys]


def run_config(cfg, docs):
    setname, mode, wmode, tau, dot = cfg
    keys = SETS[setname]
    src = FOLD2_NORM if dot else FOLD2
    return vote([src[k] for k in keys], member_weights(keys, wmode), tau, mode, docs)


def score_docs(preds, docs):
    return strict_scores({d: GOLD2[d] for d in docs}, preds, LABELS)["macro_F1"]


CONFIGS = [(s, m, w, t, dt) for s in SETS for m in MODES for w in WEIGHTS for t in TAUS for dt in DOT_RULES]
t0 = time.time()
GRID = []
for cfg in CONFIGS:
    p = run_config(cfg, DOC_IDS)
    GRID.append((strict_scores(GOLD2, p, LABELS)["macro_F1"], cfg, score_docs(p, HALF[0]), score_docs(p, HALF[1])))
GRID.sort(key=lambda r: -r[0])
print(f"{len(CONFIGS)} configurations evaluated in {time.time()-t0:.0f}s. Best 15 on fold2 (set, mode, weights, tau, dot-rule):")
for f, cfg, h0, h1 in GRID[:15]:
    print(f"  {f:.4f}  (half A {h0:.4f}, half B {h1:.4f})  {cfg}")
print("\\nbest per member set:")
for s in SETS:
    r = next(r for r in GRID if r[1][0] == s)
    print(f"  {s:14s} {r[0]:.4f}  {r[1][1:]}  ({len(SETS[s])} members)")
print("\\nbest per mode / weighting / dot rule:")
for idx, name in ((1, "mode"), (2, "weights"), (4, "dot rule")):
    for v in sorted({r[1][idx] for r in GRID}, key=str):
        r = next(r for r in GRID if r[1][idx] == v)
        print(f"  {name} = {str(v):9s} best {r[0]:.4f}  {r[1]}")

# honest estimate: choose the configuration on one half of the fold2 documents, apply it to the other half
honest = {}
for a in (0, 1):
    other = 1 - a
    best_cfg = max(CONFIGS, key=lambda c: score_docs(run_config(c, HALF[other]), HALF[other]))
    honest.update(run_config(best_cfg, HALF[a]))
    print(f"  half {a}: configuration chosen on half {other}: {best_cfg}")
HONEST_F1 = strict_scores(GOLD2, honest, LABELS)["macro_F1"]
print(f"\\nHONEST estimate of the tuned ensemble (configuration picked on the other half of the documents): {HONEST_F1:.4f}"
      f"  vs best single model {SCORES[BEST_SINGLE]['macro_F1']:.4f}  vs tuned-on-all-of-fold2 {GRID[0][0]:.4f}")
'''

ENS_REPORT = '''# ---- the chosen ensemble: report -------------------------------------------------------------------------------------
BEST_CFG = GRID[0][1]
bp = run_config(BEST_CFG, DOC_IDS)
S = strict_scores(GOLD2, bp, LABELS)
S1 = SCORES[BEST_SINGLE]
print("chosen configuration:", BEST_CFG, "| members:", SETS[BEST_CFG[0]])
print(f"fold2 strict macro-F1 {S['macro_F1']:.4f} (P {S['macro_P']:.4f}, R {S['macro_R']:.4f})  vs best single {BEST_SINGLE} {S1['macro_F1']:.4f} (P {S1['macro_P']:.4f}, R {S1['macro_R']:.4f})")
print(f"{'label':10s} {'ens F1':>7s} {'single F1':>9s} {'ens P':>6s} {'ens R':>6s}")
for l in LABELS:
    print(f"{l:10s} {S['per_label'][l]['F1']:7.3f} {S1['per_label'][l]['F1']:9.3f} {S['per_label'][l]['P']:6.3f} {S['per_label'][l]['R']:6.3f}")
print("errors, ensemble:", error_breakdown(GOLD2, bp))
print("errors, best single:", error_breakdown(GOLD2, FOLD2[BEST_SINGLE]))
print("predicted spans: ensemble", sum(len(v) for v in bp.values()), "| best single", sum(len(v) for v in FOLD2[BEST_SINGLE].values()),
      "| gold", sum(len(v) for v in GOLD2.values()))
# distinct configurations worth a second submission: the best one that uses a different member set
SECOND = next((r for r in GRID if r[1][0] != BEST_CFG[0]), None)
print("best configuration with a different member set:", SECOND[:2] if SECOND else None)
'''

ENS_TEST = '''# ---- test predictions: ensemble the members' submission zips ---------------------------------------------------------
import glob, re
ZIPS = {}
for z in (glob.glob("/kaggle/input/**/*.zip", recursive=True) + glob.glob("/kaggle/working/*.zip")
          + glob.glob("/kaggle/input/**/*.json", recursive=True)):
    name = os.path.basename(z).rsplit(".", 1)[0]
    if name == "prediction":                                   # Kaggle may unzip an uploaded zip into a folder named after it
        name = os.path.basename(os.path.dirname(z))
    ZIPS[name[len("submission_"):] if name.startswith("submission_") else name] = z
print(f"{len(ZIPS)} zip files found under /kaggle/input:", sorted(ZIPS))
TEST_PRED = {}
def find_zip(zname):
    """Exact name first; else a zip whose name is a truncated prefix of it (Codabench shortens long file names to ~35 chars,
    e.g. 'p2_15b_kd_moe13_to_lstm_plus_unlabe'); a trailing ' (1)' / '_1' from repeated downloads is ignored."""
    if zname in ZIPS:
        return ZIPS[zname]
    cands = []
    for k, path in ZIPS.items():
        base = re.sub(r"(\s*\(\d+\)|_\d+)$", "", k)
        if base == zname or (len(base) >= 30 and zname.startswith(base)):
            cands.append((len(base), path))
    return max(cands)[1] if cands else None


for key, fname, zname in MEMBERS:
    zpath = find_zip(zname)
    if zpath is not None:
        print(f"  {key}: using {zpath.replace('/kaggle/input/', '')}")
        if zpath.endswith(".json"):                       # a bare prediction.json renamed after its model
            p = json.load(open(zpath))
        else:
            with zipfile.ZipFile(zpath) as zf:
                p = json.loads(zf.read("prediction.json"))
        assert set(p) == set(TEST_TEXTS), f"{zname}: document ids differ from the test set"
        TEST_PRED[key] = {d: [tuple(x[:3]) for x in p[d]] for d in p}
missing = [k for k, _, z in MEMBERS if k in FOLD2 and k not in TEST_PRED]
print("members with a test prediction:", len(TEST_PRED), "| missing zips for:", missing)

DOT_ALL = learn_dot_rule({**TEXTS["fold1"], **TEXTS["fold2"]}, {**GOLDS["fold1"], **GOLDS["fold2"]})


def validate_prediction(pred, texts, labels):
    assert set(pred) == set(texts), f"id mismatch: missing {len(set(texts)-set(pred))}, extra {len(set(pred)-set(texts))}"
    for did, ents in pred.items():
        seen = set()
        for x in ents:
            s, e, l = x
            assert type(s) is int and type(e) is int and 0 <= s < e <= len(texts[did]) and l in labels, (did, x)
            assert (s, e, l) not in seen, ("duplicate", did, x); seen.add((s, e, l))


def make_zip(cfg, tag):
    setname, mode, wmode, tau, dot = cfg
    keys = [k for k in SETS[setname] if k in TEST_PRED]
    if len(keys) < 2:
        print(f"  {tag}: only {len(keys)} member(s) with a test prediction -> not written"); return None
    src = {k: (apply_dot_rule(TEST_TEXTS, TEST_PRED[k], DOT_ALL) if dot else TEST_PRED[k]) for k in keys}
    # the vote threshold is a SHARE of the total weight, so it is unchanged when a member has no zip
    pred = vote([src[k] for k in keys], member_weights(keys, wmode), tau, mode, sorted(TEST_TEXTS))
    pred = {d: [[int(s), int(e), l] for (s, e, l) in v] for d, v in pred.items()}
    validate_prediction(pred, TEST_TEXTS, LABELS)
    with open(f"{WORK}/prediction.json", "w", encoding="utf-8") as f:
        json.dump(pred, f)
    zp = f"{WORK}/submission_{RUN_NAME}_{tag}.zip"
    with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED) as z:
        z.write(f"{WORK}/prediction.json", arcname="prediction.json")
    with zipfile.ZipFile(zp) as z:
        assert z.namelist() == ["prediction.json"]
    n = sum(len(v) for v in pred.values())
    print(f"  wrote {zp}: {n} entities, members {keys}, config {cfg[1:]}")
    return dict(zip=os.path.basename(zp), config=list(cfg), members=keys, entities=n)


WRITTEN = [w for w in (make_zip(BEST_CFG, "A"), make_zip(SECOND[1], "B") if SECOND else None) if w]
'''

ENS_SAVE = '''RESULT = dict(run=RUN_NAME, members_fold2={k: round(v["macro_F1"], 4) for k, v in SCORES.items()}, best_single=BEST_SINGLE,
              best_config=list(BEST_CFG), best_fold2_F1=round(S["macro_F1"], 4), honest_F1=round(HONEST_F1, 4),
              per_label_F1={l: round(S["per_label"][l]["F1"], 4) for l in LABELS}, errors=error_breakdown(GOLD2, bp),
              top_grid=[(round(r[0], 4), list(r[1])) for r in GRID[:30]], written=WRITTEN,
              notebook_minutes=round((time.time() - NB_START) / 60, 1))
rp = f"{WORK}/{RUN_NAME}_results.json"
json.dump(RESULT, open(rp, "w"), indent=1, default=str)
upload_model_file(rp)
print("=" * 100)
print(f"RESULT {RUN_NAME}: fold2 {RESULT['best_fold2_F1']} (honest {RESULT['honest_F1']}) vs best single {BEST_SINGLE} {round(SCORES[BEST_SINGLE]['macro_F1'], 4)}; zips: {[w['zip'] for w in WRITTEN]}")
print("=" * 100)
print("SUMMARY_JSON_START"); print(json.dumps(RESULT, default=str)); print("SUMMARY_JSON_END")
'''


def build_ens():
    cells = []
    md = lambda s: cells.append(nbf.v4.new_markdown_cell(s))
    code = lambda s: cells.append(nbf.v4.new_code_cell(s))
    md(f"""# CS-781 Phase 2 (L-NER) — `{ENS_NAME}`: span-voting ensemble of the models already trained

No new training and **no GPU needed** (set the accelerator to *None*). The notebook takes the entity spans predicted by several of your models and keeps a span when enough of them agree:
* **fold2 check** — the fold2 predictions of the fold1-trained models are in your bucket; the notebook scores every single model and every ensemble setting (which models, vote threshold, exact-span voting vs boundary voting, weights, abbreviation-period normalisation) and reports an **honest estimate** (setting chosen on half of the fold2 documents, scored on the other half);
* **test predictions** — built from the `prediction.json` inside the submission zips of the fold1+fold2 refits, so you must attach them (see below).

**Before running — attach your zips:** on Kaggle, *Add Data → Upload → New dataset*, upload the submission `.zip` files you already made (MoE p2_06/09/12/13/14, GQA, LSTM, MoR, GDN ones; the more the better, a member without a zip is only used for the fold2 analysis) and add that dataset to this notebook. File names may be the original ones (`submission_p2s_gqa160_adapter_bilstm.zip`) or the Codabench ones (`p2s_gqa160_adapter_bilstm.zip`).

**Kaggle:** secrets `HF`, `HF_WRITE`; Accelerator *None* is enough. Takes a few minutes. Writes up to two zips (`..._A` = best setting, `..._B` = best setting with a different member set).""")
    md("## 1. Settings")
    code(ENS_SETTINGS)
    md("## 2. Setup: login, bucket")
    code(SETUP_CELL)
    md("## 3. Phase-2 data")
    code(DATA_CELL)
    md("## 4. Strict scorer and error breakdown (same code as in the other notebooks)")
    code(ALIGN_SRC)
    md("## 5. Voting, abbreviation-period normalisation")
    code(ENS_CORE)
    md("## 6. Load the members' fold2 predictions")
    code(ENS_LOAD)
    md("## 7. Search the ensemble settings on fold2")
    code(ENS_GRID)
    md("## 8. Report of the chosen ensemble")
    code(ENS_REPORT)
    md("## 9. Test predictions from the zips, submission zips")
    code(ENS_TEST)
    md("## 10. Save and print the result block")
    code(ENS_SAVE)
    nb = nbf.v4.new_notebook()
    nb["cells"] = cells
    nb["metadata"] = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                      "language_info": {"name": "python"}}
    nbf.validate(nb)
    path = OUT + ENS_NAME + ".ipynb"
    nbf.write(nb, path)
    return path




def _members_block(rows):
    return "MEMBERS = [\n" + "".join(f'    ("{k}", "{f}", "{z}"),\n' for k, f, z in rows) + "]\n"


def build_ens_variants():
    """Three focused ensembles: (a) MoE variants only, (b) several architectures, (c) MoE variants + the distilled student."""
    import re, os
    ALL = {m[0]: m for m in re.findall(r'\("(\w+)",\s*"([^"]+)",\s*"([^"]+)"\)', ENS_SETTINGS)}
    variants = [
        ("p2_16a_ensemble_moe_only", ["moe13_adapter_bilstm", "moe14_lorma_bilstm", "moe12_fullft_bilstm", "moe09_adapter"],
         "GROUPS = {\n    \"moe_best3\": [\"moe13_adapter_bilstm\", \"moe14_lorma_bilstm\", \"moe12_fullft_bilstm\"],\n}\nTOP_N_LIST = [2, 3, 99]",
         "**Variant A: MoE variants only** (adapter, LoRMA, full fine-tuning on the same Phase-1 MoE backbone; p2_09 is a linear-head adapter run). Tests whether averaging different adaptation methods of one backbone helps."),
        ("p2_16b_ensemble_cross_arch", ["moe13_adapter_bilstm", "moe14_lorma_bilstm", "gqa_adapter_bilstm", "gqa_pfeiffer_bilstm",
                                         "lstm_partial_top1", "lstm_adapter_bilstm", "kd15b_unlabelled", "kd15a_labelled"],
         "GROUPS = {\n    \"one_per_model\": [\"moe13_adapter_bilstm\", \"gqa_adapter_bilstm\", \"kd15b_unlabelled\"],\n"
         "    \"two_per_model\": [\"moe13_adapter_bilstm\", \"moe14_lorma_bilstm\", \"gqa_adapter_bilstm\", \"gqa_pfeiffer_bilstm\", \"kd15b_unlabelled\", \"lstm_partial_top1\"],\n"
         "    \"no_lstm_family\": [\"moe13_adapter_bilstm\", \"moe14_lorma_bilstm\", \"gqa_adapter_bilstm\", \"gqa_pfeiffer_bilstm\"],\n}\nTOP_N_LIST = [3, 4, 5, 99]",
         "**Variant B: cross-architecture** (MoE, GQA and LSTM-family models, including the two distilled students p2_15a/p2_15b). The grid compares `one_per_model` (3 members) with `two_per_model` (6 members) and the top-N sets, plus a `no_lstm_family` group, so one run answers whether a second model per architecture helps and whether the distilled LSTM student adds anything to MoE + GQA."),
        ("p2_16c_ensemble_moe_plus_student", ["moe13_adapter_bilstm", "moe14_lorma_bilstm", "moe12_fullft_bilstm", "kd15b_unlabelled"],
         "GROUPS = {\n    \"moe2_plus_student\": [\"moe13_adapter_bilstm\", \"moe14_lorma_bilstm\", \"kd15b_unlabelled\"],\n"
         "    \"moe3_only\": [\"moe13_adapter_bilstm\", \"moe14_lorma_bilstm\", \"moe12_fullft_bilstm\"],\n"
         "    \"teacher_plus_student\": [\"moe13_adapter_bilstm\", \"kd15b_unlabelled\"],\n}\nTOP_N_LIST = [2, 3, 4]",
         "**Variant C: MoE + its distilled student** (p2_13, p2_14, p2_12 plus the KD student p2_15b that was distilled from p2_13). Compares `moe3_only` with `moe2_plus_student` and `teacher_plus_student`, so it shows directly whether the distilled LSTM student adds anything on top of the MoE variants."),
    ]
    out = []
    base = build_ens()
    for name, keys, groups, note in variants:
        nb = nbf.read(base, as_version=4)
        for c in nb["cells"]:
            src = c["source"]
            if c["cell_type"] == "code" and "ENSEMBLE SETTINGS" in src:
                src = src.replace(ENS_NAME, name)
                src = re.sub(r"MEMBERS = \[.*?\n\]\n", lambda m: _members_block([ALL[k] for k in keys]), src, count=1, flags=re.S)
                src = re.sub(r"GROUPS = \{.*?\n\}\n", lambda m: "", src, count=1, flags=re.S)
                src = re.sub(r"TOP_N_LIST = \[[^\]]*\]", lambda m: groups, src, count=1)
                c["source"] = src
            elif c["cell_type"] == "markdown":
                c["source"] = src.replace(ENS_NAME, name)
        nb["cells"][0]["source"] = nb["cells"][0]["source"].replace("\n\nNo new training", f"\n\n{note}\n\nNo new training", 1)
        nbf.validate(nb)
        path = OUT + name + ".ipynb"
        nbf.write(nb, path)
        out.append(path)
    os.remove(base)
    return out


if __name__ == "__main__":
    print(build_ens_variants())
