"""Builds phase2_lner_step0_data_alignment_check.ipynb (CPU-only Kaggle notebook)."""
import nbformat as nbf

BPE_SRC = open("/home/claude/p2/bpe_class.py").read().replace("import json\n", "", 1)
ALIGN_SRC = open("/home/claude/p2/lner_align.py").read()
EVAL_SRC = open("/home/claude/p2/evaluate_phase2.py").read()

cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s))
code = lambda s: cells.append(nbf.v4.new_code_cell(s))

md("""# CS-781 Phase 2 (L-NER) — Step 0: data + tokenizer alignment check (no training, CPU is enough)

**What this notebook does** (≈5–10 min, no GPU needed — set *Accelerator: None*):
1. Downloads the Phase-2 data (`Phase_2_LNER/`) and **your Phase-1 BPE tokenizer** from the bucket `kavincg23/cs781-models`.
2. Checks the data format (offsets, labels, overlaps, the empty test document).
3. Builds **character offsets for every BPE token** and proves the token ids are *identical* to `BPETokenizer.encode` (so the Phase-1 models see exactly what they were trained on).
4. Measures the **strict-F1 ceiling** of token-level tagging: if a model predicted every token's tag perfectly, what strict macro-F1 would it get? Three ways of feeding text to the tokenizer are compared, each with and without a punctuation-trimming repair learned on the *other* fold.
5. Counts tokens and sliding windows per document.
6. Runs the **official `evaluate_phase2.py`** on gold-vs-gold (must be 1.0) and on the oracle predictions (must equal our own scorer).
7. Saves `phase2_alignment_report.json` and uploads it to the bucket (for the report).

Nothing here trains or submits anything. The numbers it prints decide one design choice for all later notebooks (see the last cell).

Secrets needed (same as Phase 1): **`HF`** (read, gated dataset) and **`HF_WRITE`** (bucket).""")

md("## 1. Settings")
code("""HF_DATASET_REPO = "Exploration-Lab/CS-781-Capstone"
P2_DIR = "Phase_2_LNER"
TRAIN_FILES = {f"fold{i}": (f"{P2_DIR}/Train/fold{i}-text.json", f"{P2_DIR}/Train/fold{i}-gold.json") for i in (1, 2)}
LABELS_FILE = f"{P2_DIR}/Train/labels.txt"
TEST_FILE = f"{P2_DIR}/Test/fold3-text.json"

HF_BUCKET = "kavincg23/cs781-models"
HF_CHECKPOINT_REPO = "kavincg23/cs781-checkpoints"   # old repo, read-only fallback
TOKENIZER_NAME = "a1bpe_tokenizer_160Mw_1ep_full_v40000_160000000w_seed42_minf1.json"   # shared by every Phase-1 model
WORK = "/kaggle/working"

# window sizes to report (tokens); the Phase-1 models were trained on 2048-token chunks
WINDOW_SETTINGS = [(2048, 1024), (2048, 1536), (1024, 512), (512, 256)]
REPORT_PATH = f"{WORK}/phase2_alignment_report.json"
""")

md("## 2. Install, Hugging Face login, bucket helper")
code("""!pip install -q -U "huggingface_hub>=1.5"
import os, json, shutil, subprocess, sys, time
from collections import Counter, defaultdict
from huggingface_hub import hf_hub_download, login, download_bucket_files, batch_bucket_files
from kaggle_secrets import UserSecretsClient

user_secrets = UserSecretsClient()
HF_TOKEN = user_secrets.get_secret("HF")
HF_WRITE_TOKEN = user_secrets.get_secret("HF_WRITE")
os.environ["HF_TOKEN"] = HF_TOKEN
login(token=HF_TOKEN)


def get_dataset_file(filename):
    return hf_hub_download(repo_id=HF_DATASET_REPO, filename=filename, repo_type="dataset", token=HF_TOKEN)


def fetch_model_file(name, local_path=None):
    local_path = local_path or f"{WORK}/{name}"
    if os.path.exists(local_path):
        return local_path
    part = local_path + ".part"
    try:
        download_bucket_files(HF_BUCKET, files=[(name, part)], raise_on_missing_files=True, token=HF_WRITE_TOKEN)
        os.replace(part, local_path)
        return local_path
    except Exception:
        if os.path.exists(part):
            os.remove(part)
    # fallback: the OLD model repo (the Phase-1 notebooks do the same; the tokenizer only lives there)
    p = hf_hub_download(repo_id=HF_CHECKPOINT_REPO, filename=name, repo_type="model", token=HF_WRITE_TOKEN)
    shutil.copy(p, local_path)
    print(f"(found {name} in the old repo {HF_CHECKPOINT_REPO})")
    return local_path


def upload_model_file(local_path, name=None):
    name = name or os.path.basename(local_path)
    batch_bucket_files(HF_BUCKET, add=[(local_path, name)], token=HF_WRITE_TOKEN)
    print(f"Uploaded {name} to bucket {HF_BUCKET}")

print("Logged in; bucket:", HF_BUCKET)
""")

md("## 3. Load the Phase-2 data")
code("""def load_json(p):
    with open(p, encoding="utf-8") as f:
        return json.load(f)

LABELS = [l.strip() for l in open(get_dataset_file(LABELS_FILE), encoding="utf-8") if l.strip()]
TEXTS, GOLDS = {}, {}
for fold, (tf, gf) in TRAIN_FILES.items():
    TEXTS[fold] = load_json(get_dataset_file(tf))
    GOLDS[fold] = load_json(get_dataset_file(gf))
TEST_TEXTS = load_json(get_dataset_file(TEST_FILE))

print("labels:", LABELS)
for fold in TEXTS:
    n_ent = sum(len(v) for v in GOLDS[fold].values())
    print(f"{fold}: {len(TEXTS[fold])} docs, {n_ent} entities")
print(f"test: {len(TEST_TEXTS)} docs")
""")

md("## 4. Data format checks")
code("""report = {"data": {}}
for fold in TEXTS:
    T, G = TEXTS[fold], GOLDS[fold]
    assert set(T) == set(G), f"{fold}: text/gold ids differ"
    cnt, mismatch, overlaps, ws_edge = Counter(), 0, 0, 0
    for did, ents in G.items():
        t = T[did]
        for ent in ents:
            s, e, lab = ent[0], ent[1], ent[2]
            assert lab in LABELS, lab
            cnt[lab] += 1
            if len(ent) > 3 and t[s:e] != ent[3]:
                mismatch += 1
            if t[s:e] != t[s:e].strip():
                ws_edge += 1
        es = sorted((x[0], x[1]) for x in ents)
        overlaps += sum(1 for (a, b), (c, d) in zip(es, es[1:]) if c < b)
    words = sorted(len(t.split()) for t in T.values())
    report["data"][fold] = dict(docs=len(T), entities=sum(cnt.values()), per_label=dict(cnt), text_offset_mismatch=mismatch,
                                overlapping_pairs=overlaps, spans_with_edge_whitespace=ws_edge,
                                words_min=words[0], words_median=words[len(words)//2], words_max=words[-1])
    print(fold, json.dumps(report["data"][fold]))
empty_test = [k for k, v in TEST_TEXTS.items() if not v.strip()]
report["data"]["test_empty_docs"] = empty_test
print("test docs with no text (must still be in prediction.json as []):", empty_test)
print("\\n-> end offsets are EXCLUSIVE if text_offset_mismatch == 0; overlapping_pairs == 0 means flat BIO tagging can represent every entity.")
""")

md("## 5. Your Phase-1 BPE tokenizer (class copied unchanged from the Phase-1 notebooks) — loaded from the bucket")
code(BPE_SRC + """

tok_path = fetch_model_file(TOKENIZER_NAME)
bpe_tokenizer = BPETokenizer.load(tok_path)
print("tokenizer:", TOKENIZER_NAME, "| vocab", bpe_tokenizer.get_vocab_size(), "| merges", len(bpe_tokenizer.merges))
""")

md("""## 6. Offset-aware encoding, BIO tags, span decoding, strict scorer

`encode_with_offsets(tok, text, mode)` returns the token ids **and** each token's `(start_char, end_char)` in the raw text.
- `mode="none"`: exactly the Phase-1 tokenization (ids identical to `bpe_tokenizer.encode`).
- `mode="punct"`: each whitespace word is first split into letter/digit runs and single punctuation characters, then BPE-encoded. This changes the ids versus pretraining (e.g. `Haryana,` becomes `Haryana</w>` + `,</w>`) — a trade-off to measure, not a free fix.
- `mode="punct_digit"`: also splits letters from digits (`Pradesh1` -> `Pradesh`, `1`).

Boundary repair: after decoding token spans back to characters, a table learned **only on the training fold** says, for each (label, trailing/leading punctuation char), whether gold entities usually exclude that character; if so it is trimmed.""")
code(ALIGN_SRC)

md("## 7. Token ids are identical to `BPETokenizer.encode` (mode 'none') on every document")
code("""t0 = time.time()
all_docs = {f"{fold}:{k}": v for fold in TEXTS for k, v in TEXTS[fold].items()}
all_docs.update({f"test:{k}": v for k, v in TEST_TEXTS.items()})
check_encode_identical(bpe_tokenizer, all_docs)
print(f"OK: ids identical to BPETokenizer.encode on all {len(all_docs)} documents; spans monotone and inside the text ({time.time()-t0:.0f}s)")
""")

md("## 8. Strict-F1 ceiling of token-level tagging (the 'oracle')\nFor every gold entity: does it start and end exactly on token boundaries? Then gold → token BIO tags → back to character spans → strict score. Repair is learned on the *other* fold (2-fold, no peeking).")
code("""MODES = ["none", "punct", "punct_digit"]
SPANS = {m: {fold: {} for fold in TEXTS} for m in MODES}
NTOK = {m: {} for m in MODES}
for m in MODES:
    for fold in TEXTS:
        for did, t in TEXTS[fold].items():
            ids, sp = encode_with_offsets(bpe_tokenizer, t, m)
            SPANS[m][fold][did] = sp
            NTOK[m][f"{fold}:{did}"] = len(ids)
    for did, t in TEST_TEXTS.items():
        NTOK[m][f"test:{did}"] = len(encode_with_offsets(bpe_tokenizer, t, m)[0])

report["oracle"] = {}
other = {"fold1": "fold2", "fold2": "fold1"}
for m in MODES:
    report["oracle"][m] = {}
    for fold in TEXTS:
        T, G, S = TEXTS[fold], GOLDS[fold], SPANS[m][fold]
        mis, tot = Counter(), Counter()
        pred_raw, pred_rep = {}, {}
        tr = other[fold]
        trail, lead = learn_trim_table(TEXTS[tr], GOLDS[tr], SPANS[m][tr], LABELS)
        for did in G:
            tag, bad = gold_to_token_tags(S[did], G[did], LABELS)
            for e in bad: mis[e[2]] += 1
            for e in G[did]: tot[e[2]] += 1
            raw = token_tags_to_spans(S[did], tag, LABELS)
            pred_raw[did] = finalize_doc_entities(T[did], raw)
            pred_rep[did] = finalize_doc_entities(T[did], raw, trail, lead)
        s_raw, s_rep = strict_scores(G, pred_raw, LABELS), strict_scores(G, pred_rep, LABELS)
        n_words = sum(len(t.split()) for t in T.values())
        n_tok = sum(NTOK[m][f"{fold}:{d}"] for d in T)
        report["oracle"][m][fold] = dict(
            misaligned=sum(mis.values()), entities=sum(tot.values()), misaligned_per_label=dict(mis),
            tokens_per_word=round(n_tok / n_words, 4),
            ceiling_macroF1_no_repair=round(s_raw["macro_F1"], 4), ceiling_macroF1_with_repair=round(s_rep["macro_F1"], 4),
            ceiling_F1_per_label_with_repair={l: round(v["F1"], 4) for l, v in s_rep["per_label"].items()})
        if m == "none" and fold == "fold1":
            ORACLE_FOLD1_NONE = (pred_raw, s_raw)

print(f"{'mode':12s} {'fold':6s} {'tok/word':>8s} {'misaligned':>14s} {'ceiling F1 raw':>15s} {'ceiling F1 +repair':>19s}")
for m in MODES:
    for fold in TEXTS:
        r = report["oracle"][m][fold]
        print(f"{m:12s} {fold:6s} {r['tokens_per_word']:8.3f} {r['misaligned']:6d}/{r['entities']:<6d} "
              f"{r['ceiling_macroF1_no_repair']:15.4f} {r['ceiling_macroF1_with_repair']:19.4f}")
print("\\nper-label ceiling with repair, mode 'none':")
for fold in TEXTS:
    print(" ", fold, report["oracle"]["none"][fold]["ceiling_F1_per_label_with_repair"])
""")

md("### Examples of entities that do not fall on token boundaries (mode 'none')")
code("""shown = 0
T, G, S = TEXTS["fold1"], GOLDS["fold1"], SPANS["none"]["fold1"]
for did in G:
    _, bad = gold_to_token_tags(S[did], G[did], LABELS)
    for s, e, lab, *_ in bad[:2]:
        toks = [T[did][a:b] for a, b in S[did] if a < e + 3 and b > s - 1]
        print(f"{lab:10s} gold={T[did][s:e]!r:45s} tokens={toks}")
        shown += 1
    if shown >= 20:
        break
""")

md("## 9. Tokens and sliding windows per document")
code("""report["windows"] = {}
for m in MODES:
    for split in ("fold1", "fold2", "test"):
        n = sorted(v for k, v in NTOK[m].items() if k.startswith(split + ":"))
        row = dict(tokens_min=n[0], tokens_median=n[len(n)//2], tokens_max=n[-1])
        for w, s in WINDOW_SETTINGS:
            row[f"windows_{w}_{s}"] = sum(len(make_windows(x, w, s)) for x in n)
        report["windows"][f"{m}:{split}"] = row
        print(f"{m:12s} {split:6s}", row)
""")

md("## 10. Official evaluator: gold vs gold must be 1.0; oracle must match our scorer")
code("%%writefile /kaggle/working/evaluate_phase2.py\n" + EVAL_SRC)
code("""def write_json(obj, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False)

def validate_prediction(pred, texts, labels):
    \"\"\"The checks Codabench applies: same ids, list of [start,end,label], ints, 0<=start<end<=len(text), known label, no duplicates.\"\"\"
    assert set(pred) == set(texts), f"id mismatch: missing {len(set(texts)-set(pred))}, extra {len(set(pred)-set(texts))}"
    for did, ents in pred.items():
        assert isinstance(ents, list)
        seen = set()
        for x in ents:
            assert isinstance(x, list) and len(x) in (3, 4)
            s, e, l = x[:3]
            assert type(s) is int and type(e) is int and 0 <= s < e <= len(texts[did]) and l in labels, (did, x)
            assert (s, e, l) not in seen, ("duplicate", did, x); seen.add((s, e, l))
    return True

gold_f1 = {k: [x[:3] for x in v] for k, v in GOLDS["fold1"].items()}
write_json(GOLDS["fold1"], f"{WORK}/truth_fold1.json")
write_json(gold_f1, f"{WORK}/pred_gold_fold1.json")
pred_oracle, s_oracle = ORACLE_FOLD1_NONE
validate_prediction(pred_oracle, TEXTS["fold1"], LABELS)
write_json(pred_oracle, f"{WORK}/pred_oracle_fold1.json")
with open(f"{WORK}/labels.txt", "w") as f:
    f.write("\\n".join(LABELS) + "\\n")
for name in ("pred_gold_fold1.json", "pred_oracle_fold1.json"):
    r = subprocess.run([sys.executable, f"{WORK}/evaluate_phase2.py", "--truth", f"{WORK}/truth_fold1.json",
                        "--prediction", f"{WORK}/{name}", "--labels", f"{WORK}/labels.txt"], capture_output=True, text=True)
    print(f"--- {name}\\n" + r.stdout.strip().splitlines()[-3] if r.returncode == 0 else r.stderr)
print(f"our scorer on the oracle: {s_oracle['macro_F1']:.6f}  (must equal the official Strict Macro-F1 line above)")

empty_pred = {k: [] for k in TEST_TEXTS}
validate_prediction(empty_pred, TEST_TEXTS, LABELS)
print("format validator OK on an all-empty fold3 prediction (including the empty document). NOT a submission.")
""")

md("## 11. Save the report (for the final write-up) and upload it to the bucket")
code("""report["tokenizer"] = TOKENIZER_NAME
report["created"] = time.strftime("%Y-%m-%d %H:%M:%S")
write_json(report, REPORT_PATH)
try:
    upload_model_file(REPORT_PATH)
except Exception as e:
    print("upload failed (the file is still in /kaggle/working):", type(e).__name__, e)
print(json.dumps({m: {f: (report['oracle'][m][f]['ceiling_macroF1_no_repair'], report['oracle'][m][f]['ceiling_macroF1_with_repair'])
                      for f in TEXTS} for m in MODES}, indent=1))
""")

md("""## 12. What to read from this, and the decision it feeds

- **Section 7** must say `OK`. If it fails, stop and send me the error: the offsets would be wrong for every model.
- **Section 8** is the key table. It is the best strict macro-F1 any *token-level* model could reach with each input mode:
  - `none` keeps the Phase-1 tokens exactly (best transfer from pretraining), but many entity ends sit inside a token like `Court,`; the learned repair recovers most of them.
  - `punct` / `punct_digit` make almost every boundary reachable, at the price of more tokens per word and token sequences the Phase-1 models never saw (`Court</w>` `,</w>` instead of `Court,</w>`).
  - Choosing the mode (or trying both, as a preprocessing experiment for the report) is your decision; the later notebooks will take it as a setting.
- **Section 9** tells you how many windows each document needs, i.e. the cost of training/inference at each window size.
- **Section 10**: the official evaluator must print `Strict Macro-F1: 1.000000` for gold-vs-gold, and the oracle line must equal our scorer.

Send me the printed output of sections 7–10 (or the `phase2_alignment_report.json`).""")

nb = nbf.v4.new_notebook()
nb["cells"] = cells
nb["metadata"] = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                  "language_info": {"name": "python"}}
nbf.validate(nb)
nbf.write(nb, "/home/claude/p2/phase2_lner_step0_data_alignment_check.ipynb")
print("written", len(cells), "cells")
