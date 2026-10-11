# ===== Phase 2 L-NER: character offsets for the Phase-1 BPE tokenizer, BIO tags, span decoding =====
# Ancillary pipeline code (data handling / alignment / decoding). It does NOT change the tokenizer: the token ids
# produced here are exactly BPETokenizer.encode(text) (checked on every document by check_encode_identical).

LABELS_DEFAULT = ["APP", "RESP", "A.COUNSEL", "R.COUNSEL", "JUDGE", "WIT", "AUTH", "COURT", "STAT", "PREC", "DATE", "CASENO"]
_END_MARK = "</w>"


def iter_words(text):
    """(start_char, end_char) of every whitespace-separated word, using exactly str.split()'s notion of whitespace
    (str.isspace), so the words are the same ones BPETokenizer.encode() sees."""
    n, i = len(text), 0
    while i < n:
        while i < n and text[i].isspace():
            i += 1
        if i >= n:
            break
        j = i
        while j < n and not text[j].isspace():
            j += 1
        yield i, j
        i = j


import re as _re
PIECE_MODES = {
    "none": None,                                                   # exactly the Phase-1 tokenization
    "punct": _re.compile(r"\w+|[^\w\s]"),                          # split every punctuation char off
    "punct_digit": _re.compile(r"[^\W\d_]+|\d+|_+|[^\w\s]"),       # also split letters from digits
}


def iter_pieces(text, mode="none"):
    """Words as the tokenizer sees them. mode 'none' = whitespace words (Phase-1 behaviour); the other modes split
    each whitespace word further, which changes the token ids versus pretraining (a design choice to measure)."""
    rx = PIECE_MODES[mode]
    for ws, we in iter_words(text):
        if rx is None:
            yield ws, we
        else:
            for m in rx.finditer(text, ws, we):
                yield m.start(), m.end()


def encode_with_offsets(tok, text, mode="none"):
    """Return (ids, spans): ids == tok.encode(text); spans[k] = (start_char, end_char) of token k in `text`
    (end exclusive). A token's symbols are UTF-8 bytes held as chr(b); when a token boundary falls inside a
    multi-byte character, that character is given to the token holding its first byte (the next token then starts
    after it; if a token holds only continuation bytes it gets an empty span at that position)."""
    unk = tok.stoi["<unk>"]
    ids, spans = [], []
    for ws, we in iter_pieces(text, mode):
        word = text[ws:we]
        syms = tok._tokenize_word(word)
        cached = tok._id_cache.get(word)
        if cached is None:
            cached = [tok.stoi.get(t, unk) for t in syms]
            tok._id_cache[word] = cached
        # byte offset -> char offset inside the word
        byte_start_of_char = []
        b = 0
        for ch in word:
            byte_start_of_char.append(b)
            b += len(ch.encode("utf-8"))
        total_bytes = b
        byte_to_char = [0] * (total_bytes + 1)
        for ci, bs in enumerate(byte_start_of_char):
            nb = (byte_start_of_char[ci + 1] if ci + 1 < len(byte_start_of_char) else total_bytes) - bs
            for k in range(nb):
                byte_to_char[bs + k] = ci if k == 0 else ci + 1      # a continuation byte "belongs" after its char start
        byte_to_char[total_bytes] = len(word)
        bpos = 0
        for sym in syms:
            core = sym[:-len(_END_MARK)] if sym.endswith(_END_MARK) else sym
            nbytes = len(core)                                       # each symbol char is one byte
            cs = byte_to_char[bpos] if bpos < total_bytes else len(word)
            bpos += nbytes
            ce = byte_to_char[bpos] if bpos < total_bytes else len(word)
            if ce < cs:
                ce = cs
            spans.append((ws + cs, ws + ce))
        ids.extend(cached)
    return ids, spans


def check_encode_identical(tok, docs):
    """Assert our ids equal tok.encode on every doc and that spans are monotone and inside the text."""
    for did, text in docs.items():
        ids, spans = encode_with_offsets(tok, text)
        ref = tok.encode(text)
        assert ids == ref, f"doc {did}: ids differ from BPETokenizer.encode"
        prev = 0
        for (s, e) in spans:
            assert prev <= s <= e <= len(text), f"doc {did}: bad span {(s, e)}"
            prev = e
    return True


# ---------------- BIO tags ----------------
IGNORE = -100        # label for tokens that cover no character (loss ignores them; decoding skips them)
def bio_tag_names(labels):
    return ["O"] + [f"{p}-{l}" for l in labels for p in ("B", "I")]


def gold_to_token_tags(spans, ents, labels):
    """Token tag ids (0 = O, 1+2*li = B-label, 2+2*li = I-label). A token is inside an entity if its char span
    overlaps the entity. Also returns how many entities have a start/end that is NOT on a token boundary."""
    tag = [0 if e > s else IGNORE for s, e in spans]                # empty-span tokens (e.g. a lone '</w>') -> ignored
    lab_idx = {l: i for i, l in enumerate(labels)}
    starts = {s for s, e in spans if e > s}
    ends = {e for s, e in spans if e > s}
    misaligned = []
    ents_sorted = sorted(ents, key=lambda x: (x[0], x[1]))
    k = 0
    for ent in ents_sorted:
        s, e, lab = ent[0], ent[1], ent[2]
        li = lab_idx[lab]
        if s not in starts or e not in ends:
            misaligned.append(ent)
        while k < len(spans) and spans[k][1] <= s:
            k += 1
        j, first = k, True
        while j < len(spans) and spans[j][0] < e:
            if spans[j][1] > spans[j][0]:                            # skip empty (continuation-byte) tokens
                if tag[j] == 0:                                     # never overwrite another entity's token
                    tag[j] = 1 + 2 * li if first else 2 + 2 * li
                first = False
            j += 1
    return tag, misaligned


def token_tags_to_spans(spans, tag, labels):
    """BIO tag ids -> [(start_char, end_char, label)] with lenient repair: an I- tag that does not continue an
    entity of the same label starts a new one."""
    out, cur = [], None
    for (s, e), t in zip(spans, tag):
        if e <= s or t == IGNORE:
            continue
        if t == 0:
            if cur:
                out.append(tuple(cur)); cur = None
            continue
        li, is_b = (t - 1) // 2, (t - 1) % 2 == 0
        lab = labels[li]
        if cur and not is_b and cur[2] == lab:
            cur[1] = e
        else:
            if cur:
                out.append(tuple(cur))
            cur = [s, e, lab]
    if cur:
        out.append(tuple(cur))
    return out


# ---------------- boundary repair (character-level, after token decoding) ----------------
TRAIL_PUNCT = set(",.;:)]}'\"’”`-/")
LEAD_PUNCT = set("([{'\"‘“`-:./")


def learn_trim_table(docs, golds, tok_spans_by_doc, labels):
    """Data-driven boundary repair learned on TRAINING docs only. For each (label, trailing char) seen at the end of
    an oracle token span, count how often the gold entity excludes that char -> trim if excluded > kept.
    Same for leading chars. Returns (trail_rule, lead_rule) dicts {(label, ch): True/False}."""
    from collections import Counter
    t_ex, t_keep, l_ex, l_keep = Counter(), Counter(), Counter(), Counter()
    for did, ents in golds.items():
        text, spans = docs[did], tok_spans_by_doc[did]
        tag, _ = gold_to_token_tags(spans, ents, labels)
        pred = token_tags_to_spans(spans, tag, labels)
        # match each oracle span to the gold entity it covers (same label, contained)
        gl = sorted((e[0], e[1], e[2]) for e in ents)
        for (s, e, lab) in pred:
            cands = [g for g in gl if g[2] == lab and g[0] >= s and g[1] <= e]
            if len(cands) != 1:
                continue
            gs, ge, _ = cands[0]
            # trailing chars of the token span beyond gold end
            for p in range(e - 1, s - 1, -1):
                ch = text[p]
                if ch not in TRAIL_PUNCT:
                    break
                if p >= ge:
                    t_ex[(lab, ch)] += 1
                else:
                    t_keep[(lab, ch)] += 1
                    break
            for p in range(s, e):
                ch = text[p]
                if ch not in LEAD_PUNCT:
                    break
                if p < gs:
                    l_ex[(lab, ch)] += 1
                else:
                    l_keep[(lab, ch)] += 1
                    break
    trail = {k: t_ex[k] > t_keep[k] for k in set(t_ex) | set(t_keep)}
    lead = {k: l_ex[k] > l_keep[k] for k in set(l_ex) | set(l_keep)}
    return trail, lead


def repair_span(text, s, e, lab, trail_rule, lead_rule):
    while e > s + 1 and text[e - 1] in TRAIL_PUNCT and trail_rule.get((lab, text[e - 1]), False):
        e -= 1
    while s < e - 1 and text[s] in LEAD_PUNCT and lead_rule.get((lab, text[s]), False):
        s += 1
    return s, e


def finalize_doc_entities(text, raw_spans, trail_rule=None, lead_rule=None):
    """Repair, drop empty/invalid, de-duplicate (the evaluator FAILS on duplicate (start,end,label)), sort."""
    seen, out = set(), []
    for s, e, lab in raw_spans:
        if trail_rule is not None:
            s, e = repair_span(text, s, e, lab, trail_rule, lead_rule or {})
        s, e = int(s), int(e)
        if not (0 <= s < e <= len(text)):
            continue
        key = (s, e, lab)
        if key in seen:
            continue
        seen.add(key)
        out.append([s, e, lab])
    out.sort()
    return out


# ---------------- strict scorer (same arithmetic as evaluate_phase2.py) + error breakdown ----------------
def strict_scores(golds, preds, labels):
    stats = {l: [0, 0, 0] for l in labels}
    for did, g in golds.items():
        gs = {(x[0], x[1], x[2]) for x in g}
        ps = {(x[0], x[1], x[2]) for x in preds.get(did, [])}
        for l in labels:
            a = {x for x in gs if x[2] == l}; b = {x for x in ps if x[2] == l}
            stats[l][0] += len(a & b); stats[l][1] += len(b - a); stats[l][2] += len(a - b)
    per = {}
    for l, (tp, fp, fn) in stats.items():
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        f = 2 * p * r / (p + r) if p + r else 0.0
        per[l] = dict(P=p, R=r, F1=f, tp=tp, fp=fp, fn=fn)
    n = len(labels)
    return dict(macro_F1=sum(v["F1"] for v in per.values()) / n,
                macro_P=sum(v["P"] for v in per.values()) / n,
                macro_R=sum(v["R"] for v in per.values()) / n, per_label=per)


def error_breakdown(golds, preds):
    """Classify each wrong prediction: boundary error (overlaps a gold entity of the same label), label error (same
    exact span, other label), boundary+label, or spurious (overlaps no gold entity). Also gold entities missed."""
    from collections import Counter
    c = Counter()
    for did, g in golds.items():
        gs = {(x[0], x[1], x[2]) for x in g}
        for (s, e, l) in {(x[0], x[1], x[2]) for x in preds.get(did, [])}:
            if (s, e, l) in gs:
                c["correct"] += 1; continue
            same_span = [x for x in gs if x[0] == s and x[1] == e]
            over = [x for x in gs if x[0] < e and s < x[1]]
            if same_span:
                c["label_error"] += 1
            elif any(x[2] == l for x in over):
                c["boundary_error"] += 1
            elif over:
                c["boundary_and_label_error"] += 1
            else:
                c["spurious"] += 1
        ps = {(x[0], x[1]) for x in preds.get(did, [])}
        for x in gs:
            if not any(p[0] < x[1] and x[0] < p[1] for p in ps):
                c["missed_entirely"] += 1
    return dict(c)


# ---------------- sliding windows ----------------
def make_windows(n_tokens, window, stride):
    """Start indices of windows of length `window` every `stride` tokens, the last one flush with the end."""
    if n_tokens <= window:
        return [0]
    starts = list(range(0, n_tokens - window + 1, stride))
    if starts[-1] + window < n_tokens:
        starts.append(n_tokens - window)
    return starts
