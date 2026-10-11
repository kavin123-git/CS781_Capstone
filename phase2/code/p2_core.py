# ===================== Phase 2 L-NER engine: backbone wrapper, NER head, adaptation methods, training, inference =====================
# Works with any Phase-1 model wrapped as a "backbone" exposing  features(ids, mask, soft_prompt) -> (B, T, d_model).
import math, time, random, json, os, copy
import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------------------------------------------------------------------------------------------------------------
# 1. Documents -> token ids, character spans, BIO tags; windows
# ----------------------------------------------------------------------------------------------------------------------
def prepare_docs(tok, texts, golds, labels, mode):
    docs = {}
    for did, text in texts.items():
        ids, spans = encode_with_offsets(tok, text, mode)
        tags = gold_to_token_tags(spans, golds[did], labels)[0] if golds is not None else None
        docs[did] = dict(text=text, ids=ids, spans=spans, tags=tags)
    return docs


def window_items(docs, window, stride):
    items = []
    for did, d in docs.items():
        n = len(d["ids"])
        if n == 0:
            continue
        for s in make_windows(n, window, d.get("train_stride", stride)):          # a document may carry its own stride
            items.append((did, s))
    return items


def header_len(header, s):
    """Number of document-start tokens prepended to a window that starts at token s (none for the first window)."""
    return min(header, s) if header else 0


def collate(docs, items, window, pad_id=0, with_labels=True, header=0):
    """header = H > 0: every window that does not start the document is preceded by the document's first min(H, s)
    tokens (court, parties, case number), so each window knows which case it belongs to. Header positions carry no label."""
    hs = [header_len(header, s) for _, s in items]
    L = max(h + min(window, len(docs[did]["ids"]) - s) for (did, s), h in zip(items, hs))
    B = len(items)
    ids = torch.full((B, L), pad_id, dtype=torch.long)
    mask = torch.zeros((B, L), dtype=torch.bool)
    labels = torch.full((B, L), IGNORE, dtype=torch.long) if with_labels else None
    for b, ((did, s), h) in enumerate(zip(items, hs)):
        seg = docs[did]["ids"][:h] + docs[did]["ids"][s:s + window]
        n = len(seg)
        ids[b, :n] = torch.tensor(seg, dtype=torch.long)
        mask[b, :n] = True
        if with_labels:
            labels[b, h:n] = torch.tensor(docs[did]["tags"][s:s + window], dtype=torch.long)
    return ids, mask, labels


# ----------------------------------------------------------------------------------------------------------------------
# 2. NER model = backbone + (optional soft prompt) + token-classification head
# ----------------------------------------------------------------------------------------------------------------------
class BiLSTMLayer(nn.Module):
    """A bidirectional LSTM over the backbone's hidden states (a prediction-head option). It gives every token the
    context to its RIGHT inside the window without changing the pretrained (causal) layers. Padding is packed away
    so the backward direction never reads pad positions."""

    def __init__(self, d_in, hidden, layers=1, dropout=0.1):
        super().__init__()
        self.lstm = nn.LSTM(d_in, hidden, num_layers=layers, batch_first=True, bidirectional=True,
                            dropout=dropout if layers > 1 else 0.0)
        self.out_dim = 2 * hidden

    def forward(self, h, mask):
        lengths = mask.sum(1).clamp(min=1).cpu()
        packed = nn.utils.rnn.pack_padded_sequence(h.float(), lengths, batch_first=True, enforce_sorted=False)
        with torch.autocast(device_type=h.device.type, enabled=False):
            out, _ = self.lstm(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True, total_length=h.shape[1])
        return out


class SpanScorer(nn.Module):
    """Span-classification head. Scores every candidate span (token i .. token i+k, k < max_len) as 'not an entity' or one of the
    labels, from  start-projection(i) + end-projection(i+k) + length-embedding(k)  through a small MLP."""

    def __init__(self, d_in, n_labels, dim=128, max_len=32, dropout=0.1):
        super().__init__()
        self.max_len = max_len
        self.start, self.end = nn.Linear(d_in, dim), nn.Linear(d_in, dim)
        self.length = nn.Embedding(max_len, dim)
        self.out = nn.Sequential(nn.GELU(), nn.Dropout(dropout), nn.Linear(dim, n_labels + 1))      # class 0 = no entity

    def forward(self, h):                                       # h: (B, T, d)  ->  (B, T, K, n_labels + 1)
        K = self.max_len
        s, e = self.start(h), self.end(h)
        e_pad = F.pad(e, (0, 0, 0, K - 1))                      # (B, T+K-1, D)
        ee = e_pad.unfold(1, K, 1).permute(0, 1, 3, 2)          # (B, T, K, D): end projection of token i+k
        return self.out(s.unsqueeze(2) + ee + self.length.weight)


def span_valid(ok, K):
    """ok: (B, T) bool (token usable). Candidate (i, k) is valid when token i and token i+k are both usable."""
    ok_end = F.pad(ok, (0, K - 1), value=False).unfold(1, K, 1)        # (B, T, K)
    return ok.unsqueeze(2) & ok_end


def span_targets(labels, K):
    """Gold spans from BIO tag ids -> (B, T, K) class ids (0 = none, 1+label_index); spans longer than K tokens are dropped."""
    B, T = labels.shape
    tgt = torch.zeros(B, T, K, dtype=torch.long)
    lab = labels.cpu().tolist()
    for b in range(B):
        row, i = lab[b], 0
        while i < T:
            t = row[i]
            if t > 0 and (t - 1) % 2 == 0:
                c = (t - 1) // 2
                j = i
                while j + 1 < T and row[j + 1] == 2 + 2 * c:
                    j += 1
                if j - i < K:
                    tgt[b, i, j - i] = c + 1
                i = j + 1
            else:
                i += 1
    return tgt


def span_loss(span_logits, labels, K, C):
    """CE over valid candidates: every gold span plus the hardest negatives (20x as many, at least 256)."""
    ok = labels != IGNORE
    valid = span_valid(ok, K)
    tgt = span_targets(labels, K).to(span_logits.device)
    lg, tg = span_logits[valid].float(), tgt[valid]
    if lg.shape[0] == 0:
        return span_logits.sum() * 0.0
    ce = F.cross_entropy(lg, tg, reduction="none")
    pos = tg > 0
    n_neg = max(256, 20 * int(pos.sum()))
    neg_ce = ce[~pos]
    if neg_ce.numel() > n_neg:
        neg_ce = neg_ce.topk(n_neg).values
    n = int(pos.sum()) + neg_ce.numel()
    return (ce[pos].sum() + neg_ce.sum()) / max(1, n)


class NERModel(nn.Module):
    def __init__(self, backbone, d_model, n_tags, head_type="linear", head_dropout=0.1, head_norm=False, lstm_hidden=256,
                 lstm_layers=1, span_dim=0, span_max_len=32):
        super().__init__()
        self.backbone = backbone
        self.soft_prompt = None                          # set by the prompt-tuning technique
        self.head_type = head_type
        self.bilstm = None
        layers = [nn.Dropout(head_dropout)]
        if head_norm:
            layers.append(nn.LayerNorm(d_model))
        if head_type == "linear":
            layers.append(nn.Linear(d_model, n_tags))
        elif head_type == "mlp":
            layers += [nn.Linear(d_model, d_model), nn.GELU(), nn.Dropout(head_dropout), nn.Linear(d_model, n_tags)]
        elif head_type == "bilstm":
            self.bilstm = BiLSTMLayer(d_model, lstm_hidden, lstm_layers, head_dropout)
            layers += [nn.Linear(self.bilstm.out_dim, n_tags)]
        else:
            raise ValueError(head_type)
        self.head = nn.Sequential(*layers)
        self.span = None
        if span_dim:
            d_feat = self.bilstm.out_dim if self.bilstm is not None else d_model
            self.span = SpanScorer(d_feat, (n_tags - 1) // 2, span_dim, span_max_len, head_dropout)

    def head_parameters(self):
        ps = list(self.head.parameters())
        if self.bilstm is not None:
            ps += list(self.bilstm.parameters())
        if self.span is not None:
            ps += list(self.span.parameters())
        return ps

    def forward(self, ids, mask, want_span=False):
        h = self.backbone.features(ids, mask, self.soft_prompt)
        if self.bilstm is not None:
            pre, body = self.bilstm(self.head[0](h), mask), list(self.head[1:-1])
        else:
            pre, body = h, list(self.head[:-1])
        for layer in body:
            pre = layer(pre)
        logits = self.head[-1](pre)
        if want_span:
            return logits, self.span(pre.float())
        return logits


# ----------------------------------------------------------------------------------------------------------------------
# 3. Adaptation methods
# ----------------------------------------------------------------------------------------------------------------------
class LoRALinear(nn.Module):
    """LoRA (Hu et al. 2022): y = W0 x + b + (alpha/r) * B A x.  A: (r, in) random, B: (out, r) zeros -> starts == W0."""

    def __init__(self, base, r, alpha, dropout=0.0):
        super().__init__()
        self.base = base
        self.r, self.scale = r, alpha / r
        self.A = nn.Parameter(torch.empty(r, base.in_features))
        self.B = nn.Parameter(torch.zeros(base.out_features, r))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        return self.base(x) + self.scale * F.linear(F.linear(self.drop(x), self.A), self.B)


_ROT_IDX = {}


def _row_rotation_index(d, device):
    key = (d, str(device))
    if key not in _ROT_IDX:
        i = torch.arange(d, device=device).unsqueeze(1)
        j = torch.arange(d, device=device).unsqueeze(0)
        _ROT_IDX[key] = (j - i) % d                      # Mpi[i, j] = M[i, (j - i) mod d]  (row i rotated right by i)
    return _ROT_IDX[key]


class LoRMALinear(nn.Module):
    """LoRMA (Bihany, Patel, Modi, ACL Findings 2025): a MULTIPLICATIVE low-rank update of a frozen W0 (out x in).
      variant "plus" (LoRMA+, additive rank inflation):  h = ((alpha/r) B A + I) W0 x + b
            A: (r, out) random, B: (out, r) zeros  -> starts exactly at W0.  Reordered as  y0 = W0 x;  y0 + (alpha/r) B (A y0)
            so the out x out matrix is never formed (same cost order as LoRA).
      variant "pi" (LoRMA_pi, permutation-based inflation):  h = I_pi(B A) W0 x + b, where I_pi rotates row i of BA
            cyclically by i. Init: B[:, 0] = 1, other B entries random; A[0, 0] = 1, other A entries 0 -> I_pi(BA) = I.
            Needs the out x out matrix every step (O(d^2 (r + batch)))."""

    def __init__(self, base, r, alpha, variant="plus"):
        super().__init__()
        self.base = base
        self.variant = variant
        self.r, self.scale = r, alpha / r
        d = base.out_features
        if variant == "plus":
            self.A = nn.Parameter(torch.empty(r, d))
            self.B = nn.Parameter(torch.zeros(d, r))
            nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        elif variant == "pi":
            A = torch.zeros(r, d); A[0, 0] = 1.0
            B = torch.randn(d, r) * (1.0 / math.sqrt(d)); B[:, 0] = 1.0
            self.A, self.B = nn.Parameter(A), nn.Parameter(B)
        else:
            raise ValueError(variant)

    def forward(self, x):
        y0 = F.linear(x, self.base.weight)
        if self.variant == "plus":
            y = y0 + self.scale * F.linear(F.linear(y0, self.A), self.B)
        else:
            M = self.B @ self.A                                                        # (d, d)
            Mpi = torch.gather(M, 1, _row_rotation_index(M.shape[0], M.device))
            y = F.linear(y0, Mpi.to(y0.dtype))
        if self.base.bias is not None:
            y = y + self.base.bias
        return y


class BottleneckAdapter(nn.Module):
    """Houlsby/Pfeiffer bottleneck: out + up(act(down(out))); up is zero-initialised -> starts as identity."""

    def __init__(self, d, m):
        super().__init__()
        self.down = nn.Linear(d, m)
        self.up = nn.Linear(m, d)
        nn.init.zeros_(self.up.weight); nn.init.zeros_(self.up.bias)

    def forward(self, h):
        return h + self.up(F.gelu(self.down(h)))


def _replace_linears(root, name_suffixes, make):
    """Replace every nn.Linear whose qualified name ends with one of name_suffixes by make(linear). Returns names."""
    done = []
    for qual, mod in list(root.named_modules()):
        for child_name, child in list(mod.named_children()):
            full = f"{qual}.{child_name}" if qual else child_name
            if isinstance(child, nn.Linear) and any(full.endswith(s) for s in name_suffixes):
                setattr(mod, child_name, make(child))
                done.append(full)
    return done


def apply_technique(ner, C, log=print):
    """Freezes/unfreezes and inserts modules according to C["TECHNIQUE"]. The head is always trainable.
    Returns a short description dict (for the results file)."""
    tech = C["TECHNIQUE"]
    bb = ner.backbone
    info = dict(technique=tech)
    for p in bb.parameters():
        p.requires_grad = False
    if tech == "full_ft":
        for p in bb.parameters():
            p.requires_grad = True
    elif tech == "partial_ft":
        k = C["PARTIAL_TOP_BLOCKS"]
        blocks = bb.layer_list()
        for blk in blocks[len(blocks) - k:]:
            for p in blk.parameters():
                p.requires_grad = True
        for p in bb.final_modules_params():
            p.requires_grad = True
        info.update(top_blocks=k, of_blocks=len(blocks))
    elif tech == "bitfit":
        n = 0
        for name, p in bb.named_parameters():
            if name.split(".")[-1].startswith("bias"):            # Linear/LayerNorm ".bias" and LSTM "bias_ih_l0"/"bias_hh_l0"
                p.requires_grad = True; n += 1
        info.update(bias_tensors=n)
    elif tech == "lora":
        done = _replace_linears(bb, C["PEFT_TARGETS"], lambda lin: LoRALinear(lin, C["RANK"], C["ALPHA"], C.get("LORA_DROPOUT", 0.0)))
        assert done, f"no linear layer matches PEFT_TARGETS={C['PEFT_TARGETS']} in this model"
        info.update(targets=len(done), rank=C["RANK"], alpha=C["ALPHA"])
    elif tech == "lorma":
        done = _replace_linears(bb, C["PEFT_TARGETS"], lambda lin: LoRMALinear(lin, C["RANK"], C["ALPHA"], C["LORMA_VARIANT"]))
        assert done, f"no linear layer matches PEFT_TARGETS={C['PEFT_TARGETS']} in this model"
        info.update(targets=len(done), rank=C["RANK"], alpha=C["ALPHA"], variant=C["LORMA_VARIANT"])
    elif tech == "adapter":
        sites = bb.adapter_sites(C["ADAPTER_TYPE"])            # list of (module whose output gets an adapter, d)
        ner.adapters = nn.ModuleList()
        for mod, d in sites:
            ad = BottleneckAdapter(d, C["ADAPTER_DIM"])
            ner.adapters.append(ad)
            mod.register_forward_hook(lambda m, inp, out, ad=ad: ad(out))
        info.update(adapter_type=C["ADAPTER_TYPE"], sites=len(sites), bottleneck=C["ADAPTER_DIM"])
    elif tech == "prompt":
        P = C["PROMPT_LEN"]
        emb = bb.embedding_weight()
        idx = torch.randint(100, emb.shape[0], (P,), generator=torch.Generator().manual_seed(0))
        ner.soft_prompt = nn.Parameter(emb[idx].detach().clone().float())
        info.update(prompt_len=P)
    elif tech == "prefix":
        P = C["PREFIX_LEN"]
        bb.enable_prefix(P, C.get("PREFIX_HIDDEN", 0))
        info.update(prefix_len=P)
    else:
        raise ValueError(f"unknown TECHNIQUE {tech}")
    for p in ner.head_parameters():
        p.requires_grad = True
    total = sum(p.numel() for p in ner.parameters())
    trainable = sum(p.numel() for p in ner.parameters() if p.requires_grad)
    backbone_orig = bb.original_param_count
    info.update(params_total=total, params_trainable=trainable, params_backbone_original=backbone_orig,
                trainable_pct_of_backbone=round(100.0 * trainable / backbone_orig, 4))
    log(f"Technique {tech}: trainable {trainable:,} of {total:,} parameters ({info['trainable_pct_of_backbone']}% of the Phase-1 model's {backbone_orig:,})")
    return info


def trainable_state(ner):
    """Everything that can differ from the Phase-1 weights: trainable params + routing-bias buffers (if updated)."""
    st = {n: p.detach().to("cpu", copy=True) for n, p in ner.named_parameters() if p.requires_grad}
    for n, b in ner.named_buffers():
        if n.endswith(("select_bias", ".tau", ".tau_ready")):         # MoE routing bias, MoR routing thresholds (updated in training)
            st[n] = b.detach().to("cpu", copy=True)
    return st


def load_trainable_state(ner, st):
    missing = [k for k in st if k not in dict(ner.named_parameters()) and k not in dict(ner.named_buffers())]
    assert not missing, f"state has unknown keys: {missing[:5]}"
    ner.load_state_dict(st, strict=False)


# ----------------------------------------------------------------------------------------------------------------------
# 4. Training (optionally with a knowledge-distillation teacher)
# ----------------------------------------------------------------------------------------------------------------------
def tag_class_weights(docs, n_tags, scheme):
    if scheme == "none":
        return None
    cnt = torch.zeros(n_tags)
    for d in docs.values():
        for t in d["tags"]:
            if t >= 0:
                cnt[t] += 1
    cnt = cnt.clamp(min=1)
    if scheme == "inv_sqrt":
        w = (cnt.sum() / cnt).sqrt()
    elif scheme == "inv":
        w = cnt.sum() / cnt
    else:
        raise ValueError(scheme)
    return w / w[0]                                         # O keeps weight 1


def token_loss(logits, labels, cw, C):
    """Token cross-entropy with optional class weights, label smoothing and focal modulation ((1-p_t)^gamma)."""
    n_tags = logits.shape[-1]
    lg, lb = logits.reshape(-1, n_tags), labels.reshape(-1)
    gamma = C.get("FOCAL_GAMMA", 0.0)
    if gamma > 0:
        keep = lb != IGNORE
        lg, lb = lg[keep], lb[keep]
        ce = F.cross_entropy(lg, lb, weight=cw, reduction="none")
        pt = torch.exp(-F.cross_entropy(lg, lb, reduction="none"))
        return (ce * (1 - pt) ** gamma).sum() / (cw[lb].sum() if cw is not None else max(1, lb.numel()))
    return F.cross_entropy(lg, lb, weight=cw, ignore_index=IGNORE, label_smoothing=C.get("LABEL_SMOOTHING", 0.0))


def oversample_items(items, docs, window, tag_ids, factor):
    """Repeat (factor-1 extra times) every training window that contains a token of the given (rare) tag ids."""
    if factor <= 1 or not tag_ids:
        return items
    tag_ids = set(tag_ids)
    extra = []
    for did, s in items:
        if any(t in tag_ids for t in docs[did]["tags"][s:s + window]):
            extra += [(did, s)] * (int(factor) - 1)
    return items + extra


def _lr_lambda(total, warmup):
    def f(step):
        if step < warmup:
            return (step + 1) / max(1, warmup)
        return max(0.0, (total - step) / max(1, total - warmup))
    return f


def train_ner(ner, train_docs, C, device, eval_fn=None, teacher=None, log=print):
    """Trains the trainable parameters. eval_fn(ner) -> (score, details) is called after each epoch (if given); the
    best epoch's trainable state is kept. teacher(ids, mask) -> logits enables KD.
    Returns dict(history, best_epoch, best_score, best_state, train_seconds, peak_mem_gb, tokens_per_sec)."""
    ner.to(device)
    torch.manual_seed(C["SEED"]); random.seed(C["SEED"])
    items = window_items(train_docs, C["WINDOW"], C["TRAIN_STRIDE"])
    n_windows_raw = len(items)
    items = oversample_items(items, train_docs, C["WINDOW"], C.get("OVERSAMPLE_TAGS", []), C.get("OVERSAMPLE", 1))
    HDR = C.get("HEADER", 0)
    n_tags = ner.head[-1].out_features
    cw = tag_class_weights(train_docs, n_tags, C["CLASS_WEIGHTING"])
    cw = cw.to(device) if cw is not None else None
    head_params = [p for p in ner.head_parameters() if p.requires_grad]
    hp = set(id(p) for p in head_params)
    other = [p for p in ner.parameters() if p.requires_grad and id(p) not in hp]
    groups = [dict(params=head_params, lr=C["HEAD_LR"])]
    if other:
        groups.append(dict(params=other, lr=C["LR"]))
    opt = torch.optim.AdamW(groups, weight_decay=C["WEIGHT_DECAY"], betas=(0.9, 0.98))
    steps_per_epoch = math.ceil(len(items) / (C["BATCH"] * C["ACCUM"]))
    total = steps_per_epoch * C["EPOCHS"]
    sched = torch.optim.lr_scheduler.LambdaLR(opt, _lr_lambda(total, int(C["WARMUP_FRAC"] * total)))
    use_amp = device.type == "cuda" and C["AMP"] == "fp16"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    hist, best, best_state, best_ep, bad_epochs = [], -1.0, None, 0, 0
    t0, tokens = time.time(), 0
    log(f"training: {len(items)} windows/epoch" + (f" ({n_windows_raw} distinct)" if len(items) != n_windows_raw else "") + f", {steps_per_epoch} optimizer steps/epoch, {C['EPOCHS']} epochs max")
    for ep in range(1, C["EPOCHS"] + 1):
        ner.train()
        rng = random.Random(C["SEED"] * 1000 + ep)
        order = items[:]; rng.shuffle(order)
        run_loss, nb, skipped = 0.0, 0, 0
        opt.zero_grad(set_to_none=True)
        for bi in range(0, len(order), C["BATCH"]):
            batch_items = order[bi:bi + C["BATCH"]]
            ids, mask, labels = collate(train_docs, batch_items, C["WINDOW"], header=HDR)
            ids, mask, labels = ids.to(device), mask.to(device), labels.to(device)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                if ner.span is not None:
                    logits, span_lg = ner(ids, mask, want_span=True)
                else:
                    logits = ner(ids, mask)
            logits = logits.float()
            if bool((labels != IGNORE).any()):
                loss = token_loss(logits, labels, cw, C)
                if ner.span is not None:
                    loss = loss + C.get("SPAN_WEIGHT", 1.0) * span_loss(span_lg, labels, ner.span.max_len, C)
            else:                                              # a batch of only unlabelled (teacher-only) windows: no hard-label term
                loss = logits.sum() * 0.0
            if teacher is not None and C.get("KD_ALPHA", 0) > 0:
                with torch.no_grad():
                    t_logits = teacher(ids, mask).float()
                unl = torch.tensor([bool(train_docs[did].get("unlabeled", False)) for did, _ in batch_items], device=device)
                keep = ((labels != IGNORE) | (unl.unsqueeze(1) & mask)).view(-1)     # unlabelled windows: every real token
                T = C["KD_TEMPERATURE"]
                kd = F.kl_div(F.log_softmax(logits.view(-1, n_tags)[keep] / T, -1),
                              F.softmax(t_logits.view(-1, n_tags)[keep] / T, -1), reduction="batchmean") * T * T
                loss = (1 - C["KD_ALPHA"]) * loss + C["KD_ALPHA"] * kd
            if not torch.isfinite(loss):                      # e.g. an fp16 overflow: skip this batch, never back-propagate NaN/inf
                skipped += 1
                continue
            scaler.scale(loss / C["ACCUM"]).backward()
            nb += 1
            tokens += int(mask.sum())
            run_loss += loss.item()
            if nb % C["ACCUM"] == 0 or bi + C["BATCH"] >= len(order):
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_([p for g in groups for p in g["params"]], C["GRAD_CLIP"])
                scaler.step(opt); scaler.update(); sched.step()
                opt.zero_grad(set_to_none=True)
        row = dict(epoch=ep, train_loss=round(run_loss / max(1, nb), 5), seconds=round(time.time() - t0, 1))
        if skipped:
            row["skipped_nonfinite_batches"] = skipped
        if eval_fn is not None:
            score, det = eval_fn(ner)
            row.update(heldout_macroF1=round(score, 5), **det)
            if score > best:
                best, best_ep, best_state, bad_epochs = score, ep, trainable_state(ner), 0
            else:
                bad_epochs += 1
        hist.append(row)
        log(f"  epoch {ep}: " + ", ".join(f"{k}={v}" for k, v in row.items() if k != "epoch"))
        if nb == 0:
            log("  every batch of this epoch was non-finite: stopping (the best earlier epoch is kept)")
            break
        if eval_fn is not None and C["PATIENCE"] and bad_epochs >= C["PATIENCE"]:
            log(f"  early stop: no improvement for {bad_epochs} epochs")
            break
    if eval_fn is None:
        best_state, best_ep = trainable_state(ner), len(hist)
    secs = time.time() - t0
    peak = torch.cuda.max_memory_allocated(device) / 1e9 if device.type == "cuda" else None
    return dict(history=hist, best_epoch=best_ep, best_score=best, best_state=best_state, train_seconds=round(secs, 1),
                peak_mem_gb=round(peak, 3) if peak else None, tokens_per_sec=round(tokens / max(secs, 1e-9), 1))


# ----------------------------------------------------------------------------------------------------------------------
# 5. Inference: windows -> per-token logits (merged) -> tags -> character spans
# ----------------------------------------------------------------------------------------------------------------------
@torch.no_grad()
def doc_logits(ner, doc, C, device, causal):
    n = len(doc["ids"])
    n_tags = ner.head[-1].out_features
    if n == 0:
        return torch.zeros(0, n_tags)
    W, S, H = C["WINDOW"], C["INFER_STRIDE"], C.get("HEADER", 0)
    starts = make_windows(n, W, S)
    out = torch.zeros(n, n_tags)
    best = torch.full((n,), -1.0)
    cnt = torch.zeros(n)
    use_amp = device.type == "cuda" and C["AMP"] == "fp16"
    ner.eval()
    for bi in range(0, len(starts), C["INFER_BATCH"]):
        chunk = [("_", s) for s in starts[bi:bi + C["INFER_BATCH"]]]
        ids, mask, _ = collate({"_": doc}, chunk, W, with_labels=False, header=H)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            lg = ner(ids.to(device), mask.to(device)).float().cpu()
        for b, (_, s) in enumerate(chunk):
            h = header_len(H, s)
            L = int(mask[b].sum()) - h
            lg_b = lg[b, h:h + L]
            pos = torch.arange(L)
            if C["MERGE"] == "average":
                out[s:s + L] += lg_b; cnt[s:s + L] += 1
            else:                                                # "context": keep the window giving each token most context
                score = pos.float() if causal else torch.minimum(pos, L - 1 - pos).float()
                upd = score > best[s:s + L]
                out[s:s + L][upd] = lg_b[upd]
                best[s:s + L] = torch.where(upd, score, best[s:s + L])
    if C["MERGE"] == "average":
        out = out / cnt.clamp(min=1).unsqueeze(1)
    return out


def bio_allowed(n_tags):
    """allowed[prev, cur]: I-X may only follow B-X or I-X."""
    a = torch.ones(n_tags, n_tags, dtype=torch.bool)
    for cur in range(1, n_tags):
        if (cur - 1) % 2 == 1:                                    # I- tag
            b_same, i_same = cur - 1, cur
            for prev in range(n_tags):
                if prev not in (b_same, i_same):
                    a[prev, cur] = False
    return a


def viterbi_bio(logits):
    """Best tag sequence under BIO constraints (no transition scores, only the hard constraint)."""
    n, K = logits.shape
    if n == 0:
        return []
    lp = F.log_softmax(logits.float(), -1)
    allowed = bio_allowed(K)
    trans = torch.where(allowed, torch.tensor(0.0), torch.tensor(-1e9))
    start_ok = torch.tensor([0.0] + [0.0 if (k - 1) % 2 == 0 else -1e9 for k in range(1, K)])
    score = lp[0] + start_ok
    back = torch.zeros(n, K, dtype=torch.long)
    for t in range(1, n):
        cand = score.unsqueeze(1) + trans                         # (prev, cur)
        best, arg = cand.max(0)
        back[t] = arg
        score = best + lp[t]
    path = [int(score.argmax())]
    for t in range(n - 1, 0, -1):
        path.append(int(back[t, path[-1]]))
    return path[::-1]


@torch.no_grad()
def doc_span_candidates(ner, doc, C, device, min_prob):
    """Span-head decoding: candidate (token_i, token_j, label_index, prob) with prob >= min_prob from every window."""
    n = len(doc["ids"])
    W, H, K = C["WINDOW"], C.get("HEADER", 0), ner.span.max_len
    starts = make_windows(n, W, C["INFER_STRIDE"])
    use_amp = device.type == "cuda" and C["AMP"] == "fp16"
    cands = {}
    ner.eval()
    for bi in range(0, len(starts), C["INFER_BATCH"]):
        chunk = [("_", s) for s in starts[bi:bi + C["INFER_BATCH"]]]
        ids, mask, _ = collate({"_": doc}, chunk, W, with_labels=False, header=H)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            _, sp = ner(ids.to(device), mask.to(device), want_span=True)
        prob = sp.float().softmax(-1).cpu()                               # (B, T, K, n_labels+1)
        valid = span_valid(mask, K)
        prob[..., 0] = 0.0
        prob[~valid] = 0.0
        best_p, best_c = prob.max(-1)                                     # (B, T, K)
        for b, (_, s) in enumerate(chunk):
            h = header_len(H, s)
            idx = (best_p[b] >= min_prob).nonzero().tolist()
            for i, k in idx:
                if i < h:
                    continue
                gi, gj = s + i - h, s + i - h + k
                if gj >= n:
                    continue
                p = float(best_p[b, i, k])
                key = (gi, gj)
                if key not in cands or p > cands[key][1]:
                    cands[key] = (int(best_c[b, i, k]) - 1, p)
    return [(i, j, c, p) for (i, j), (c, p) in cands.items()]


def decode_spans(doc, cands, labels, C, trail=None, lead=None):
    """Greedy non-overlapping selection by probability, then the same character-level finalisation as the token decoder."""
    chosen, taken = [], set()
    for i, j, c, p in sorted(cands, key=lambda x: -x[3]):
        if p < C.get("SPAN_MIN_PROB", 0.5) or any(t in taken for t in range(i, j + 1)):
            continue
        taken.update(range(i, j + 1))
        chosen.append((i, j, c))
    spans = doc["spans"]
    raw = sorted((spans[i][0], spans[j][1], labels[c]) for i, j, c in chosen)
    return finalize_doc_entities(doc["text"], raw, trail, lead) if C["BOUNDARY_REPAIR"] else finalize_doc_entities(doc["text"], raw)


def decode_doc(doc, logits, labels, C, trail=None, lead=None):
    if len(doc["ids"]) == 0:
        return []
    if C["DECODE"] == "viterbi":
        tags = viterbi_bio(logits)
    else:
        tags = logits.argmax(-1).tolist()
    spans = doc["spans"]
    tags = [IGNORE if e <= s else t for (s, e), t in zip(spans, tags)]
    raw = token_tags_to_spans(spans, tags, labels)
    if C["BOUNDARY_REPAIR"]:
        return finalize_doc_entities(doc["text"], raw, trail, lead)
    return finalize_doc_entities(doc["text"], raw)


def predict_docs(ner, docs, labels, C, device, causal, trail=None, lead=None):
    t0 = time.time()
    if C["DECODE"] == "span":
        assert ner.span is not None, "DECODE='span' needs SPAN_DIM > 0"
        preds = {did: (decode_spans(d, doc_span_candidates(ner, d, C, device, C.get("SPAN_MIN_PROB", 0.5)), labels, C, trail, lead)
                       if len(d["ids"]) else []) for did, d in docs.items()}
        return preds, time.time() - t0
    preds = {did: decode_doc(d, doc_logits(ner, d, C, device, causal), labels, C, trail, lead) for did, d in docs.items()}
    return preds, time.time() - t0


def confusion_pairs(golds, preds):
    """(gold label -> predicted label) counts for predictions that overlap a gold entity of a different label."""
    from collections import Counter
    c = Counter()
    for did, g in golds.items():
        for p in preds.get(did, []):
            for x in g:
                if x[0] < p[1] and p[0] < x[1] and x[2] != p[2]:
                    c[f"{x[2]}->{p[2]}"] += 1
    return dict(c.most_common(25))


def detailed_analysis(golds, preds, texts, n_tokens=None):
    """Error analysis for the report: (1) what kind of boundary errors (too long / too short / shifted; differing only by
    punctuation/space; the most common differing strings), (2) where in the document errors happen (header region vs
    body), (3) document length (short / medium / long thirds by characters)."""
    from collections import Counter
    kinds, diffs = Counter(), Counter()
    for did, g in golds.items():
        gl = [tuple(x[:3]) for x in g]
        gs = set(gl)
        t = texts[did]
        for (s, e, l) in {tuple(x[:3]) for x in preds.get(did, [])}:
            if (s, e, l) in gs:
                continue
            ov = [x for x in gl if x[2] == l and x[0] < e and s < x[1]]
            if not ov:
                continue
            x = min(ov, key=lambda x: abs(x[0] - s) + abs(x[1] - e))
            if s <= x[0] and e >= x[1]:
                kinds["pred_too_long"] += 1
                extra = (t[s:x[0]] + " | " + t[x[1]:e]).strip()
            elif s >= x[0] and e <= x[1]:
                kinds["pred_too_short"] += 1
                extra = (t[x[0]:s] + " | " + t[e:x[1]]).strip()
            else:
                kinds["shifted"] += 1
                extra = None
            if extra is not None:
                if not any(c.isalnum() for c in extra):
                    kinds["differs_only_by_punct_or_space"] += 1
                diffs[extra[:40]] += 1
    out = dict(boundary_error_kinds=dict(kinds), boundary_diff_top=diffs.most_common(15))

    def region_of(s, n):
        r = s / max(n, 1)
        return "first_5pct(header)" if r < 0.05 else ("5-50pct" if r < 0.5 else "50-100pct")
    reg = {k: Counter() for k in ("first_5pct(header)", "5-50pct", "50-100pct")}
    for did, g in golds.items():
        n = len(texts[did])
        gs = {tuple(x[:3]) for x in g}
        ps = {tuple(x[:3]) for x in preds.get(did, [])}
        for x in gs:
            reg[region_of(x[0], n)]["gold"] += 1
            reg[region_of(x[0], n)]["tp"] += x in ps
        for x in ps:
            reg[region_of(x[0], n)]["pred"] += 1
            if not any(y[0] < x[1] and x[0] < y[1] for y in gs):
                reg[region_of(x[0], n)]["spurious"] += 1
    out["by_region"] = {k: dict(gold=v["gold"], pred=v["pred"], tp=v["tp"], spurious=v["spurious"],
                                recall=round(v["tp"] / max(v["gold"], 1), 4), precision=round(v["tp"] / max(v["pred"], 1), 4))
                        for k, v in reg.items()}
    ids = sorted(golds, key=lambda d: len(texts[d]))
    k = max(1, len(ids) // 3)
    groups = {"short_third": ids[:k], "middle_third": ids[k:2 * k], "long_third": ids[2 * k:]}
    out["by_doc_length"] = {}
    for name, ds in groups.items():
        if not ds:
            continue
        tp = sum(len({tuple(x[:3]) for x in golds[d]} & {tuple(x[:3]) for x in preds.get(d, [])}) for d in ds)
        ng = sum(len(golds[d]) for d in ds)
        npd = sum(len(preds.get(d, [])) for d in ds)
        p, r = tp / max(npd, 1), tp / max(ng, 1)
        out["by_doc_length"][name] = dict(docs=len(ds), chars_min=len(texts[ds[0]]), chars_max=len(texts[ds[-1]]),
                                          micro_P=round(p, 4), micro_R=round(r, 4), micro_F1=round(2 * p * r / max(p + r, 1e-9), 4))
    return out
