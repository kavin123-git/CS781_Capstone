USE_CANON = True          # False = the plain GQA + RoPE baseline model (no Canon modules at all)
CANON_BOTTLENECK = 128    # each stream compresses the 768-d hidden state down to this many numbers before mixing
CANON_N_TOK = 4           # token stream:    this position + the 3 before it
CANON_N_WORD = 3          # word stream:     the last 3 FINISHED words (a word ends at a token whose string ends with </w>)
CANON_N_SENT = 2          # sentence stream: the last 2 FINISHED sentences (a sentence ends at '.</w>', '?</w>' or '!</w>')
CANON_USE_TOK = True      # stream switches, for ablations (at least one must be True)
CANON_USE_WORD = True
CANON_USE_SENT = True
CANON_SITE_A = True       # a Canon module after norm1, right before attention
CANON_SITE_C = True       # a Canon module after norm2, right before the feed-forward
CANON_KW = dict(bottleneck=CANON_BOTTLENECK, n_tok=CANON_N_TOK, n_word=CANON_N_WORD, n_sent=CANON_N_SENT,
                use_tok=CANON_USE_TOK, use_word=CANON_USE_WORD, use_sent=CANON_USE_SENT)
if USE_CANON:
    CANON_TAG = (f"bcanon_b{CANON_BOTTLENECK}_t{CANON_N_TOK if CANON_USE_TOK else 0}w{CANON_N_WORD if CANON_USE_WORD else 0}"
                 f"s{CANON_N_SENT if CANON_USE_SENT else 0}_{'A' if CANON_SITE_A else ''}{'C' if CANON_SITE_C else ''}")
else:
    CANON_TAG = "nocanon"


class RMSNorm(nn.Module):
    """Like LayerNorm, but simpler: it only rescales by the root-mean-square
    of the values, it does not re-center the mean. Cheaper, and used in
    models like LLaMA."""

    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        rms = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).sqrt()
        return x / rms * self.weight


def make_norm(dim, norm_type):
    """norm_type is "layernorm", "rmsnorm" or "deepnorm".
    deepnorm reuses a plain LayerNorm -- what makes deepnorm special isn't
    the norm formula, it's the residual scaling (alpha) and weight init
    (beta) we apply around it below."""
    if norm_type == "rmsnorm":
        return RMSNorm(dim)
    else:  # "layernorm" or "deepnorm"
        return nn.LayerNorm(dim)


def deepnorm_alpha(n_layers):
    """Residual scaling factor from the DeepNet paper: alpha = (2N)^0.25."""
    return (2.0 * n_layers) ** 0.25


def deepnorm_init(output_projections, n_layers):
    """Scales the init of a sub-layer's OUTPUT projection(s) (e.g. attention
    out_proj, the feed-forward's second linear) by beta = (8N)^-0.25, so a
    deep post-norm stack stays stable to train. Only used when
    NORM_TYPE == "deepnorm"."""
    beta = (8.0 * n_layers) ** -0.25
    for module in output_projections:
        nn.init.xavier_normal_(module.weight, gain=beta)
        if module.bias is not None:
            nn.init.zeros_(module.bias)

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---- Boundary Canon module (standalone-tested separately; see the markdown above) ----
WORD_END_SUFFIX = "</w>"
SENT_END_SUFFIXES = (".</w>", "?</w>", "!</w>")


def build_boundary_tables(itos):
    """itos: list or dict mapping token id -> token string (A1 BPE style, words end with '</w>').

    Returns (word_end_table, sent_end_table): bool tensors of length vocab. A token ends a word iff its string ends
    with '</w>'; it ends a sentence iff its string ends with '.</w>', '?</w>' or '!</w>'. Special tokens
    ('<pad>', '<unk>', '<s>', '</s>') are neither."""
    if isinstance(itos, dict):
        size = max(itos.keys()) + 1
        items = itos.items()
    else:
        size = len(itos)
        items = enumerate(itos)
    word = torch.zeros(size, dtype=torch.bool)
    sent = torch.zeros(size, dtype=torch.bool)
    for i, s in items:
        if isinstance(s, str):
            if s.endswith(WORD_END_SUFFIX):
                word[i] = True
            if s.endswith(SENT_END_SUFFIXES):
                sent[i] = True
    return word, sent


def _prev_and_upto(flags):
    """flags (B, T) bool.
    upto[b, t] = latest flagged position <= t (or -1).
    prev[b, t] = latest flagged position strictly < t (or -1)."""
    B, T = flags.shape
    pos = torch.arange(T, device=flags.device).unsqueeze(0).expand(B, T)
    cand = torch.where(flags, pos, torch.full_like(pos, -1))
    upto = cand.cummax(dim=1).values
    prev = torch.cat([torch.full((B, 1), -1, dtype=upto.dtype, device=flags.device), upto[:, :-1]], dim=1)
    return prev, upto


def _chain(prev, first, n):
    """Follow 'what was flagged before this?' n-1 more times. first: (B, N) starting indices (-1 = none).
    Returns a list of n index tensors (B, N); entry h is the (h+1)-th most recent flagged position."""
    out = [first]
    cur = first
    for _ in range(n - 1):
        nxt = prev.gather(1, cur.clamp(min=0))
        nxt = torch.where(cur >= 0, nxt, torch.full_like(nxt, -1))
        out.append(nxt)
        cur = nxt
    return out


def _gather(Z, idx):
    """Z (B, T, b), idx (B, N) with -1 meaning 'none' -> (B, N, b), zeros where idx < 0."""
    g = Z.gather(1, idx.clamp(min=0).unsqueeze(-1).expand(-1, -1, Z.size(-1)))
    return g * (idx >= 0).unsqueeze(-1).to(Z.dtype)


class BoundaryCanon(nn.Module):
    def __init__(self, d_model, bottleneck=128, n_tok=4, n_word=3, n_sent=2,
                 use_tok=True, use_word=True, use_sent=True):
        super().__init__()
        assert n_tok >= 2 and n_word >= 1 and n_sent >= 1
        assert use_tok or use_word or use_sent, "need at least one stream"
        self.d, self.b = d_model, bottleneck
        self.n_tok, self.n_word, self.n_sent = n_tok, n_word, n_sent
        self.use_tok, self.use_word, self.use_sent = use_tok, use_word, use_sent
        if use_tok:
            self.down_tok = nn.Linear(d_model, bottleneck, bias=False)
            self.w_tok = nn.Parameter(torch.full((n_tok, bottleneck), 1.0 / n_tok))
        if use_word:
            self.down_word = nn.Linear(d_model, bottleneck, bias=False)
            self.w_word = nn.Parameter(torch.full((n_word, bottleneck), 1.0 / n_word))
        if use_sent:
            self.down_sent = nn.Linear(d_model, bottleneck, bias=False)
            self.w_sent = nn.Parameter(torch.full((n_sent, bottleneck), 1.0 / n_sent))
        n_streams = int(use_tok) + int(use_word) + int(use_sent)
        self.up = nn.Linear(n_streams * bottleneck, d_model, bias=False)
        nn.init.zeros_(self.up.weight)

    # ---------------------------------------------------------------- full sequence (training / prefill)
    def forward(self, x, word_end, sent_end, valid=None, return_state=False):
        B, T, _ = x.shape
        if valid is None:
            valid = torch.ones(B, T, dtype=torch.bool, device=x.device)
        vm = valid.unsqueeze(-1).to(x.dtype)
        outs = []
        tok_buf = word_buf = sent_buf = None

        if self.use_tok:
            Z = self.down_tok(x) * vm
            acc = self.w_tok[0] * Z
            for k in range(1, self.n_tok):
                acc = acc + self.w_tok[k] * F.pad(Z, (0, 0, k, 0))[:, :T]
            outs.append(acc)
            if return_state:
                Zp = F.pad(Z, (0, 0, self.n_tok - 1, 0))
                tok_buf = Zp[:, -(self.n_tok - 1):].flip(1)       # index 0 = most recent position

        if self.use_word:
            outs_w, word_buf = self._boundary_stream(self.down_word(x) * vm, word_end & valid,
                                                     self.w_word, self.n_word, return_state)
            outs.append(outs_w)

        if self.use_sent:
            outs_s, sent_buf = self._boundary_stream(self.down_sent(x) * vm, sent_end & valid,
                                                     self.w_sent, self.n_sent, return_state)
            outs.append(outs_s)

        out = x + self.up(torch.cat(outs, dim=-1))
        if return_state:
            return out, (tok_buf, word_buf, sent_buf)
        return out

    @staticmethod
    def _boundary_stream(Z, flags, weights, n, want_state):
        prev, upto = _prev_and_upto(flags)
        hops = _chain(prev, prev, n)
        acc = 0
        for h in range(n):
            acc = acc + weights[h] * _gather(Z, hops[h])
        state = None
        if want_state:
            nxt = _chain(prev, upto[:, -1:], n)                     # the n latest boundaries <= T-1
            state = torch.cat([_gather(Z, i) for i in nxt], dim=1)  # (B, n, b), index 0 = most recent
        return acc, state

    # ---------------------------------------------------------------- one new token (generation)
    def step(self, x, word_end, sent_end, state):
        """x (B, 1, d); word_end, sent_end (B, 1) bool; state from forward(..., return_state=True) or a previous step."""
        tok_buf, word_buf, sent_buf = state
        outs = []
        new_tok = new_word = new_sent = None
        # buffers made during prefill and the fresh vectors can differ in dtype under autocast; unify them
        dt = self.up.weight.dtype if not torch.is_autocast_enabled() else torch.float32
        if tok_buf is not None:
            tok_buf = tok_buf.to(dt)
        if word_buf is not None:
            word_buf = word_buf.to(dt)
        if sent_buf is not None:
            sent_buf = sent_buf.to(dt)
        if self.use_tok:
            Z = self.down_tok(x).to(dt)
            acc = self.w_tok[0] * Z[:, 0]
            for k in range(1, self.n_tok):
                acc = acc + self.w_tok[k] * tok_buf[:, k - 1]
            outs.append(acc.unsqueeze(1))
            new_tok = torch.cat([Z, tok_buf[:, :-1]], dim=1)
        if self.use_word:
            acc = 0
            for h in range(self.n_word):
                acc = acc + self.w_word[h] * word_buf[:, h]
            outs.append(acc.unsqueeze(1))
            pushed = torch.cat([self.down_word(x).to(dt), word_buf[:, :-1]], dim=1)
            new_word = torch.where(word_end.unsqueeze(-1), pushed, word_buf)
        if self.use_sent:
            acc = 0
            for h in range(self.n_sent):
                acc = acc + self.w_sent[h] * sent_buf[:, h]
            outs.append(acc.unsqueeze(1))
            pushed = torch.cat([self.down_sent(x).to(dt), sent_buf[:, :-1]], dim=1)
            new_sent = torch.where(sent_end.unsqueeze(-1), pushed, sent_buf)
        out = x + self.up(torch.cat(outs, dim=-1))
        return out, (new_tok, new_word, new_sent)

    def extra_params(self):
        return sum(p.numel() for p in self.parameters())


# ---- Settings specific to this notebook ------------------------------
GQA_GROUP_SIZE = 4      # query heads that share one key/value head. 4 is exactly
                        # what ATTENTION_TYPE = "gqa" meant in the original notebooks.
CACHE_HEADROOM = 512    # slots reserved inside CONTEXT_LEN for GENERATED tokens.
                        # The prompt is cut to its last CONTEXT_LEN - CACHE_HEADROOM
                        # tokens; if generation fills the cache, the window is
                        # re-prefilled (see the markdown above for why). With
                        # CONTEXT_LEN = 2048 that leaves room for a 1536-token prompt
                        # (a 512-word legal prompt is roughly 700-1000 tokens) AND
                        # ~512 generated tokens, so in practice nothing is cut.
USE_NATIVE_GQA = False  # decided automatically by a micro-benchmark in the next
                        # section (True = pass enable_gqa=True to SDPA instead of
                        # expanding K/V with repeat_interleave)
POSITIONAL_ENCODING = "rope"   # "rope" | "learned"
                        # rope    = rotary embeddings: Q and K are rotated by a
                        #           position-dependent angle inside attention, so
                        #           attention scores depend on RELATIVE distance
                        #           (RoFormer, Su et al. 2021). No position table.
                        # learned = a trainable CONTEXT_LEN-row position table added to
                        #           the token embeddings (GPT-2 style, what the original
                        #           decoder-only notebook used).
ROPE_BASE = 10000.0     # RoPE frequency base: theta_i = ROPE_BASE ** (-2i / head_dim)
assert POSITIONAL_ENCODING in ("rope", "learned"), "POSITIONAL_ENCODING must be 'rope' or 'learned'"


def build_rope_cache(max_seq_len, head_dim, base=10000.0):
    """cos/sin tables for RoPE, each (max_seq_len, head_dim). Split-half
    convention: channel i is paired with channel i + head_dim/2, so the angle
    table is cat([freqs, freqs]) (NOT repeat_interleave, which pairs neighbours
    instead). Same construction as the RoPE written for Assignment 3."""
    assert head_dim % 2 == 0, "RoPE requires an even head dimension"
    inv_freq = 1.0 / (base ** (2 * torch.arange(0, head_dim // 2).float() / head_dim))
    positions = torch.arange(0, max_seq_len).float()
    freqs = torch.outer(positions, inv_freq)              # (max_seq_len, head_dim // 2)
    angles = torch.cat([freqs, freqs], dim=-1)            # (max_seq_len, head_dim)
    return angles.cos(), angles.sin()


def rotate_half(x):
    """Split the last dim in half and return (-x2, x1): a 90-degree rotation step."""
    x1, x2 = torch.chunk(x, 2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def apply_rope(x, cos, sin):
    """x_rotated = x * cos + rotate_half(x) * sin. x: (..., head_dim); cos/sin
    already broadcastable against x (the model gathers them per position). The
    arithmetic is done in fp32 (even under fp16 autocast) and cast back, so the
    rotation stays accurate and q/k keep the same dtype as v for SDPA."""
    xf = x.float()
    return (xf * cos + rotate_half(xf) * sin).to(x.dtype)


class KVCache:
    """Pre-allocated key/value storage for EVERY layer, written in place (the
    'static cache' idea from gpt-fast). Shapes are (batch, n_kv_heads, max_len,
    head_dim) -- n_kv_heads, not n_heads, which is where GQA's 4x saving
    lives. `length` is how many positions are currently filled (the same for
    every row in the batch: shorter prompts are left-padded)."""

    def __init__(self, n_layers, batch_size, n_kv_heads, max_len, head_dim, device, dtype):
        shape = (batch_size, n_kv_heads, max_len, head_dim)
        self.k = [torch.zeros(shape, device=device, dtype=dtype) for _ in range(n_layers)]
        self.v = [torch.zeros(shape, device=device, dtype=dtype) for _ in range(n_layers)]
        self.length = 0
        self.max_len = max_len
        self.canon = None      # per layer: [state of Canon site A, state of Canon site C]; filled by prefill, updated by decode_step

    def reset(self):
        self.length = 0

    def num_bytes(self):
        return sum(t.numel() * t.element_size() for t in self.k + self.v)


class CachedGQAttention(nn.Module):
    """Grouped-query self-attention (causal) with two entry points:
      forward()      -- the parallel form used for training and for PREFILL.
                        Optionally also stores this call's keys/values in a cache.
      decode_step()  -- ONE new token, reading/writing the cache.
    Parameter names (q_proj/k_proj/v_proj/out_proj) match the original
    MultiHeadAttention with ATTENTION_TYPE="gqa", so checkpoints are interchangeable."""

    def __init__(self, d_model, n_heads, n_kv_heads):
        super().__init__()
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = d_model // n_heads
        self.group_size = n_heads // n_kv_heads

        self.q_proj = nn.Linear(d_model, n_heads * self.head_dim)
        self.k_proj = nn.Linear(d_model, n_kv_heads * self.head_dim)
        self.v_proj = nn.Linear(d_model, n_kv_heads * self.head_dim)
        self.out_proj = nn.Linear(n_heads * self.head_dim, d_model)

    def forward(self, x, key_padding_mask=None, k_cache=None, v_cache=None, rope=None):
        """x: (batch, seq, d_model). key_padding_mask (optional): (batch, seq)
        bool, True = real (non-pad) position; leave None when nothing is
        padded so the fused kernels can run mask-free. If k_cache/v_cache are
        given, this call's (un-expanded) keys/values are written to
        positions [0:seq] of them -- that is the PREFILL step. rope (optional):
        (cos, sin), each (batch or 1, 1, seq, head_dim), gathered from the
        model's RoPE tables -- None when POSITIONAL_ENCODING == "learned"."""
        batch_size, seq_len, _ = x.shape
        q = self.q_proj(x).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.n_kv_heads, self.head_dim).transpose(1, 2)

        if rope is not None:
            # rotate BEFORE caching: the cache stores already-rotated keys (each
            # key is rotated by its own position exactly once, when it is written)
            q = apply_rope(q, *rope)
            k = apply_rope(k, *rope)

        if k_cache is not None:
            k_cache[:, :, :seq_len] = k.to(k_cache.dtype)
            v_cache[:, :, :seq_len] = v.to(v_cache.dtype)

        attn_mask = None
        if key_padding_mask is not None:
            positions = torch.arange(seq_len, device=x.device)
            causal = positions.unsqueeze(1) >= positions.unsqueeze(0)               # (seq, seq)
            allowed = causal.unsqueeze(0) & key_padding_mask.unsqueeze(1)           # (batch, seq, seq)
            # a query may always see itself -- guarantees no row is fully
            # masked (which would produce NaNs for pad-position queries)
            allowed = allowed | torch.eye(seq_len, dtype=torch.bool, device=x.device).unsqueeze(0)
            attn_mask = allowed.unsqueeze(1)                                        # (batch, 1, seq, seq)

        use_causal_flag = attn_mask is None
        if USE_NATIVE_GQA and self.group_size > 1:
            out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask,
                                                 is_causal=use_causal_flag, enable_gqa=True)
        else:
            if self.group_size > 1:
                k = k.repeat_interleave(self.group_size, dim=1)
                v = v.repeat_interleave(self.group_size, dim=1)
            out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask,
                                                 is_causal=use_causal_flag)

        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.n_heads * self.head_dim)
        return self.out_proj(out)

    def decode_step(self, x_t, k_cache, v_cache, cache_len, key_mask=None, rope=None):
        """x_t: (batch, d_model) -- ONE new position per row. Writes its K/V at
        index cache_len and attends over cache positions 0..cache_len
        (inclusive). key_mask (optional): (batch, cache_len + 1) bool, True =
        real key; None when the batch has no padding. rope (optional): (cos,
        sin), each (batch, head_dim) for this step's position."""
        batch_size = x_t.shape[0]

        # Fold the `group_size` query heads that share one K/V head into the
        # query-LENGTH axis: (batch, n_kv_heads, group_size, head_dim). The
        # cache is then used as-is (never expanded), and because the new token
        # is the last position no causal mask is needed.
        q = self.q_proj(x_t).view(batch_size, self.n_kv_heads, self.group_size, self.head_dim)
        k_new = self.k_proj(x_t).view(batch_size, self.n_kv_heads, self.head_dim)
        v_new = self.v_proj(x_t).view(batch_size, self.n_kv_heads, self.head_dim)

        if rope is not None:
            cos, sin = rope
            q = apply_rope(q, cos.view(batch_size, 1, 1, -1), sin.view(batch_size, 1, 1, -1))
            k_new = apply_rope(k_new, cos.view(batch_size, 1, -1), sin.view(batch_size, 1, -1))

        k_cache[:, :, cache_len] = k_new.to(k_cache.dtype)
        v_cache[:, :, cache_len] = v_new.to(v_cache.dtype)
        k = k_cache[:, :, :cache_len + 1]
        v = v_cache[:, :, :cache_len + 1]
        if q.dtype != k.dtype:
            q = q.to(k.dtype)

        attn_mask = None if key_mask is None else key_mask.view(batch_size, 1, 1, -1)
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)   # (batch, n_kv, group, head_dim)
        out = out.reshape(batch_size, self.n_heads * self.head_dim).to(x_t.dtype)
        return self.out_proj(out)


class CachedDecoderBlock(nn.Module):
    """Same block as DecoderOnlyBlock (same norm / deepnorm wiring), with a
    decode_step() that mirrors forward() for one position."""

    def __init__(self, d_model, n_heads, d_ff, n_layers):
        super().__init__()
        self.norm_position = "post" if NORM_TYPE == "deepnorm" else NORM_POSITION

        n_kv_heads = max(1, n_heads // GQA_GROUP_SIZE)
        self.attn = CachedGQAttention(d_model, n_heads, n_kv_heads)
        self.ff = nn.Sequential(nn.Linear(d_model, d_ff), nn.GELU(), nn.Linear(d_ff, d_model))
        self.norm1 = make_norm(d_model, NORM_TYPE)
        self.norm2 = make_norm(d_model, NORM_TYPE)

        # Boundary Canon modules (Section 7b): site A sits between norm1 and attention, site C between norm2 and the FFN
        self.canon_a = BoundaryCanon(d_model, **CANON_KW) if (USE_CANON and CANON_SITE_A) else None
        self.canon_c = BoundaryCanon(d_model, **CANON_KW) if (USE_CANON and CANON_SITE_C) else None
        if USE_CANON:
            assert self.norm_position == "pre", "Boundary Canon is wired for pre-norm blocks (NORM_POSITION='pre', NORM_TYPE != 'deepnorm')"

        if NORM_TYPE == "deepnorm":
            self.alpha = deepnorm_alpha(n_layers)
            deepnorm_init([self.attn.out_proj, self.ff[2]], n_layers)
        else:
            self.alpha = 1.0

    def _canon(self, slot, h, boundary, states):
        """Parallel Canon (training / prefill). slot 0 = site A, 1 = site C. boundary = (word_end, sent_end, valid).
        If `states` is a list, the Canon state needed for later one-token steps is stored in states[slot]."""
        mod = self.canon_a if slot == 0 else self.canon_c
        if mod is None:
            return h
        word_end, sent_end, valid = boundary
        if states is None:
            return mod(h, word_end, sent_end, valid)
        out, states[slot] = mod(h, word_end, sent_end, valid, return_state=True)
        return out

    def _canon_step(self, slot, h, boundary, states):
        """One-token Canon (decode). h: (batch, d_model); boundary = (word_end, sent_end), each (batch,) bool.
        Reads and updates states[slot]."""
        mod = self.canon_a if slot == 0 else self.canon_c
        if mod is None:
            return h
        word_end, sent_end = boundary
        out, states[slot] = mod.step(h.unsqueeze(1), word_end.unsqueeze(1), sent_end.unsqueeze(1), states[slot])
        return out.squeeze(1)

    def forward(self, x, key_padding_mask=None, k_cache=None, v_cache=None, rope=None, boundary=None, canon_states=None):
        """boundary = (word_end, sent_end, valid) for the Canon modules (None only when USE_CANON is False).
        canon_states: optional 2-slot list that receives the Canon states (prefill)."""
        if self.norm_position == "pre":
            x = x + self.attn(self._canon(0, self.norm1(x), boundary, canon_states), key_padding_mask=key_padding_mask,
                              k_cache=k_cache, v_cache=v_cache, rope=rope)
            x = x + self.ff(self._canon(1, self.norm2(x), boundary, canon_states))
        else:  # "post" (also deepnorm, with the alpha residual scale)
            x = self.norm1(x * self.alpha + self.attn(x, key_padding_mask=key_padding_mask,
                                                       k_cache=k_cache, v_cache=v_cache, rope=rope))
            x = self.norm2(x * self.alpha + self.ff(x))
        return x

    def decode_step(self, x_t, k_cache, v_cache, cache_len, key_mask=None, rope=None, boundary=None, canon_state=None):
        """x_t: (batch, d_model). Norms and the FFN act per position, so only
        the attention part needs the KV cache; the Canon modules need their small state (canon_state)."""
        if self.norm_position == "pre":
            a_in = self._canon_step(0, self.norm1(x_t), boundary, canon_state)
            x_t = x_t + self.attn.decode_step(a_in, k_cache, v_cache, cache_len, key_mask, rope)
            x_t = x_t + self.ff(self._canon_step(1, self.norm2(x_t), boundary, canon_state))
        else:
            x_t = self.norm1(x_t * self.alpha + self.attn.decode_step(x_t, k_cache, v_cache, cache_len, key_mask, rope))
            x_t = self.norm2(x_t * self.alpha + self.ff(x_t))
        return x_t


class CachedDecoderModel(nn.Module):
    """GPT-style decoder, GQA attention, KV-cache aware. Positions come from
    POSITIONAL_ENCODING: "learned" (pos_emb table added to the token
    embeddings) or "rope" (rotations inside attention, no table)."""

    def __init__(self, vocab_size, d_model, n_heads, n_layers, d_ff, context_len, word_end_table=None, sent_end_table=None):
        super().__init__()
        if USE_CANON:
            assert word_end_table is not None and sent_end_table is not None, "USE_CANON needs the word/sentence-end tables"
            assert len(word_end_table) == vocab_size and len(sent_end_table) == vocab_size
            # one bool per vocabulary entry: does this token END a word / a sentence? Derived from the tokenizer, not learned,
            # and not saved in checkpoints (persistent=False) -- they are rebuilt from the tokenizer every run.
            self.register_buffer("word_end_table", word_end_table.clone(), persistent=False)
            self.register_buffer("sent_end_table", sent_end_table.clone(), persistent=False)
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.use_rope = (POSITIONAL_ENCODING == "rope")
        if self.use_rope:
            cos, sin = build_rope_cache(context_len, d_model // n_heads, ROPE_BASE)
            # tables, not parameters; persistent=False keeps them out of state_dict
            # (one pair shared by every layer -- they depend only on position)
            self.register_buffer("rope_cos", cos, persistent=False)
            self.register_buffer("rope_sin", sin, persistent=False)
        else:
            self.pos_emb = nn.Embedding(context_len, d_model)
        self.blocks = nn.ModuleList([
            CachedDecoderBlock(d_model, n_heads, d_ff, n_layers) for _ in range(n_layers)
        ])
        self.fc_out = nn.Linear(d_model, vocab_size)

    def _embed(self, token_ids, position_ids):
        """Token embedding (+ learned position embedding when not using RoPE)."""
        h = self.token_emb(token_ids)
        if not self.use_rope:
            h = h + self.pos_emb(position_ids)
        return h

    def _boundary(self, token_ids, valid):
        """(word_end, sent_end, valid) flags for the Canon modules, looked up from the token ids."""
        if not USE_CANON:
            return None
        return self.word_end_table[token_ids], self.sent_end_table[token_ids], valid

    def forward(self, x, key_padding_mask=None, position_ids=None, last_position_only=False):
        """Training / full-recompute forward. x: (batch, seq) token ids.
        position_ids (optional, (batch, seq)) lets padded batches give each
        row's first real token position 0. last_position_only=True skips the
        big vocab projection for every position but the last (used by the
        uncached reference generator)."""
        batch_size, seq_len = x.shape
        if position_ids is None:
            position_ids = torch.arange(seq_len, device=x.device).unsqueeze(0)
        h = self._embed(x, position_ids)
        rope = None
        if self.use_rope:
            cos, sin = self.rope_cos[position_ids], self.rope_sin[position_ids]   # (batch or 1, seq, head_dim)
            rope = (cos.unsqueeze(1), sin.unsqueeze(1))                            # add the heads axis
        boundary = self._boundary(x, key_padding_mask)      # key_padding_mask (True = real token) doubles as the Canon 'valid' mask
        for block in self.blocks:
            h = block(h, key_padding_mask=key_padding_mask, rope=rope, boundary=boundary)
        if last_position_only:
            h = h[:, -1, :]
        return self.fc_out(h)

    def prefill(self, x, pad_lens, cache):
        """PREFILL. x: (batch, P) token ids, LEFT-padded; pad_lens: list of
        ints (pad count per row). Fills cache positions [0:P] for every layer,
        sets cache.length = P, and returns ONLY the last position's logits
        (batch, vocab) -- the one thing generation needs."""
        batch_size, prefill_len = x.shape
        assert prefill_len <= cache.max_len, "prompt window longer than the cache"
        pad_t = torch.tensor(pad_lens, dtype=torch.long, device=x.device)
        arange = torch.arange(prefill_len, device=x.device).unsqueeze(0)
        position_ids = (arange - pad_t.unsqueeze(1)).clamp(min=0)   # first REAL token -> position 0
        key_padding_mask = (arange >= pad_t.unsqueeze(1)) if any(p > 0 for p in pad_lens) else None

        h = self._embed(x, position_ids)
        rope = None
        if self.use_rope:
            cos, sin = self.rope_cos[position_ids], self.rope_sin[position_ids]
            rope = (cos.unsqueeze(1), sin.unsqueeze(1))
        boundary = self._boundary(x, key_padding_mask)
        cache.canon = [[None, None] for _ in self.blocks] if USE_CANON else None
        for layer_idx, block in enumerate(self.blocks):
            h = block(h, key_padding_mask=key_padding_mask,
                      k_cache=cache.k[layer_idx], v_cache=cache.v[layer_idx], rope=rope,
                      boundary=boundary, canon_states=cache.canon[layer_idx] if USE_CANON else None)
        cache.length = prefill_len
        return self.fc_out(h[:, -1, :])

    def decode_step(self, token_ids, pad_t, has_padding, cache):
        """DECODE. token_ids: (batch,) -- ONE new token per row. pad_t: (batch,)
        long tensor of pad counts. Appends to the cache, returns logits
        (batch, vocab) for the NEXT token. O(1) new work per step (plus one
        read of the cache)."""
        cache_len = cache.length
        assert cache_len < cache.max_len, "cache is full -- re-prefill before decoding further"
        position_ids = (cache_len - pad_t).clamp(min=0)              # (batch,)
        key_mask = None
        if has_padding:
            key_mask = torch.arange(cache_len + 1, device=token_ids.device).unsqueeze(0) >= pad_t.unsqueeze(1)

        h = self._embed(token_ids, position_ids)
        rope = (self.rope_cos[position_ids], self.rope_sin[position_ids]) if self.use_rope else None   # each (batch, head_dim)
        boundary = (self.word_end_table[token_ids], self.sent_end_table[token_ids]) if USE_CANON else None
        for layer_idx, block in enumerate(self.blocks):
            h = block.decode_step(h, cache.k[layer_idx], cache.v[layer_idx], cache_len, key_mask, rope,
                                  boundary=boundary, canon_state=cache.canon[layer_idx] if USE_CANON else None)
        cache.length = cache_len + 1
        return self.fc_out(h)