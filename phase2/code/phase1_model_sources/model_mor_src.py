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

import math

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


# =====================================================================================================================
#  LegalMoR : Mixture-of-Recursions with full-context deep passes, causal threshold routing and pass/depth embeddings.
#
#  BASE PAPER   Mixture-of-Recursions (Bae et al., NeurIPS 2025, arXiv 2507.10524). One shared block of layers is applied several times
#               (Middle-Cycle: unique first layer, shared middle block, unique last layer). A tiny router decides PER TOKEN whether it gets
#               another pass, so easy tokens stop early and hard tokens get more depth; skipped tokens keep their state.
#  OUR CHANGES  (each has a switch below so it can be ablated)
#    1. FULL-CONTEXT DEEP PASSES (MOR_FULL_CONTEXT): in the paper a deep pass attends only among the tokens still selected, so a hard token at
#       pass 3 sees about a third of the sequence -- and the tokens it cannot see are exactly the easy ones ("Section", "the appellant") it often
#       needs. Here only the selected tokens send queries and go through the FFN, but EVERY token supplies keys/values from its latest state
#       (a skipped token's frozen state). Deep passes keep the whole context; the FFN saving is unchanged.
#    2. CAUSAL THRESHOLD ROUTING (MOR_ROUTING): the paper's expert-choice router keeps the top-k scores of the whole sequence, so a token's
#       fate depends on later tokens (a leak, and impossible to reproduce while generating one token at a time). Here a token is kept iff its
#       OWN score beats a threshold tau_r; tau_r is a running quantile estimate (updated after each training step, never learned) that keeps
#       the intended share of tokens on average. Training and generation use the same rule; nothing looks at other tokens.
#    3. PASS + DEPTH EMBEDDINGS (MOR_PASS_EMBED / MOR_DEPTH_EMBED): the shared block cannot tell which pass it is on, and the unique last layer
#       receives rows with 1, 2 or 3 passes without knowing which. A learned vector per pass is added to the block input, and a learned
#       vector per reached depth is added before the last layer. Both start at zero.
#    4. RELAXED SHARING (MOR_SHARE_FFN = False): attention is shared across passes, but every pass has its own FFN. Compute is identical (the
#       same rows are processed); only the stored weights grow (~92M -> ~121M). FFNs act as key-value memories, so this adds storage for
#       facts and phrases where small models are short of capacity -- the paper itself found MoR behind a plain Transformer at ~135M.
#  Sizes: d_model 768, 12 heads (GQA, 3 KV heads), FFN 3072; layout 1 + 3 shared layers x 3 passes + 1 = up to 11 layers deep.
# =====================================================================================================================

# ---- Settings of the recursion (each is an ablation switch) ------------------------------------------------------
MOR_SHARED_LAYERS = 3        # layers in the shared middle block
MOR_RECURSIONS = 3           # passes through the shared block
MOR_KEEP = (1.0, 7 / 8, 6 / 8)   # target share of ALL tokens processed at pass 1, 2, 3. Gentle skipping on purpose (the paper uses 1, 2/3, 1/3): the per-pass FFNs see most tokens and little is dropped; compute saving is small
MOR_ROUTING = "threshold"    # "threshold" = our causal router | "none" = every token gets every pass (fixed-depth recursive baseline)
MOR_FULL_CONTEXT = True      # change 1. False = the paper's recursion-wise attention (deep passes see only the tokens still selected)
MOR_PASS_EMBED = True        # change 3a
MOR_DEPTH_EMBED = True       # change 3b
MOR_TAU_MOMENTUM = 0.1       # how fast each threshold follows the batch quantile. The router keeps pushing scores up, so a slow threshold lags: in 600-step CPU tests 0.01 let up to 53% of tokens reach full depth (target 33%); 0.1 kept 28-38%
MOR_SHARE_FFN = False        # False = each pass has its OWN FFN (attention stays shared): ~28M more parameters, same compute ("relaxed sharing")
                             # True  = the paper's full sharing (attention AND FFN reused at every pass)
MOR_QUERY_GATHER = "auto"    # speed: True = deep passes compute attention only for the selected queries (fewer FLOPs, needs an explicit mask);
                             # False = attention for every query, unselected rows discarded (fused causal kernel). "auto" = time both on the GPU, keep the faster
assert MOR_ROUTING in ("threshold", "none") and MOR_QUERY_GATHER in ("auto", True, False)
assert len(MOR_KEEP) == MOR_RECURSIONS and MOR_KEEP[0] == 1.0 and all(MOR_KEEP[i] >= MOR_KEEP[i + 1] > 0 for i in range(MOR_RECURSIONS - 1))
assert NORM_POSITION == "pre" and NORM_TYPE != "deepnorm", "the recursion block is pre-norm only"


class KVCache:
    """Pre-allocated key/value storage for EVERY layer application (first layer, each shared layer at each pass, last layer -- a shared layer has
    separate K/V per pass because its input differs), written in place. Also the per-pass selection history, which the paper-style attention
    (MOR_FULL_CONTEXT = False) needs to know which earlier tokens a deep pass may look at. `length` = filled positions (same for every row)."""

    def __init__(self, n_layers, batch_size, n_kv_heads, max_len, head_dim, device, dtype):
        shape = (batch_size, n_kv_heads, max_len, head_dim)
        self.k = [torch.zeros(shape, device=device, dtype=dtype) for _ in range(n_layers)]
        self.v = [torch.zeros(shape, device=device, dtype=dtype) for _ in range(n_layers)]
        self.sel = torch.ones(MOR_RECURSIONS, batch_size, max_len, dtype=torch.bool, device=device)
        self.length = 0
        self.max_len = max_len

    def reset(self):
        self.length = 0

    def num_bytes(self):
        return sum(t.numel() * t.element_size() for t in self.k + self.v + [self.sel])



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

        if NORM_TYPE == "deepnorm":
            self.alpha = deepnorm_alpha(n_layers)
            deepnorm_init([self.attn.out_proj, self.ff[2]], n_layers)
        else:
            self.alpha = 1.0

    def forward(self, x, key_padding_mask=None, k_cache=None, v_cache=None, rope=None):
        if self.norm_position == "pre":
            x = x + self.attn(self.norm1(x), key_padding_mask=key_padding_mask,
                              k_cache=k_cache, v_cache=v_cache, rope=rope)
            x = x + self.ff(self.norm2(x))
        else:  # "post" (also deepnorm, with the alpha residual scale)
            x = self.norm1(x * self.alpha + self.attn(x, key_padding_mask=key_padding_mask,
                                                       k_cache=k_cache, v_cache=v_cache, rope=rope))
            x = self.norm2(x * self.alpha + self.ff(x))
        return x

    def decode_step(self, x_t, k_cache, v_cache, cache_len, key_mask=None, rope=None):
        """x_t: (batch, d_model). Norms and the FFN act per position, so only
        the attention part needs the cache."""
        if self.norm_position == "pre":
            x_t = x_t + self.attn.decode_step(self.norm1(x_t), k_cache, v_cache, cache_len, key_mask, rope)
            x_t = x_t + self.ff(self.norm2(x_t))
        else:
            x_t = self.norm1(x_t * self.alpha + self.attn.decode_step(x_t, k_cache, v_cache, cache_len, key_mask, rope))
            x_t = self.norm2(x_t * self.alpha + self.ff(x_t))
        return x_t


class CachedDecoderModel(nn.Module):
    """Mixture-of-Recursions decoder (GQA + RoPE), KV-cache aware, same entry points as the other notebooks: forward() / prefill() / decode_step().
    self.blocks is the UNROLLED execution order  [first, shared_0..shared_{S-1} (pass 1), ... (pass R), last]  -- the shared layers appear once per
    pass but are the SAME modules (parameters are counted and optimised once), and position i in this list is also KV-cache slot i.
    The n_layers argument is ignored: the layout comes from MOR_SHARED_LAYERS and MOR_RECURSIONS."""

    def __init__(self, vocab_size, d_model, n_heads, n_layers, d_ff, context_len):
        super().__init__()
        assert POSITIONAL_ENCODING == "rope", "this notebook is RoPE-only"
        self.token_emb = nn.Embedding(vocab_size, d_model)
        cos, sin = build_rope_cache(context_len, d_model // n_heads, ROPE_BASE)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)
        S, R = MOR_SHARED_LAYERS, MOR_RECURSIONS
        first = CachedDecoderBlock(d_model, n_heads, d_ff, 2 + S)
        shared = [CachedDecoderBlock(d_model, n_heads, d_ff, 2 + S) for _ in range(S)]
        passes = [shared]
        for _ in range(1, R):
            if MOR_SHARE_FFN:
                passes.append(shared)
            else:   # relaxed sharing: a fresh block per pass whose attention (and its norm) IS the pass-1 module; its FFN (and norm) is its own
                row = []
                for j in range(S):
                    b = CachedDecoderBlock(d_model, n_heads, d_ff, 2 + S)
                    b.attn, b.norm1 = shared[j].attn, shared[j].norm1
                    row.append(b)
                passes.append(row)
        last = CachedDecoderBlock(d_model, n_heads, d_ff, 2 + S)
        self.blocks = nn.ModuleList([first] + [passes[r][j] for r in range(R) for j in range(S)] + [last])
        self.query_gather = (MOR_QUERY_GATHER is True)     # "auto" is resolved by the speed-probe cell on the GPU
        self.fc_out = nn.Linear(d_model, vocab_size)
        self.S, self.R = S, R
        self.routed = (MOR_ROUTING == "threshold")
        if self.routed:   # one router per pass that is routed (pass 1 takes every token)
            self.router_norm = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(R - 1)])
            self.router = nn.ModuleList([nn.Linear(d_model, 1) for _ in range(R - 1)])
            for lin in self.router:
                nn.init.normal_(lin.weight, std=0.02)
                nn.init.zeros_(lin.bias)
            self.register_buffer("tau", torch.zeros(R - 1))
            self.register_buffer("tau_ready", torch.zeros(R - 1, dtype=torch.bool))
            self._ready_py = [None] * (R - 1)   # python copy of tau_ready, read from the buffer once (avoids a GPU->CPU sync every step)
        self.pass_emb = nn.Parameter(torch.zeros(R, d_model)) if MOR_PASS_EMBED else None
        self.depth_emb = nn.Parameter(torch.zeros(R, d_model)) if MOR_DEPTH_EMBED else None

    def slot(self, r, j):
        """KV-cache slot / index in self.blocks of shared layer j at pass r (both 0-based)."""
        return 1 + r * self.S + j

    def _rope(self, position_ids):
        cos, sin = self.rope_cos[position_ids], self.rope_sin[position_ids]
        return (cos.unsqueeze(1), sin.unsqueeze(1))

    # ---- routing -------------------------------------------------------------------------------------------------------
    def _route(self, r, h, eligible):
        """Pass r >= 1 (0-based). h (...,d); eligible (...) bool = selected at the previous pass. Returns (selected, gate). The decision for a token
        uses ONLY its own score and the stored threshold, so it is causal. In training the threshold then moves towards the batch quantile that
        would keep the target share (used from the NEXT step on)."""
        z = self.router[r - 1](self.router_norm[r - 1](h)).squeeze(-1).float()
        tau = self.tau[r - 1].clone()          # decisions use the threshold from BEFORE this batch (no information from other tokens)
        if self.training:
            with torch.no_grad():   # no boolean indexing and no .item(): nothing here waits for the GPU
                q = torch.nanquantile(z.detach().masked_fill(~eligible, float("nan")).flatten(), 1.0 - MOR_KEEP[r] / MOR_KEEP[r - 1])
                old = self.tau[r - 1]
                if self._ready_py[r - 1] is None:
                    self._ready_py[r - 1] = bool(self.tau_ready[r - 1])
                new = old + MOR_TAU_MOMENTUM * (q - old) if self._ready_py[r - 1] else q
                self.tau[r - 1].copy_(torch.where(torch.isnan(q), old, new))
                self.tau_ready[r - 1] = True
                self._ready_py[r - 1] = True
        selected = eligible & (z > tau)
        gate = torch.sigmoid(z - tau)
        return selected, gate

    # ---- one pass through the shared block (parallel form) ------------------------------------------------------------
    def _pass(self, r, h, sel, key_padding_mask, rope, cache):
        """h (B,T,d). sel (B,T) bool or None (= every token). Only selected rows change; every row supplies keys/values (full context) unless
        MOR_FULL_CONTEXT is False, in which case keys are restricted to the selected rows (the paper's recursion-wise attention)."""
        x = h + self.pass_emb[r] if self.pass_emb is not None else h
        x_in = x
        kpm = key_padding_mask
        if sel is not None and not MOR_FULL_CONTEXT:
            kpm = sel if kpm is None else (kpm & sel)
        selm = None if sel is None else sel.unsqueeze(-1)
        for j in range(self.S):
            blk = self.blocks[self.slot(r, j)]
            kc, vc = (cache.k[self.slot(r, j)], cache.v[self.slot(r, j)]) if cache is not None else (None, None)
            if sel is not None and self.query_gather:
                a = self._attn_selected(blk.attn, blk.norm1(x), sel, kpm, rope, kc, vc)
            else:
                a = blk.attn(blk.norm1(x), key_padding_mask=kpm, k_cache=kc, v_cache=vc, rope=rope)
            if selm is None:
                x = x + a
                x = x + blk.ff(blk.norm2(x))
            else:
                x = x + a * selm.to(a.dtype)
                rows = x[sel]                                                    # (N, d): only the selected tokens go through the FFN
                x = x.masked_scatter(selm.expand_as(x), rows + blk.ff(blk.norm2(rows)))
        return x - x_in                                                          # the block's change (zero on unselected rows)

    def _attn_selected(self, att, xn, sel, key_mask, rope, kc, vc):
        """Same result as att(xn, ...) on the selected rows (zeros elsewhere), but queries are computed only for the selected tokens: each row's
        selected positions are gathered (in order, padded to the batch maximum), keys/values come from EVERY position (and are written to the
        cache exactly as the plain path does), and the causal mask uses the original positions."""
        B, T, D = xn.shape
        k = att.k_proj(xn).view(B, T, att.n_kv_heads, att.head_dim).transpose(1, 2)
        v = att.v_proj(xn).view(B, T, att.n_kv_heads, att.head_dim).transpose(1, 2)
        k = apply_rope(k, rope[0], rope[1])
        if kc is not None:
            kc[:, :, :T] = k.to(kc.dtype)
            vc[:, :, :T] = v.to(vc.dtype)
        counts = sel.sum(1)
        kmax = int(counts.max())
        if kmax == 0:
            return att.out_proj(xn.new_zeros(B, T, att.n_heads * att.head_dim)) * 0
        order = torch.sort(sel.to(torch.int8), dim=1, descending=True, stable=True).indices[:, :kmax]     # selected positions first, in order
        valid = torch.arange(kmax, device=xn.device).unsqueeze(0) < counts.unsqueeze(1)
        q = att.q_proj(xn.gather(1, order.unsqueeze(-1).expand(-1, -1, D))).view(B, kmax, att.n_heads, att.head_dim).transpose(1, 2)
        cos, sin = rope[0].expand(B, -1, -1, -1), rope[1].expand(B, -1, -1, -1)
        idx = order.unsqueeze(1).unsqueeze(-1).expand(-1, 1, -1, cos.shape[-1])
        q = apply_rope(q, cos.gather(2, idx), sin.gather(2, idx))
        jpos = torch.arange(T, device=xn.device).view(1, 1, T)
        allowed = jpos <= order.unsqueeze(-1)                                    # (B, kmax, T): causal on ORIGINAL positions
        if key_mask is not None:
            allowed = allowed & key_mask.unsqueeze(1)
        allowed = allowed | (jpos == order.unsqueeze(-1)) | ~valid.unsqueeze(-1)  # a query always sees itself; padded query rows are discarded
        if att.group_size > 1:
            k = k.repeat_interleave(att.group_size, dim=1)
            v = v.repeat_interleave(att.group_size, dim=1)
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=allowed.unsqueeze(1))
        out = att.out_proj(out.transpose(1, 2).reshape(B, kmax, att.n_heads * att.head_dim))
        out = out * valid.unsqueeze(-1).to(out.dtype)
        return out.new_zeros(B, T, out.shape[-1]).scatter_add(1, order.unsqueeze(-1).expand_as(out), out)

    def _trunk(self, x_ids, key_padding_mask, position_ids, cache=None):
        h = self.token_emb(x_ids)
        rope = self._rope(position_ids)
        kc = (cache.k[0], cache.v[0]) if cache is not None else (None, None)
        h = self.blocks[0](h, key_padding_mask=key_padding_mask, k_cache=kc[0], v_cache=kc[1], rope=rope)
        B, T = x_ids.shape
        sel = None
        depth = torch.ones(B, T, dtype=torch.long, device=x_ids.device)
        for r in range(self.R):
            if r == 0 or not self.routed:
                gate = None
                if self.routed:
                    sel = torch.ones(B, T, dtype=torch.bool, device=x_ids.device)
            else:
                sel, gate = self._route(r, h, sel)
                depth = depth + sel.long()
            if cache is not None and sel is not None:
                cache.sel[r, :, :T] = sel
            pass_sel = None if (r == 0 or not self.routed) else sel     # pass 1 takes every token: no gather needed
            delta = self._pass(r, h, pass_sel, key_padding_mask, rope, cache)
            h = h + (delta if gate is None else gate.unsqueeze(-1).to(delta.dtype) * delta)
        if not self.routed:
            depth = depth + (self.R - 1)
        if self.depth_emb is not None:
            h = h + self.depth_emb[depth - 1]
        kc = (cache.k[-1], cache.v[-1]) if cache is not None else (None, None)
        h = self.blocks[-1](h, key_padding_mask=key_padding_mask, k_cache=kc[0], v_cache=kc[1], rope=rope)
        self._last_depth = depth
        return h

    # ---- training / full recompute ---------------------------------------------------------------------------------
    def forward(self, x, key_padding_mask=None, position_ids=None, last_position_only=False):
        B, T = x.shape
        if position_ids is None:
            position_ids = torch.arange(T, device=x.device).unsqueeze(0)
        h = self._trunk(x, key_padding_mask, position_ids)
        if last_position_only:
            h = h[:, -1, :]
        return self.fc_out(h)

    # ---- generation --------------------------------------------------------------------------------------------------
    def prefill(self, x, pad_lens, cache):
        B, P = x.shape
        assert P <= cache.max_len, "prompt window longer than the cache"
        pad_t = torch.tensor(pad_lens, dtype=torch.long, device=x.device)
        arange = torch.arange(P, device=x.device).unsqueeze(0)
        position_ids = (arange - pad_t.unsqueeze(1)).clamp(min=0)
        key_padding_mask = (arange >= pad_t.unsqueeze(1)) if any(p > 0 for p in pad_lens) else None
        h = self._trunk(x, key_padding_mask, position_ids, cache=cache)
        cache.length = P
        return self.fc_out(h[:, -1, :])

    def decode_step(self, token_ids, pad_t, has_padding, cache):
        """ONE new token per row. A token that is NOT selected at a pass still writes its keys/values for that pass (from its frozen state), so
        later tokens can attend to it -- exactly what the parallel form does."""
        L = cache.length
        assert L < cache.max_len, "cache is full -- re-prefill before decoding further"
        B = token_ids.shape[0]
        dev = token_ids.device
        position_ids = (L - pad_t).clamp(min=0)
        real = torch.arange(L + 1, device=dev).unsqueeze(0) >= pad_t.unsqueeze(1) if has_padding else None
        rope = (self.rope_cos[position_ids], self.rope_sin[position_ids])
        h = self.token_emb(token_ids)
        h = self.blocks[0].decode_step(h, cache.k[0], cache.v[0], L, real, rope)
        sel = torch.ones(B, dtype=torch.bool, device=dev)
        depth = torch.ones(B, dtype=torch.long, device=dev)
        for r in range(self.R):
            gate = None
            if r > 0 and self.routed:
                sel, gate = self._route(r, h, sel)
                depth = depth + sel.long()
            cache.sel[r, :, L] = sel
            key_mask = real
            if self.routed and not MOR_FULL_CONTEXT:
                key_mask = cache.sel[r, :, :L + 1].clone() if real is None else (cache.sel[r, :, :L + 1] & real)
                key_mask[:, L] = True                             # a query always sees itself (as in the parallel form)
            x = h + self.pass_emb[r] if self.pass_emb is not None else h
            x_in = x
            for j in range(self.S):
                blk = self.blocks[self.slot(r, j)]
                x_new = blk.decode_step(x, cache.k[self.slot(r, j)], cache.v[self.slot(r, j)], L, key_mask, rope)
                x = torch.where(sel.unsqueeze(-1), x_new, x) if self.routed else x_new
            delta = x - x_in
            h = h + (delta if gate is None else gate.unsqueeze(-1).to(delta.dtype) * delta)
        if not self.routed:
            depth = depth + (self.R - 1)
        if self.depth_emb is not None:
            h = h + self.depth_emb[depth - 1]
        h = self.blocks[-1].decode_step(h, cache.k[-1], cache.v[-1], L, real, rope)
        cache.length = L + 1
        return self.fc_out(h)

    # ---- diagnostics for the report ----------------------------------------------------------------------------------
    @torch.no_grad()
    def diagnostics(self, x):
        """x (B,T) token ids. Returns per-token depth (B,T), the share of tokens at each depth, the thresholds and the mean gate of the selected
        tokens at each routed pass."""
        T = x.shape[1]
        gates = []
        if self.routed:
            orig = self._route
            def spy(r, h, eligible, _orig=orig):
                s, g = _orig(r, h, eligible)
                gates.append(g[s].float().mean().item() if s.any() else float("nan"))
                return s, g
            self._route = spy
        try:
            self._trunk(x, None, torch.arange(T, device=x.device).unsqueeze(0))
        finally:
            if self.routed:
                del self._route
        depth = self._last_depth
        out = {"depth": depth, "depth_share": [(depth == d + 1).float().mean().item() for d in range(self.R)],
               "mean_gate_selected": gates}
        if self.routed:
            out["tau"] = self.tau.tolist()
        return out
