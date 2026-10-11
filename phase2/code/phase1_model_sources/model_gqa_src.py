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
    """GPT-style decoder, GQA attention, KV-cache aware. Positions come from
    POSITIONAL_ENCODING: "learned" (pos_emb table added to the token
    embeddings) or "rope" (rotations inside attention, no table)."""

    def __init__(self, vocab_size, d_model, n_heads, n_layers, d_ff, context_len):
        super().__init__()
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
        for block in self.blocks:
            h = block(h, key_padding_mask=key_padding_mask, rope=rope)
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
        for layer_idx, block in enumerate(self.blocks):
            h = block(h, key_padding_mask=key_padding_mask,
                      k_cache=cache.k[layer_idx], v_cache=cache.v[layer_idx], rope=rope)
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
        for layer_idx, block in enumerate(self.blocks):
            h = block.decode_step(h, cache.k[layer_idx], cache.v[layer_idx], cache_len, key_mask, rope)
        cache.length = cache_len + 1
        return self.fc_out(h)