import math
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

# ---- Settings specific to this notebook (encoder-decoder) ---------------------------------------------------
GQA_GROUP_SIZE = 4      # query heads that share one key/value head (12 query heads -> 3 KV heads), in every attention layer
USE_NATIVE_GQA = False  # (kept for symmetry with the decoder-only notebook: K/V are expanded with repeat_interleave)
POSITIONAL_ENCODING = "rope"
assert POSITIONAL_ENCODING == "rope", "this notebook is RoPE-only"
ROPE_BASE = 10000.0

PAD_ID = 0              # "<pad>" is the tokenizer's first special token
MASK_ID = 2             # "<s>" is never produced by the text stream, so it doubles as the [MASK] token of the encoder loss

# Shapes (ENC_MAX / ENC_MIN / DEC_LEN) are defined in the training-examples cell, which runs before this one.
ENC_LAYOUT = ["bigdn"] * 6   # "bigdn" = bidirectional Gated DeltaNet block, "attn" = bidirectional GQA + RoPE block (see EncoderBlock)
N_ENC_LAYERS = len(ENC_LAYOUT)
N_DEC_LAYERS = 4        # the decoder: causal GQA self-attention + cross-attention to the prompt + MLP
assert ENC_MAX + DEC_LEN <= CONTEXT_LEN, "RoPE tables are CONTEXT_LEN long: ENC_MAX + DEC_LEN must fit"

# Optional extra loss on the encoder (BERT-style masked tokens). Only a fraction of each batch is corrupted, so the decoder
# mostly still sees clean prompts. ENC_MLM = False gives the plain prefix-LM objective.
ENC_MLM = True
MLM_BATCH_FRACTION = 0.25   # share of the examples in a batch whose encoder input is corrupted
MLM_MASK_RATE = 0.15        # share of those examples' prompt tokens that are selected (80% -> [MASK], 10% random, 10% kept)
MLM_WEIGHT = 0.5            # weight of the masked-token loss next to the decoder's next-token loss

# --- Novel modifications (beyond vanilla GDN encoder-decoder) ---
# 3. ENCODER LAYER MIXING: instead of using only the final encoder layer output for
#    cross-attention, each decoder layer attends to a LEARNED WEIGHTED COMBINATION of
#    all encoder layer outputs. Inspired by ELMo (Peters et al., 2018): different
#    encoder depths capture different features. Shallow GDN layers may recognise local
#    legal phrases ("under section 482"); deeper layers capture discourse structure and
#    argument flow. Each decoder layer learns which encoder depths matter most for its
#    role -- early decoder layers may weight shallow encoder features for lexical
#    grounding, while later decoder layers may weight deeper encoder features for
#    coherent continuation.
ENCODER_LAYER_MIXING = True   # False = use only final encoder output (vanilla)

# 4. GATED CROSS-ATTENTION: each decoder layer scales its cross-attention output by a
#    learned scalar gate before adding to the residual stream. In a continuation task
#    the decoder must balance faithfulness to the prompt (encoder) against fluency
#    (self-attention + FFN). Not every decoder layer needs the same amount of encoder
#    context: early layers may need strong prompt grounding, while later layers may rely
#    on the autoregressive context for surface fluency.
CROSS_GATE = True             # False = vanilla (all cross-attention residuals at 1.0)
CROSS_GATE_INIT = 0.5         # initial gate (0.5 = half the encoder signal initially)


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


import math

GDN_HEADS = 4          # heads per Gated DeltaNet layer
GDN_HEAD_DIM = 128     # key dim = value dim per head (inner width = 4 x 128 = 512)
GDN_CONV = 4           # width of the short causal depthwise convolution over q, k, v
GDN_CHUNK = 64         # tokens per chunk in the chunkwise form


def _l2norm(x, eps=1e-6):
    return x / (x.norm(dim=-1, keepdim=True) + eps)


def gdn_recurrent(q, k, v, log_alpha, beta, state=None):
    """REFERENCE (slow, one token at a time). q,k: (B,H,L,dk); v: (B,H,L,dv); log_alpha, beta: (B,H,L); state: (B,H,dv,dk) or None.
    Returns outputs (B,H,L,dv) and the final state."""
    B, H, L, dk = q.shape
    dv = v.shape[-1]
    S = q.new_zeros(B, H, dv, dk) if state is None else state
    outs = []
    for t in range(L):
        kt, vt, qt = k[:, :, t], v[:, :, t], q[:, :, t]
        a = log_alpha[:, :, t].exp()[..., None, None]
        b = beta[:, :, t][..., None, None]
        Sk = torch.einsum("bhvk,bhk->bhv", S, kt)
        S = a * (S - b * Sk.unsqueeze(-1) * kt.unsqueeze(-2)) + b * vt.unsqueeze(-1) * kt.unsqueeze(-2)
        outs.append(torch.einsum("bhvk,bhk->bhv", S, qt))
    return torch.stack(outs, dim=2), S


def gdn_chunkwise(q, k, v, log_alpha, beta, state=None, chunk=64):
    """Same function as gdn_recurrent, computed `chunk` tokens at a time.
    Inside a chunk (c_i = cumulative log-decay, gamma_i = exp(c_i)) the state is  S_i = gamma_i S0 + sum_{j<=i} (gamma_i/gamma_j) d_j k_j^T,
    where the 'corrected values' d_j solve a small lower-triangular system:  (I + A) D = beta*V - (beta*gamma*K) S0^T ,
    A_ij = beta_i (gamma_i/gamma_j) k_i.k_j for j < i.   With T = (I+A)^-1:  U = T (beta V),  W = T (beta gamma K),  D = U - W S0^T.
    Then   O = diag(gamma) Q S0^T + ((Q K^T) * Lambda) D ,   S_end = gamma_C S0 + D^T diag(gamma_C/gamma) K.
    Decay ratios are built as exp(c_i - c_j) AFTER masking j > i (never a division of cumulative products), so nothing over/underflows."""
    B, H, L, dk = q.shape
    dv = v.shape[-1]
    pad = (-L) % chunk
    if pad:   # right-pad with 'do nothing' tokens (k = v = q = 0, beta = 0, alpha = 1): the state is untouched by them
        q, k, v = (F.pad(x, (0, 0, 0, pad)) for x in (q, k, v))
        log_alpha, beta = (F.pad(x, (0, pad)) for x in (log_alpha, beta))
    N = (L + pad) // chunk
    q, k, v = (x.reshape(B, H, N, chunk, x.shape[-1]) for x in (q, k, v))
    la, be = (x.reshape(B, H, N, chunk) for x in (log_alpha, beta))
    c = la.cumsum(dim=-1)                                                         # (B,H,N,C) inclusive cumulative log-decay
    on_or_below = torch.tril(torch.ones(chunk, chunk, dtype=torch.bool, device=q.device))
    strictly_below = torch.tril(torch.ones(chunk, chunk, dtype=torch.bool, device=q.device), diagonal=-1)
    lam = (c.unsqueeze(-1) - c.unsqueeze(-2)).masked_fill(~on_or_below, float("-inf")).exp()   # gamma_i/gamma_j for j<=i, else 0
    A = (be.unsqueeze(-1) * (k @ k.transpose(-1, -2)) * lam).masked_fill(~strictly_below, 0.0)
    eye = torch.eye(chunk, dtype=q.dtype, device=q.device)
    T = torch.linalg.solve_triangular(eye + A, eye.expand_as(A), upper=False)      # (I + A)^-1
    u = T @ (be.unsqueeze(-1) * v)                                                 # (B,H,N,C,dv)
    w = T @ ((be * c.exp()).unsqueeze(-1) * k)                                     # (B,H,N,C,dk)
    qk = (q @ k.transpose(-1, -2)) * lam                                           # (B,H,N,C,C)
    q_g = c.exp().unsqueeze(-1) * q
    c_last = c[..., -1]                                                            # (B,H,N)
    k_dec = (c_last.unsqueeze(-1) - c).exp().unsqueeze(-1) * k
    S = q.new_zeros(B, H, dv, dk) if state is None else state
    outs = []
    for n in range(N):
        St = S.transpose(-1, -2)                                                   # (B,H,dk,dv)
        d = u[:, :, n] - w[:, :, n] @ St                                           # corrected values (B,H,C,dv)
        outs.append(q_g[:, :, n] @ St + qk[:, :, n] @ d)
        S = c_last[:, :, n].exp()[..., None, None] * S + d.transpose(-1, -2) @ k_dec[:, :, n]
    o = torch.stack(outs, dim=2).reshape(B, H, N * chunk, dv)[:, :, :L]
    return o, S


class GatedDeltaNetMixer(nn.Module):
    """The sequence-mixing part of a Gated DeltaNet layer (the block wraps it with the norm, residual and MLP).
    x -> q,k,v projections -> short causal depthwise conv + SiLU -> L2-normalised q,k -> delta-rule memory (gate alpha, step beta)
      -> per-head RMSNorm -> times SiLU(output gate) -> output projection.   Always computed in fp32 (the recurrence is the numerically
    sensitive part; it is a small share of the compute)."""

    def __init__(self, d_model, n_heads=GDN_HEADS, head_dim=GDN_HEAD_DIM, conv_width=GDN_CONV, chunk=GDN_CHUNK):
        super().__init__()
        self.n_heads, self.head_dim, self.conv_width, self.chunk = n_heads, head_dim, conv_width, chunk
        inner = n_heads * head_dim
        self.inner = inner
        self.q_proj = nn.Linear(d_model, inner, bias=False)
        self.k_proj = nn.Linear(d_model, inner, bias=False)
        self.v_proj = nn.Linear(d_model, inner, bias=False)
        self.g_proj = nn.Linear(d_model, inner, bias=False)          # output gate
        self.o_proj = nn.Linear(inner, d_model, bias=False)
        self.a_proj = nn.Linear(d_model, n_heads, bias=False)        # -> decay alpha_t (per head)
        self.b_proj = nn.Linear(d_model, n_heads, bias=False)        # -> step size beta_t (per head)
        self.conv = nn.Conv1d(3 * inner, 3 * inner, conv_width, groups=3 * inner, bias=False)   # depthwise, applied to [q;k;v]
        self.out_norm = RMSNorm(head_dim)
        # decay parameterisation (Mamba-2 style): alpha = exp(-exp(A_log) * softplus(a_proj(x) + dt_bias)), A in [1,16], dt in [1e-3, 1e-1]
        self.A_log = nn.Parameter(torch.log(torch.empty(n_heads).uniform_(1.0, 16.0)))
        dt = torch.exp(torch.rand(n_heads) * (math.log(0.1) - math.log(0.001)) + math.log(0.001))
        self.dt_bias = nn.Parameter(dt + torch.log(-torch.expm1(-dt)))        # inverse softplus of dt

    def _project(self, x):
        return torch.cat([self.q_proj(x), self.k_proj(x), self.v_proj(x)], dim=-1)

    def _split_heads(self, qkv_act):
        B, L, _ = qkv_act.shape
        q, k, v = qkv_act.split(self.inner, dim=-1)
        sh = lambda t: t.reshape(B, L, self.n_heads, self.head_dim).transpose(1, 2)    # (B,H,L,hd)
        q, k, v = sh(q), sh(k), sh(v)
        return _l2norm(q) * (self.head_dim ** -0.5), _l2norm(k), v

    def _decay_and_step(self, x):
        beta = torch.sigmoid(self.b_proj(x))                                            # (..., H)
        log_alpha = -self.A_log.exp() * F.softplus(self.a_proj(x) + self.dt_bias)       # (..., H), <= 0
        return log_alpha, beta

    def _finish(self, o, x):
        """o: (B,H,L,hd) memory read-out -> per-head norm -> output gate -> output projection."""
        B, H, L, hd = o.shape
        o = self.out_norm(o.transpose(1, 2))                                            # (B,L,H,hd)
        gate = F.silu(self.g_proj(x)).reshape(B, L, H, hd)
        return self.o_proj((o * gate).reshape(B, L, H * hd))

    def forward(self, x, key_padding_mask=None, return_state=False, use_recurrent=False):
        """x: (B,L,D). key_padding_mask (optional, (B,L) bool, True = real token): LEFT-padding positions are made 'do nothing' tokens
        (zero input, beta = 0, alpha = 1) so the state stays exactly zero until the first real token, whatever the padding."""
        dtype_in = x.dtype
        with torch.autocast(device_type=x.device.type, enabled=False):
            x = x.float()
            B, L, _ = x.shape
            if key_padding_mask is not None:
                x = x * key_padding_mask.unsqueeze(-1).to(x.dtype)
            qkv = self._project(x)                                                      # (B,L,3*inner) pre-conv
            conv_out = self.conv(F.pad(qkv.transpose(1, 2), (self.conv_width - 1, 0)))  # causal: left zero-pad, output length L
            q, k, v = self._split_heads(F.silu(conv_out.transpose(1, 2)))
            log_alpha, beta = self._decay_and_step(x)                                   # (B,L,H)
            if key_padding_mask is not None:
                m = key_padding_mask.unsqueeze(-1).to(x.dtype)
                log_alpha, beta = log_alpha * m, beta * m
            log_alpha, beta = log_alpha.transpose(1, 2), beta.transpose(1, 2)           # (B,H,L)
            if use_recurrent:
                o, S = gdn_recurrent(q, k, v, log_alpha, beta)
            else:
                o, S = gdn_chunkwise(q, k, v, log_alpha, beta, chunk=self.chunk)
            out = self._finish(o, x)
            if return_state:
                keep = self.conv_width - 1                                              # last pre-conv rows = the decode conv buffer
                buf = qkv[:, -keep:]
                if buf.shape[1] < keep:
                    buf = F.pad(buf, (0, 0, keep - buf.shape[1], 0))
                return out.to(dtype_in), S, buf
        return out.to(dtype_in)

    def step(self, x_t, S, conv_buf):
        """ONE new token per row. x_t: (B,D); S: (B,H,hd,hd) memory; conv_buf: (B,conv_width-1,3*inner) last pre-conv rows.
        Returns (out (B,D), new S, new conv_buf). O(1) work and memory per token."""
        dtype_in = x_t.dtype
        with torch.autocast(device_type=x_t.device.type, enabled=False):
            x = x_t.float()
            qkv_t = self._project(x)                                                    # (B,3*inner)
            window = torch.cat([conv_buf, qkv_t.unsqueeze(1)], dim=1)                   # (B,conv_width,3*inner), oldest first
            w = self.conv.weight.squeeze(1)                                             # (3*inner, conv_width)
            conv_out = (window * w.t().unsqueeze(0)).sum(dim=1)                         # same as the causal conv at this position
            q, k, v = self._split_heads(F.silu(conv_out).unsqueeze(1))                  # (B,H,1,hd)
            q, k, v = q[:, :, 0], k[:, :, 0], v[:, :, 0]                                # (B,H,hd)
            log_alpha, beta = self._decay_and_step(x)                                   # (B,H)
            a, b = log_alpha.exp()[..., None, None], beta[..., None, None]
            Sk = torch.einsum("bhvk,bhk->bhv", S, k)
            S = a * (S - b * Sk.unsqueeze(-1) * k.unsqueeze(-2)) + b * v.unsqueeze(-1) * k.unsqueeze(-2)
            o = torch.einsum("bhvk,bhk->bhv", S, q).unsqueeze(2)                        # (B,H,1,hd)
            out = self._finish(o, x.unsqueeze(1))[:, 0]
        return out.to(dtype_in), S, window[:, 1:]


class BiGDNEncoderBlock(nn.Module):
    """ENCODER block built on Gated DeltaNet. The memory update is causal (left to right), so to let every prompt token see the whole
    prompt the block runs TWO mixers: one reads the prompt left-to-right, the other right-to-left (the sequence is flipped, mixed
    causally, and flipped back); their outputs are added. Pre-norm residual block with the same GELU MLP as every other block.
    Cost is linear in the prompt length. Left padding is harmless: pad positions are 'do nothing' tokens in both directions."""

    def __init__(self, d_model, d_ff):
        super().__init__()
        assert NORM_POSITION == "pre" and NORM_TYPE != "deepnorm", "written for pre-norm blocks"
        self.fwd = GatedDeltaNetMixer(d_model)
        self.bwd = GatedDeltaNetMixer(d_model)
        self.ff = nn.Sequential(nn.Linear(d_model, d_ff), nn.GELU(), nn.Linear(d_ff, d_model))
        self.norm1, self.norm2 = make_norm(d_model, NORM_TYPE), make_norm(d_model, NORM_TYPE)

    def forward(self, x, rope, key_padding_mask):          # rope is unused: the recurrence carries position implicitly
        h = self.norm1(x)
        y = self.fwd(h, key_padding_mask)
        flipped_mask = None if key_padding_mask is None else key_padding_mask.flip(1)
        y = y + self.bwd(h.flip(1), flipped_mask).flip(1)
        x = x + y
        return x + self.ff(self.norm2(x))


class SelfAttention(nn.Module):
    """Grouped-query self-attention with RoPE. causal=False: the ENCODER (every prompt token sees the whole prompt).
    causal=True: the DECODER (parallel form for training; decode_step() for one token with a KV cache)."""

    def __init__(self, d_model, n_heads, n_kv_heads, causal):
        super().__init__()
        self.n_heads, self.n_kv_heads, self.causal = n_heads, n_kv_heads, causal
        self.head_dim = d_model // n_heads
        self.group_size = n_heads // n_kv_heads
        self.q_proj = nn.Linear(d_model, n_heads * self.head_dim)
        self.k_proj = nn.Linear(d_model, n_kv_heads * self.head_dim)
        self.v_proj = nn.Linear(d_model, n_kv_heads * self.head_dim)
        self.out_proj = nn.Linear(n_heads * self.head_dim, d_model)

    def forward(self, x, rope, key_padding_mask=None, k_cache=None, v_cache=None):
        """x: (B, S, d). rope: (cos, sin), each (B, 1, S, head_dim). key_padding_mask: (B, S) bool, True = real token (encoder only).
        If k_cache/v_cache are given the (rotated) keys/values are also written to positions [0:S]."""
        B, S, _ = x.shape
        q = self.q_proj(x).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, S, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, S, self.n_kv_heads, self.head_dim).transpose(1, 2)
        q, k = apply_rope(q, *rope), apply_rope(k, *rope)
        if k_cache is not None:
            k_cache[:, :, :S] = k.to(k_cache.dtype)
            v_cache[:, :, :S] = v.to(v_cache.dtype)
        if self.group_size > 1:
            k = k.repeat_interleave(self.group_size, dim=1)
            v = v.repeat_interleave(self.group_size, dim=1)
        if self.causal:
            assert key_padding_mask is None, "decoder sequences are never padded"
            out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        else:
            mask = None if key_padding_mask is None else key_padding_mask.view(B, 1, 1, S)
            out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        return self.out_proj(out.transpose(1, 2).contiguous().view(B, S, self.n_heads * self.head_dim))

    def decode_step(self, x_t, rope_t, k_cache, v_cache, cache_len):
        """ONE new decoder position per row. x_t: (B, d); rope_t: (cos, sin) each (B, head_dim). Writes K/V at index cache_len."""
        B = x_t.shape[0]
        q = self.q_proj(x_t).view(B, self.n_kv_heads, self.group_size, self.head_dim)     # query heads folded into the query axis
        k_new = self.k_proj(x_t).view(B, self.n_kv_heads, self.head_dim)
        v_new = self.v_proj(x_t).view(B, self.n_kv_heads, self.head_dim)
        cos, sin = rope_t
        q = apply_rope(q, cos.view(B, 1, 1, -1), sin.view(B, 1, 1, -1))
        k_new = apply_rope(k_new, cos.view(B, 1, -1), sin.view(B, 1, -1))
        k_cache[:, :, cache_len] = k_new.to(k_cache.dtype)
        v_cache[:, :, cache_len] = v_new.to(v_cache.dtype)
        k, v = k_cache[:, :, :cache_len + 1], v_cache[:, :, :cache_len + 1]
        out = F.scaled_dot_product_attention(q.to(k.dtype), k, v)                           # (B, n_kv, group, head_dim)
        return self.out_proj(out.reshape(B, self.n_heads * self.head_dim).to(x_t.dtype))


class CrossAttention(nn.Module):
    """Decoder -> encoder attention (GQA). Queries come from the decoder, keys/values from the encoder output, which are computed
    ONCE per prompt (kv()) and reused at every decoding step.

    Positions: the prompt tokens sit at positions 0..L-1 and the decoder token number i sits at position L-1+i (the decoder's first
    input is the prompt's last token). RoPE is applied to the queries (decoder positions) and to the keys (prompt positions), so
    the attention score depends on how far back in the PROMPT a token is -- 'the sentence that just ended' is distinguishable from
    'a clause 600 tokens ago', which a plain position-free cross-attention cannot tell."""

    def __init__(self, d_model, n_heads, n_kv_heads):
        super().__init__()
        self.n_heads, self.n_kv_heads = n_heads, n_kv_heads
        self.head_dim = d_model // n_heads
        self.group_size = n_heads // n_kv_heads
        self.q_proj = nn.Linear(d_model, n_heads * self.head_dim)
        self.k_proj = nn.Linear(d_model, n_kv_heads * self.head_dim)
        self.v_proj = nn.Linear(d_model, n_kv_heads * self.head_dim)
        self.out_proj = nn.Linear(n_heads * self.head_dim, d_model)

    def kv(self, enc_out, enc_rope):
        B, S, _ = enc_out.shape
        k = self.k_proj(enc_out).view(B, S, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(enc_out).view(B, S, self.n_kv_heads, self.head_dim).transpose(1, 2)
        return apply_rope(k, *enc_rope), v

    def forward(self, x, enc_k, enc_v, q_rope, enc_mask):
        """x: (B, T, d) decoder states. enc_k/enc_v: (B, n_kv, S, head_dim). q_rope: (cos, sin) each (B, 1, T, head_dim).
        enc_mask: (B, S) bool, True = real prompt token (None if the batch has no padding)."""
        B, T, _ = x.shape
        q = apply_rope(self.q_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2), *q_rope)
        k, v = enc_k, enc_v
        if self.group_size > 1:
            k = k.repeat_interleave(self.group_size, dim=1)
            v = v.repeat_interleave(self.group_size, dim=1)
        mask = None if enc_mask is None else enc_mask.view(B, 1, 1, -1)
        out = F.scaled_dot_product_attention(q.to(k.dtype), k, v, attn_mask=mask)
        return self.out_proj(out.transpose(1, 2).contiguous().view(B, T, self.n_heads * self.head_dim).to(x.dtype))

    def decode_step(self, x_t, enc_k, enc_v, rope_t, enc_mask):
        B = x_t.shape[0]
        q = self.q_proj(x_t).view(B, self.n_kv_heads, self.group_size, self.head_dim)
        cos, sin = rope_t
        q = apply_rope(q, cos.view(B, 1, 1, -1), sin.view(B, 1, 1, -1))
        mask = None if enc_mask is None else enc_mask.view(B, 1, 1, -1)
        out = F.scaled_dot_product_attention(q.to(enc_k.dtype), enc_k, enc_v, attn_mask=mask)
        return self.out_proj(out.reshape(B, self.n_heads * self.head_dim).to(x_t.dtype))


class EncoderBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff):
        super().__init__()
        self.attn = SelfAttention(d_model, n_heads, max(1, n_heads // GQA_GROUP_SIZE), causal=False)
        self.ff = nn.Sequential(nn.Linear(d_model, d_ff), nn.GELU(), nn.Linear(d_ff, d_model))
        self.norm1, self.norm2 = make_norm(d_model, NORM_TYPE), make_norm(d_model, NORM_TYPE)

    def forward(self, x, rope, key_padding_mask):
        x = x + self.attn(self.norm1(x), rope, key_padding_mask=key_padding_mask)
        return x + self.ff(self.norm2(x))


class DecoderBlock(nn.Module):
    """causal self-attention -> cross-attention over the prompt -> feed-forward (pre-norm, residual around each)."""

    def __init__(self, d_model, n_heads, d_ff):
        super().__init__()
        n_kv = max(1, n_heads // GQA_GROUP_SIZE)
        self.self_attn = SelfAttention(d_model, n_heads, n_kv, causal=True)
        self.cross_attn = CrossAttention(d_model, n_heads, n_kv)
        self.ff = nn.Sequential(nn.Linear(d_model, d_ff), nn.GELU(), nn.Linear(d_ff, d_model))
        self.norm1, self.norm2, self.norm3 = (make_norm(d_model, NORM_TYPE) for _ in range(3))
        # Novel: gated cross-attention residual
        if CROSS_GATE:
            self.cross_gate = nn.Parameter(torch.tensor(CROSS_GATE_INIT))
        else:
            self.register_buffer('cross_gate', torch.tensor(1.0))

    def forward(self, x, rope, enc_k, enc_v, enc_mask, k_cache=None, v_cache=None):
        x = x + self.self_attn(self.norm1(x), rope, k_cache=k_cache, v_cache=v_cache)
        x = x + self.cross_gate * self.cross_attn(self.norm2(x), enc_k, enc_v, rope, enc_mask)
        return x + self.ff(self.norm3(x))

    def decode_step(self, x_t, rope_t, enc_k, enc_v, enc_mask, k_cache, v_cache, cache_len):
        x_t = x_t + self.self_attn.decode_step(self.norm1(x_t), rope_t, k_cache, v_cache, cache_len)
        x_t = x_t + self.cross_gate * self.cross_attn.decode_step(self.norm2(x_t), enc_k, enc_v, rope_t, enc_mask)
        return x_t + self.ff(self.norm3(x_t))


class EncDecCache:
    """Everything generation needs per batch: the cross-attention keys/values of every decoder layer (computed once from the
    prompt), the decoder's own self-attention KV cache, and the prompt padding mask."""

    def __init__(self, model, batch_size, device, dtype):
        n_kv, hd = model.n_kv_heads, model.head_dim
        self.self_k = [torch.zeros(batch_size, n_kv, DEC_LEN, hd, device=device, dtype=dtype) for _ in model.dec_blocks]
        self.self_v = [torch.zeros_like(t) for t in self.self_k]
        self.cross_k, self.cross_v = [], []
        self.enc_mask = None
        self.start_pos = None     # (B,) long: position of the decoder's first input token (= prompt length - 1)
        self.length = 0           # decoder positions filled so far
        self.max_len = DEC_LEN

    def num_bytes(self):
        return sum(t.numel() * t.element_size() for t in self.self_k + self.self_v + self.cross_k + self.cross_v)


class EncoderDecoderLM(nn.Module):
    """Prefix-LM encoder-decoder. The ENCODER (bidirectional, RoPE) reads the prompt; the DECODER (causal, RoPE) writes the continuation,
    attending to its own past and, through cross-attention, to the encoder output. The token embedding is shared by encoder and
    decoder; the output projection is separate (as in the decoder-only notebook)."""

    def __init__(self, vocab_size, d_model, n_heads, enc_layout, n_dec_layers, d_ff, context_len):
        super().__init__()
        self.n_heads, self.head_dim = n_heads, d_model // n_heads
        self.n_kv_heads = max(1, n_heads // GQA_GROUP_SIZE)
        self.token_emb = nn.Embedding(vocab_size, d_model)
        cos, sin = build_rope_cache(context_len, self.head_dim, ROPE_BASE)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)
        self.enc_blocks = nn.ModuleList([BiGDNEncoderBlock(d_model, d_ff) if kind == "bigdn" else EncoderBlock(d_model, n_heads, d_ff) for kind in enc_layout])
        self.enc_norm = make_norm(d_model, NORM_TYPE)
        self.dec_blocks = nn.ModuleList([DecoderBlock(d_model, n_heads, d_ff) for _ in range(n_dec_layers)])
        self.dec_norm = make_norm(d_model, NORM_TYPE)
        self.fc_out = nn.Linear(d_model, vocab_size)
        # Novel: encoder layer mixing (each decoder layer learns its own weighted view of encoder layers)
        n_enc = len(enc_layout)
        if ENCODER_LAYER_MIXING:
            init_w = torch.full((n_dec_layers, n_enc), -10.0)
            init_w[:, -1] = 0.0   # init: only the final encoder layer contributes (= vanilla)
            self.enc_mix_logits = nn.Parameter(init_w)
        else:
            self.enc_mix_logits = None

    def _rope(self, positions):
        """positions (B, S) long -> (cos, sin), each (B, 1, S, head_dim)."""
        return self.rope_cos[positions].unsqueeze(1), self.rope_sin[positions].unsqueeze(1)

    def encode(self, enc_ids, enc_mask, enc_pos):
        """enc_ids (B, S) left-padded prompt ids; enc_mask (B, S) bool or None; enc_pos (B, S) positions (first real token = 0).
        Returns the final encoder states (B, S, d) and the encoder RoPE (needed for the cross-attention keys)."""
        rope = self._rope(enc_pos)
        h = self.token_emb(enc_ids)
        if self.enc_mix_logits is not None:
            self._enc_intermediates = []
        for block in self.enc_blocks:
            h = block(h, rope, enc_mask)
            if self.enc_mix_logits is not None:
                self._enc_intermediates.append(h)
        return self.enc_norm(h), rope

    def forward(self, enc_ids, dec_ids, enc_mask, enc_pos, dec_pos, last_position_only=False, return_enc=False):
        """Training / reference forward. dec_ids (B, T): the decoder INPUT tokens (the prompt's last token, then the continuation
        shifted right); dec_pos (B, T): their positions (prompt_length - 1 + i). Returns logits (B, T, V) over the next tokens."""
        enc_h, enc_rope = self.encode(enc_ids, enc_mask, enc_pos)
        rope = self._rope(dec_pos)
        h = self.token_emb(dec_ids)
        if self.enc_mix_logits is not None:
            stacked_enc = torch.stack(self._enc_intermediates, dim=0)  # (n_enc, B, S, d)
        for i, block in enumerate(self.dec_blocks):
            if self.enc_mix_logits is not None:
                w = F.softmax(self.enc_mix_logits[i], dim=0)          # (n_enc,)
                mixed = (stacked_enc * w.view(-1, 1, 1, 1)).sum(0)    # (B, S, d)
                k, v = block.cross_attn.kv(self.enc_norm(mixed), enc_rope)
            else:
                k, v = block.cross_attn.kv(enc_h, enc_rope)
            h = block(h, rope, k, v, enc_mask)
        h = self.dec_norm(h)
        if last_position_only:
            h = h[:, -1, :]
        logits = self.fc_out(h)
        return (logits, enc_h) if return_enc else logits

    def mlm_logits(self, enc_states):
        """Masked-token prediction from encoder states, through the (tied) token embedding matrix: no extra parameters."""
        return F.linear(enc_states, self.token_emb.weight)

    # ---------------- cached generation ----------------
    def start_generation(self, enc_ids, pad_lens, cache):
        """Run the encoder ONCE, store every decoder layer's cross-attention K/V, then feed the decoder's first input token (the
        prompt's last token) and return the logits for the first generated token (B, V)."""
        B, S = enc_ids.shape
        pad_t = torch.tensor(pad_lens, dtype=torch.long, device=enc_ids.device)
        arange = torch.arange(S, device=enc_ids.device).unsqueeze(0)
        enc_pos = (arange - pad_t.unsqueeze(1)).clamp(min=0)
        cache.enc_mask = (arange >= pad_t.unsqueeze(1)) if any(p > 0 for p in pad_lens) else None
        cache.start_pos = (S - pad_t - 1)
        enc_h, enc_rope = self.encode(enc_ids, cache.enc_mask, enc_pos)
        cache.cross_k, cache.cross_v = [], []
        if self.enc_mix_logits is not None:
            stacked_enc = torch.stack(self._enc_intermediates, dim=0)
        for i, block in enumerate(self.dec_blocks):
            if self.enc_mix_logits is not None:
                w = F.softmax(self.enc_mix_logits[i], dim=0)
                mixed = (stacked_enc * w.view(-1, 1, 1, 1)).sum(0)
                k, v = block.cross_attn.kv(self.enc_norm(mixed), enc_rope)
            else:
                k, v = block.cross_attn.kv(enc_h, enc_rope)
            cache.cross_k.append(k.to(cache.self_k[0].dtype))
            cache.cross_v.append(v.to(cache.self_k[0].dtype))
        cache.length = 0
        return self.decode_step(enc_ids[:, -1], cache)

    def decode_step(self, token_ids, cache):
        """ONE decoder token per row (token_ids (B,)). Appends to the self-attention cache; returns logits (B, V) for the NEXT token."""
        assert cache.length < cache.max_len, "decoder cache is full"
        pos = cache.start_pos + cache.length                                   # (B,)
        rope_t = (self.rope_cos[pos], self.rope_sin[pos])
        h = self.token_emb(token_ids)
        for i, block in enumerate(self.dec_blocks):
            h = block.decode_step(h, rope_t, cache.cross_k[i], cache.cross_v[i], cache.enc_mask,
                                  cache.self_k[i], cache.self_v[i], cache.length)
        cache.length += 1
        return self.fc_out(self.dec_norm(h))
