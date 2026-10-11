# ===================== Backbone wrapper for the Phase-1 decoder models (GQA / MoE CachedDecoderModel) =====================
# The Phase-1 model code above is UNCHANGED. For NER we only (a) run its blocks and take the hidden states instead of the
# next-token logits, and (b) optionally replace the attention's forward with one that can drop the causal mask
# ("bidirectional") and/or prepend learned prefix keys/values (prefix tuning). With both options off it calls the
# original forward unchanged.

_ORIG_GQA_FORWARD = CachedGQAttention.forward


def ner_gqa_forward(self, x, key_padding_mask=None, k_cache=None, v_cache=None, rope=None):
    bidir = getattr(self, "ner_bidirectional", False)
    pk = getattr(self, "prefix_k", None)
    if not bidir and pk is None:
        return _ORIG_GQA_FORWARD(self, x, key_padding_mask=key_padding_mask, k_cache=k_cache, v_cache=v_cache, rope=rope)
    B, T, _ = x.shape
    q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
    k = self.k_proj(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
    v = self.v_proj(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
    if rope is not None:
        q = apply_rope(q, *rope)
        k = apply_rope(k, *rope)
    P = 0
    if pk is not None:                                          # learned prefix keys/values (no position: never rotated)
        P = pk.shape[1]
        k = torch.cat([pk.unsqueeze(0).expand(B, -1, -1, -1).to(k.dtype), k], dim=2)
        v = torch.cat([self.prefix_v.unsqueeze(0).expand(B, -1, -1, -1).to(v.dtype), v], dim=2)
    if self.group_size > 1:
        k = k.repeat_interleave(self.group_size, dim=1)
        v = v.repeat_interleave(self.group_size, dim=1)
    if bidir and P == 0 and key_padding_mask is None:          # full-attention fast path (no mask tensor at all)
        out = F.scaled_dot_product_attention(q, k, v, is_causal=False)
        out = out.transpose(1, 2).contiguous().view(B, T, self.n_heads * self.head_dim)
        return self.out_proj(out)
    pos = torch.arange(T, device=x.device)
    allowed = torch.ones(T, T, dtype=torch.bool, device=x.device) if bidir else (pos.unsqueeze(1) >= pos.unsqueeze(0))
    allowed = allowed.unsqueeze(0).expand(B, T, T)
    if key_padding_mask is not None:
        allowed = allowed & key_padding_mask.unsqueeze(1)
        allowed = allowed | torch.eye(T, dtype=torch.bool, device=x.device).unsqueeze(0)
    if P:
        allowed = torch.cat([torch.ones(B, T, P, dtype=torch.bool, device=x.device), allowed], dim=2)
    out = F.scaled_dot_product_attention(q, k, v, attn_mask=allowed.unsqueeze(1))
    out = out.transpose(1, 2).contiguous().view(B, T, self.n_heads * self.head_dim)
    return self.out_proj(out)


CachedGQAttention.forward = ner_gqa_forward


class DecoderBackbone(nn.Module):
    def __init__(self, lm, bidirectional=False):
        super().__init__()
        self.original_param_count = sum(p.numel() for p in lm.parameters())
        self.lm = lm
        self.causal = not bidirectional
        for blk in lm.blocks:
            blk.attn.ner_bidirectional = bidirectional
        if isinstance(lm.fc_out, nn.Sequential):                # the final LayerNorm is kept; the vocab projection is dropped
            self.final_norm = lm.fc_out[0]
        else:
            self.final_norm = nn.Identity()
        lm.fc_out = nn.Identity()
        self.d_model = lm.token_emb.embedding_dim
        self.max_len = lm.rope_cos.shape[0] if lm.use_rope else lm.pos_emb.num_embeddings

    def layer_list(self):
        return list(self.lm.blocks)

    def final_modules_params(self):
        return list(self.final_norm.parameters())

    def embedding_weight(self):
        return self.lm.token_emb.weight

    def adapter_sites(self, kind):
        d = self.d_model
        if kind == "houlsby":                                   # after attention AND after the feed-forward, every block
            return [(m, d) for blk in self.lm.blocks for m in (blk.attn, blk.ff)]
        if kind == "pfeiffer":                                  # after the feed-forward only
            return [(blk.ff, d) for blk in self.lm.blocks]
        raise ValueError(kind)

    def enable_prefix(self, P, hidden=0):
        for blk in self.lm.blocks:
            a = blk.attn
            a.prefix_k = nn.Parameter(torch.randn(a.n_kv_heads, P, a.head_dim) * 0.02)
            a.prefix_v = nn.Parameter(torch.randn(a.n_kv_heads, P, a.head_dim) * 0.02)

    def features(self, ids, mask, soft_prompt=None):
        lm = self.lm
        h = lm.token_emb(ids)
        B = ids.shape[0]
        P = 0
        if soft_prompt is not None:
            P = soft_prompt.shape[0]
            h = torch.cat([soft_prompt.unsqueeze(0).expand(B, -1, -1).to(h.dtype), h], dim=1)
            mask = torch.cat([torch.ones(B, P, dtype=torch.bool, device=mask.device), mask], dim=1)
        T = h.shape[1]
        assert T <= self.max_len, f"window ({T} incl. {P} prompt tokens) longer than the model's {self.max_len} positions"
        pos = torch.arange(T, device=ids.device)
        rope = None
        if lm.use_rope:
            rope = (lm.rope_cos[pos].unsqueeze(0).unsqueeze(0), lm.rope_sin[pos].unsqueeze(0).unsqueeze(0))
        else:
            h = h + lm.pos_emb(pos).unsqueeze(0)
        # right padding: with causal attention real tokens never see the pads, so no mask is needed (fast fused kernel);
        # bidirectional attention needs the padding mask
        kpm = None if (self.causal or bool(mask.all())) else mask
        for blk in lm.blocks:
            h = blk(h, key_padding_mask=kpm, rope=rope)
        h = self.final_norm(h)
        return h[:, P:]


@torch.no_grad()
def lm_sanity_loss(lm, docs, device, n_windows=6, length=1024):
    """Next-token loss of the loaded Phase-1 model on a few NER-document windows (BEFORE the LM head is dropped).
    A correctly loaded model gives roughly its Phase-1 held-out loss (~2.5-3.5 nats); random weights give ~10.6."""
    lm.eval().to(device)
    tot, cnt = 0.0, 0
    for d in list(docs.values())[:n_windows]:
        ids = torch.tensor(d["ids"][:length + 1], dtype=torch.long, device=device).unsqueeze(0)
        if ids.shape[1] < 2:
            continue
        logits = lm(ids[:, :-1]).float()
        tot += F.cross_entropy(logits.view(-1, logits.shape[-1]), ids[:, 1:].reshape(-1), reduction="sum").item()
        cnt += ids.shape[1] - 1
    return tot / max(cnt, 1)
