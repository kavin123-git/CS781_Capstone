# ===================== NER backbone wrappers for the other Phase-1 architectures =====================
# The Phase-1 model code above is UNCHANGED. Each wrapper only exposes the hidden states that feed the NER head.

def _unique(mods):
    seen, out = set(), []
    for m in mods:
        if id(m) not in seen:
            seen.add(id(m)); out.append(m)
    return out


class ForwardBackbone(nn.Module):
    """For decoder models with their own forward logic (MoR recursion + routing, Boundary Canon): the model's own forward()
    is called with the vocabulary projection replaced by the identity, so it returns its last hidden states."""

    def __init__(self, lm):
        super().__init__()
        self.original_param_count = sum(p.numel() for p in lm.parameters())
        self.lm = lm
        self.causal = True
        if isinstance(lm.fc_out, nn.Sequential):
            self.final_norm = lm.fc_out[0]
        else:
            self.final_norm = nn.Identity()
        lm.fc_out = nn.Identity()
        self.d_model = lm.token_emb.embedding_dim

    def layer_list(self):
        return _unique(list(self.lm.blocks))

    def final_modules_params(self):
        return list(self.final_norm.parameters())

    def embedding_weight(self):
        return self.lm.token_emb.weight

    def adapter_sites(self, kind):
        blocks = _unique(list(self.lm.blocks))
        if kind == "houlsby":
            return [(m, self.d_model) for m in _unique([x for b in blocks for x in (b.attn, b.ff)])]
        if kind == "pfeiffer":
            return [(m, self.d_model) for m in _unique([b.ff for b in blocks])]
        raise ValueError(kind)

    def enable_prefix(self, P, hidden=0):
        raise NotImplementedError("prefix tuning is only wired for the plain GQA/MoE decoders")

    def features(self, ids, mask, soft_prompt=None):
        assert soft_prompt is None, "prompt tuning is only wired for the plain GQA/MoE decoders"
        # right padding + causal model: real tokens never see the pads, so no mask is passed (as in Phase-1 training)
        return self.final_norm(self.lm(ids))


class LSTMBackbone(nn.Module):
    """The Phase-1 3-layer LSTM, split into one nn.LSTM per layer (weights copied exactly) so that adapters can sit between
    layers and partial fine-tuning can pick the top layers. Hidden states of the last layer (LSTM_HIDDEN wide) feed the head."""

    def __init__(self, lm):
        super().__init__()
        self.original_param_count = sum(p.numel() for p in lm.parameters())
        self.emb = lm.emb
        L, H = lm.lstm.num_layers, lm.lstm.hidden_size
        self.layers = nn.ModuleList()
        for k in range(L):
            layer = nn.LSTM(lm.lstm.input_size if k == 0 else H, H, num_layers=1, batch_first=True)
            with torch.no_grad():
                for n in ("weight_ih", "weight_hh", "bias_ih", "bias_hh"):
                    getattr(layer, f"{n}_l0").copy_(getattr(lm.lstm, f"{n}_l{k}"))
            self.layers.append(layer)
        self.taps = nn.ModuleList([nn.Identity() for _ in range(L)])    # adapter attachment points (after each layer)
        self.d_model = H
        self.causal = True

    def layer_list(self):
        return list(self.layers)

    def final_modules_params(self):
        return []

    def embedding_weight(self):
        return self.emb.weight

    def adapter_sites(self, kind):
        if kind == "houlsby":
            return [(t, self.d_model) for t in self.taps]
        if kind == "pfeiffer":
            return [(self.taps[-1], self.d_model)]
        raise ValueError(kind)

    def enable_prefix(self, P, hidden=0):
        raise NotImplementedError("prefix tuning does not apply to an LSTM")

    def features(self, ids, mask, soft_prompt=None):
        assert soft_prompt is None, "prompt tuning is not wired for the LSTM"
        h = self.emb(ids)
        for layer, tap in zip(self.layers, self.taps):
            layer.flatten_parameters()
            h, _ = layer(h)
            h = tap(h)
        return h


class GDNEncoderBackbone(nn.Module):
    """The ENCODER of the Phase-1 Gated-DeltaNet encoder-decoder (6 bidirectional GDN blocks + final norm). Its decoder and
    output head are dropped. Bidirectional by construction (a forward and a backward GDN mixer in every block)."""

    def __init__(self, lm):
        super().__init__()
        self.original_param_count = sum(p.numel() for p in lm.parameters())
        self.lm = lm
        lm.dec_blocks = nn.ModuleList()
        lm.fc_out = nn.Identity()
        lm.enc_mix_logits = None            # only used by the decoder; also stops it storing every layer's states
        self.causal = False
        self.d_model = lm.token_emb.embedding_dim

    def layer_list(self):
        return list(self.lm.enc_blocks)

    def final_modules_params(self):
        return list(self.lm.enc_norm.parameters())

    def embedding_weight(self):
        return self.lm.token_emb.weight

    def adapter_sites(self, kind):
        blocks = list(self.lm.enc_blocks)
        if kind == "houlsby":
            return [(m, self.d_model) for b in blocks for m in (b.fwd, b.bwd, b.ff)]
        if kind == "pfeiffer":
            return [(b.ff, self.d_model) for b in blocks]
        raise ValueError(kind)

    def enable_prefix(self, P, hidden=0):
        raise NotImplementedError("prefix tuning does not apply to the GDN encoder")

    def features(self, ids, mask, soft_prompt=None):
        assert soft_prompt is None, "prompt tuning is not wired for the GDN encoder"
        B, T = ids.shape
        pos = torch.arange(T, device=ids.device).unsqueeze(0).expand(B, T)
        enc_mask = None if bool(mask.all()) else mask            # the backward mixer must not read right-padding
        h, _ = self.lm.encode(ids, enc_mask, pos)
        return h


@torch.no_grad()
def gdn_sanity_loss(lm, docs, device, n_windows=6, enc_len=768, dec_len=256):
    """Phase-1 objective of the encoder-decoder on NER text: encoder reads enc_len tokens, decoder predicts the next dec_len."""
    lm.eval().to(device)
    tot, cnt = 0.0, 0
    for d in list(docs.values())[:n_windows]:
        ids = torch.tensor(d["ids"][:enc_len + dec_len], dtype=torch.long, device=device).unsqueeze(0)
        if ids.shape[1] < enc_len + 8:
            continue
        enc, rest = ids[:, :enc_len], ids[:, enc_len - 1:]
        dec_in, tgt = rest[:, :-1], rest[:, 1:]
        enc_pos = torch.arange(enc.shape[1], device=device).unsqueeze(0)
        dec_pos = (enc_len - 1 + torch.arange(dec_in.shape[1], device=device)).unsqueeze(0)
        logits = lm(enc, dec_in, None, enc_pos, dec_pos).float()
        tot += F.cross_entropy(logits.view(-1, logits.shape[-1]), tgt.reshape(-1), reduction="sum").item()
        cnt += tgt.numel()
    return tot / max(cnt, 1)


@torch.no_grad()
def lm_sanity_loss(lm, docs, device, n_windows=6, length=1024):
    """Next-token loss of the loaded Phase-1 model on a few NER-document windows (BEFORE the LM head is dropped).
    A correctly loaded model gives roughly its Phase-1 held-out loss; random weights give ~ln(vocab) = 10.6."""
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
