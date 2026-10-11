import torch, torch.nn as nn, torch.nn.functional as F, math, time, os, random

class LSTMLM(nn.Module):
    """Tied-embedding LSTM language model. forward(x) -> logits; forward(x, y) -> mean cross-entropy (computed inside, so with
    DataParallel only a scalar per GPU travels back). step(x, state) -> (logits of the LAST position, new state): used for generation."""
    def __init__(self, vocab, embed, hidden, layers):
        super().__init__()
        self.vocab = vocab
        self.emb = nn.Embedding(vocab, embed, padding_idx=0)
        self.lstm = nn.LSTM(embed, hidden, num_layers=layers, batch_first=True)
        self.proj = nn.Linear(hidden, embed)
        self.bias = nn.Parameter(torch.zeros(vocab))
        nn.init.normal_(self.emb.weight, std=0.02)
        with torch.no_grad():
            self.emb.weight[0].zero_()
        for name, p in self.lstm.named_parameters():
            if "bias" in name:
                nn.init.zeros_(p)
                n = p.numel() // 4
                p.data[n:2 * n].fill_(1.0)                 # forget-gate bias (ih and hh biases both 1 -> effective 2)
            else:
                nn.init.uniform_(p, -1.0 / math.sqrt(hidden), 1.0 / math.sqrt(hidden))

    def _logits(self, h):
        return F.linear(self.proj(h), self.emb.weight, self.bias)

    def forward(self, x, y=None):
        self.lstm.flatten_parameters()
        h, _ = self.lstm(self.emb(x))
        logits = self._logits(h)
        if y is None:
            return logits
        return F.cross_entropy(logits.float().reshape(-1, self.vocab), y.reshape(-1))

    def step(self, x, state=None):
        self.lstm.flatten_parameters()
        h, state = self.lstm(self.emb(x), state)
        return self._logits(h[:, -1]), state

