# Phase 1 · `5-boundary-canon`: Boundary Canon layers

Our own idea: add small "Canon" modules that mix in the previous tokens, words and sentences around each position. Notebook: `22-boundary-canon-layers.ipynb`. Submission: `submissions/5-boundary-canon/`.

## Architecture diagram

One decoder block with the two Canon modules added:

```mermaid
flowchart TD
    X["Block input"] --> N1["LayerNorm"]
    N1 --> CA["Canon module A<br/>token stream: this + 3 before<br/>word stream: last 3 words<br/>sentence stream: last 2 sentences<br/>each squeezed to 128 numbers"]
    CA --> A["GQA attention<br/>12 × 64, 3 KV heads"]
    A --> R1["+ residual"]
    R1 --> N2["LayerNorm"]
    N2 --> CC["Canon module C<br/>(same three streams)"]
    CC --> F["Feed-forward 768 → 3072 → 768"]
    F --> R2["+ residual"]
```

| | |
|---|---|
| Backbone | the dense GQA model: 8 layers, 12 query heads × 64, 3 key/value heads, FFN 3072, RoPE, context 2,048 |
| Canon modules | two per layer: one after the first norm (right before attention) and one after the second norm (right before the feed-forward) |
| Bottleneck | each stream compresses the 768-d state to **128** numbers before mixing |
| Token stream | this token and the 3 before it (4 positions) |
| Word stream | the last **3 finished words** (a word ends at a token whose string ends with `</w>`) |
| Sentence stream | the last **2 finished sentences** (a sentence ends at `.`, `?` or `!`) |
| Parameters | **120,551,488 (120.6M)** |
| Training | **240M words** (60M per source), 1 epoch, batch 2, AdamW, learning rate 3e-4 |

Result: BERTScore F1 0.8054.
