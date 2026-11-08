# Phase 2 · `batch3-moe-one-factor`

**Question.** Which single factor helps the MoE: learning rate, window size, rare-label oversampling, adapter size, a document header, or other techniques on the stacked recipe? **Finding.** The winners gained much less on the test fold than on our own fold (best single model 0.5485), so instead of tuning further we turned to unlabelled text in batch 4.

## Architecture

The pipeline all notebooks in this phase build on:

```mermaid
flowchart LR
    D["Judgment text"] --> W["Windows of 2,048 tokens<br/>stride 1,024"]
    W --> B["Phase-1 backbone<br/>(adapted)"]
    B --> H["Head: linear,<br/>or BiLSTM from batch 2"]
    H --> T["BIO tag for each token<br/>12 entity types"]
    T --> V["Viterbi:<br/>valid BIO sequences only"]
    V --> TR["Trim span edges<br/>(learned table)"]
    TR --> S["Entity spans:<br/>start, end, type"]
```

## Setup shared by these notebooks

- **Task framing:** token classification with BIO tags over 12 entity types. The Phase-1 backbone reads sliding windows of 2,048 tokens (stride 1,024); overlapping windows are merged by keeping, for each token, the window in which it has most context.
- **Output side (from batch 1):** a Viterbi decoder that allows only valid BIO sequences, and a learned table that trims span edges (one entry per label and edge character).
- **Head:** a linear layer, or from batch 2 a 1-layer BiLSTM (256 per direction) on top of the backbone.
- **Backbone:** the Phase-1 MoE (continuation 2, "hc") unless a notebook says otherwise.
- **Training:** AdamW (betas 0.9/0.98, weight decay 0.01), head learning rate 1e-3, backbone 3e-4 for adapter-type methods and 3e-5 for full fine-tuning, 6% warm-up then linear decay, gradient clip 1.0, fp16. Class weights (inverse square root of label frequency) from `p2-06` on ("clsw" in a name).
- **Protocol:** train on fold 1, choose the epoch on fold 2, refit on both folds, predict the test fold. Hyperparameters are our placeholders and were not tuned.

## Notebooks in this folder

| Notebook | What it does |
|---|---|
| `sweep-p2s-moe3.ipynb` | The batch-2 winners stacked, plus a 256-token document header, a 512-token window and oversampling |
| `sweep-p2s-moe5.ipynb` | The stacked recipe with the other techniques: full fine-tuning, LoRA, Pfeiffer, prompt tuning |
| *(`p2s_moe2`; zips only, notebook not kept)* | Higher learning rate, 1,024-token window, rare-label oversampling, larger adapters |

Each notebook was run on Kaggle with "Run All" and is saved **with its outputs**, so the printed tables, curves and scores are the record of that run.

## Submissions produced

11 files in [`../../submissions/batch3-moe-one-factor/`](../../submissions/batch3-moe-one-factor/). See that folder's README for every file; the scores that the phase README lists are:

| File | What it is | Score |
|---|---|---|
| `p2s_moe3_stack_window512.zip` | best batch-3 single model | 0.5485 |
