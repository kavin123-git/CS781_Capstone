# Phase 2 · `batch1-moe-techniques`

**Question.** With the output side fixed (Viterbi + edge trimming), which adaptation technique works best on the MoE? **Finding.** They finished close together, so the technique was not the bottleneck. Prefix tuning failed (0.12), removing the causal mask hurt, and splitting punctuation fixed span edges but raised the Phase-1 loss. The baseline `p2-01` scored 0.4343 on Codabench.

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
| `p2-01-moe-hc-full-ft.ipynb` | Baseline: BIO tags, linear head, full fine-tuning |
| `p2-02-moe-hc-lorma.ipynb` | LoRMA (the compulsory technique) |
| `p2-03-moe-hc-lora.ipynb` | LoRA |
| `p2-04-moe-hc-full-ft-bidir.ipynb` | Full fine-tuning with the causal mask removed (bidirectional attention). This hurt. |
| `p2-05-moe-hc-full-ft-punct.ipynb` | Punctuation split before tokenizing. It fixes span edges but raises the Phase-1 loss. |
| `p2-06-moe-hc-full-ft-clsw.ipynb` | Full fine-tuning + class weights |
| `p2-07-moe-hc-lorma-clsw.ipynb` | LoRMA + class weights |
| `p2-08-moe-hc-lora-clsw.ipynb` | LoRA + class weights |
| `p2-09-moe-hc-adapter-clsw.ipynb` | Houlsby adapters + class weights |
| `p2-10-moe-hc-bitfit-clsw.ipynb` | BitFit (biases only) + class weights |
| `p2-11-moe-hc-prefix-clsw.ipynb` | Prefix tuning. Failed (0.12). |

Each notebook was run on Kaggle with "Run All" and is saved **with its outputs**, so the printed tables, curves and scores are the record of that run.

## Submissions produced

11 files in [`../../submissions/batch1-moe-techniques/`](../../submissions/batch1-moe-techniques/). See that folder's README for every file; the scores that the phase README lists are:

| File | What it is | Score |
|---|---|---|
| `p2_01_moe_hc_full_ft.zip` | baseline | 0.4343 |
| `p2_11_moe_hc_prefix_clsw.zip` | prefix tuning (failed) | 0.1220 |
