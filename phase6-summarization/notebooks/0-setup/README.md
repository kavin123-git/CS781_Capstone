# Phase 6 · `0-setup`

The data inspection (CPU): document and summary lengths, compression, and how extractive the summaries are (about 37% of summary 3-grams are new).

## Architecture

The pipeline all notebooks in this phase build on:

```mermaid
flowchart LR
    D["Judgment<br/>1,274-token budget"] --> P["+ prompt<br/>SUMMARY OF THE JUDGMENT:"]
    P --> M["Decoder-only model<br/>Phase-1 backbone, adapted<br/>context 2,048"]
    M --> G["Generate up to 768 tokens<br/>n-gram blocking"]
    G --> S["Summary"]
```

## Setup shared by these notebooks

- **Framing:** decoder-only continuation. The sequence is the selected document (budget 1,274 tokens, head and tail 70:30), the prompt `SUMMARY OF THE JUDGMENT:`, then the summary (at most 768 tokens). The context is 2,048 tokens and the loss is computed on the summary only. The GDN encoder–decoder instead encodes up to 1,024 document tokens and decodes up to 768.
- **Training (batch 1):** 3 epochs over 6,728 training documents, batch 16 via accumulation; learning rate 3e-4 for LoRA-type methods, 5e-4 for adapters, 3e-5 for full fine-tuning.
- **Validation:** 300 training documents are held out and never trained on (fixed seed); generation is scored on 100 of them with the course's evaluation code (BERTScore F1 with roberta-large).
- **Generation:** greedy, with an n-gram repeat block; batch 2 and 3 vary it. Hyperparameters are our placeholders and were not tuned.

## Notebooks in this folder

| Notebook | What it does |
|---|---|
| `p6-data-inspect.ipynb` | Document and summary lengths, compression, how extractive the summaries are (CPU) |

Each notebook was run on Kaggle with "Run All" and is saved **with its outputs**, so the printed tables, curves and scores are the record of that run.

## Submissions produced

None: these notebooks produce analysis only.
