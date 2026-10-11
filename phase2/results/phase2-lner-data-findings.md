# Phase 2 (L-NER) — data findings (measured 2026-10-07 on the real files)

Files (HF `Phase_2_LNER/`): Train/fold1-text.json, fold1-gold.json, fold2-text.json, fold2-gold.json, labels.txt; Test/fold3-text.json. Evaluator: `evaluate_phase2.py` (Drive "Evaluation Script" folder).

## Format
- text files: `{doc_id: "full judgment text"}` (value is the string itself). No newlines in the text; 115 U+00AD soft hyphens and ~1.5k non-ASCII chars in fold1 (curly quotes etc.).
- gold files: `{doc_id: [[start, end, LABEL, entity_text], ...]}`. **end is exclusive**: `text[start:end] == entity_text` for all 8,393 gold entities (0 mismatches).
- labels (12): APP RESP A.COUNSEL R.COUNSEL JUDGE WIT AUTH COURT STAT PREC DATE CASENO.
- **No overlapping or nested gold spans** → flat BIO/BIOES tagging can represent every gold entity.

## Size (small!)
- fold1: 35 docs, 4,120 entities; fold2: 35 docs, 4,273 entities. Test fold3: 35 docs. Total train = 70 docs.
- Doc length (words): fold1 min/median/max 1714/4511/17975; fold2 2817/4486/19978; fold3 0/4815/16402. Max 114k chars. At ~1.15 tokens/word ≈ 5.2k tokens median, ~23k max → several 2048-token windows per doc.
- **One fold3 doc (id 1305957) is EMPTY ("")** → must still appear in prediction.json with `[]`.

## Label counts (fold1 / fold2)
APP 207/200, RESP 191/190, A.COUNSEL 70/125, R.COUNSEL 87/97, JUDGE 123/122, WIT 67/120, AUTH 535/468, COURT 455/558, STAT 525/735, PREC 624/559, DATE 813/762, CASENO 423/337. Macro-F1 → rare labels (A.COUNSEL, WIT, R.COUNSEL) weigh as much as DATE.

## Boundary facts (matter for strict matching)
- 8078/8393 entities start at a whitespace boundary; only **5169/8393 end at one**. When the end is inside a whitespace-word, the next char is mostly `,` (1502) or `.` (1079), then `)` 150, alnum 169 (footnote digits glued: "State of Madhya Pradesh1", "JUDGMENT08/12/1994", "Railway Board|s"), `;` `'` `]` etc.
- So ~38% of entities end before attached punctuation. The Phase-1 BPE tokenizer splits on whitespace only (punctuation stays inside the word, `</w>` end marker, byte-level symbols), so whether a gold end lands on a token boundary depends on the learned merges — must be measured with the real tokenizer file.
- 16 fold2 gold spans include a trailing space (annotation noise, e.g. "Constitution "); unreachable if we trim; negligible.
- Annotation quirks: "Criminal Appeal No" (CASENO stops before "."), section numbers not labelled (only Act names as STAT).

## Evaluator (`evaluate_phase2.py`) rules
- pred IDs must equal truth IDs exactly; each entity `[start,end,label]` or with 4th string; ints, 0 <= start < end; label in the 12; **duplicate (start,end,label) in one doc → evaluation fails**. Per-label TP/FP/FN on exact (start,end,label) triples, macro over 12 labels.
