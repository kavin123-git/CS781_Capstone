"""Generates the Phase-2 L-NER experiment notebooks (one per experiment) from shared code."""
import nbformat as nbf, json, os

P = "/home/claude/p2/"
OUT = P + "notebooks/"
os.makedirs(OUT, exist_ok=True)
BPE_SRC = open(P + "bpe_class.py").read().replace("import json\n", "", 1)
ALIGN_SRC = open(P + "lner_align.py").read()
CORE_SRC = open(P + "p2_core.py").read()
EVAL_SRC = open(P + "evaluate_phase2.py").read()

# ------------------------------------------------------------------------------------------------------------------
# Model families: Phase-1 model code (verbatim from the user's notebooks) + how to build/load it + backbone wrapper
# ------------------------------------------------------------------------------------------------------------------
FAMILIES = {
    "moe_hc": dict(
        title="MoE (Muon + final LayerNorm), continuation 2 on high-court data — Phase-1 Codabench 0.8376",
        checkpoint="phase1_decoder_only_gqa_kvcache_moe_muon_finalnorm_continued2_highcourt_fp16_full_checkpoint.pt",
        source_nb="moe-muonfinallyer-mbr-highsuperemecourt (Phase-1 notebook, cells 'Model code')",
        pre="NORM_TYPE = \"layernorm\"; NORM_POSITION = \"pre\"   # as in the Phase-1 run\nCONTEXT_LEN = 2048\n",
        model_src=open(P + "model_moe_src.py").read(),
        backbone_src=open(P + "p2_backbone_decoder.py").read(),
        build_src='''def build_phase1_lm():
    """The Phase-1 MoE exactly as trained (218.7M params), with its trained weights."""
    torch.manual_seed(0)
    lm = CachedDecoderModel(VOCAB, d_model=768, n_heads=12, n_layers=8, d_ff=3072, context_len=CONTEXT_LEN)
    lm.fc_out = nn.Sequential(make_norm(768, NORM_TYPE), lm.fc_out)          # the final LayerNorm (FINAL_NORM = True in Phase 1)
    lm.load_state_dict(PHASE1_STATE, strict=True)
    for m in lm.modules():
        if isinstance(m, MoEFFN):
            m.bias_rate = MOE_BIAS_RATE_FT
    return lm


def make_backbone(lm, C=None):
    return DecoderBackbone(lm, bidirectional=((C or {}).get("ATTENTION", ATTENTION) == "bidirectional"))


def phase1_sanity(lm, docs):
    return lm_sanity_loss(lm, docs, device)
''',
        extra_settings='''MOE_BIAS_RATE_FT = 0.0       # MoE routing-bias update rate during fine-tuning (Phase 1 used 0.001). 0.0 = keep the pretrained routing fixed.
''',
        default_targets='["attn.q_proj", "attn.k_proj", "attn.v_proj", "attn.out_proj"]',
    ),
    "gqa160": dict(
        title="dense GQA + RoPE decoder, 160M words, 1 epoch (111.1M params) — Phase-1 Codabench 0.8254",
        checkpoint="phase1_decoder_only_gqa_kvcache_full_layernorm_pre_adamw_rope_ctx2048_160Mw_a1bpe_160Mw_1ep_checkpoint.pt",
        source_nb="gqaRoPe160M1 (Phase-1 notebook, cells 'Building blocks' + 'The model')",
        pre="NORM_TYPE = \"layernorm\"; NORM_POSITION = \"pre\"   # as in the Phase-1 run\nCONTEXT_LEN = 2048\n",
        model_src=open(P + "model_gqa_src.py").read(),
        backbone_src=open(P + "p2_backbone_decoder.py").read(),
        build_src='''def build_phase1_lm():
    """The Phase-1 dense GQA model exactly as trained (111.1M params, no final LayerNorm), with its trained weights."""
    torch.manual_seed(0)
    lm = CachedDecoderModel(VOCAB, d_model=768, n_heads=12, n_layers=8, d_ff=3072, context_len=CONTEXT_LEN)
    lm.load_state_dict(PHASE1_STATE, strict=True)
    return lm


def make_backbone(lm, C=None):
    return DecoderBackbone(lm, bidirectional=((C or {}).get("ATTENTION", ATTENTION) == "bidirectional"))


def phase1_sanity(lm, docs):
    return lm_sanity_loss(lm, docs, device)
''',
        extra_settings="",
        default_targets='["attn.q_proj", "attn.k_proj", "attn.v_proj", "attn.out_proj"]',
    ),
    "mor": dict(
        title="LegalMoR: Mixture-of-Recursions decoder (GQA + RoPE, shared 3-layer block x 3 passes, causal threshold routing), 160M words — Phase-1 Codabench 0.7954",
        checkpoint="phase1_decoder_only_gqa_kvcache_full_layernorm_pre_adamw_rope_ctx2048_160Mw_a1bpe_mor160Mw_1ep_checkpoint.pt",
        source_nb="mor-gqa-rope (Phase-1 notebook, cells 'Building blocks' + 'The model')",
        pre="NORM_TYPE = \"layernorm\"; NORM_POSITION = \"pre\"   # as in the Phase-1 run\nCONTEXT_LEN = 2048\n",
        model_src=open(P + "model_mor_src.py").read(),
        backbone_src=open(P + "p2_backbone_generic.py").read(),
        build_src='''MOR_FREEZE_ROUTING = True    # keep the Phase-1 routing thresholds (tau) FIXED while fine-tuning. In Phase 1 they are updated inside the
                             # forward pass in training mode (a running quantile, not learned); on NER they drift with the new data, so
                             # tokens change depth between steps, and one fp16 overflow turns them into NaN for good (seen on Kaggle).
_MOR_ORIG_ROUTE = CachedDecoderModel._route


def _mor_route_frozen(self, r, h, eligible):
    """Same decision rule as Phase-1 _route (own score vs threshold -> selected, sigmoid gate), but never updates the threshold."""
    z = self.router[r - 1](self.router_norm[r - 1](h)).squeeze(-1).float()
    tau = self.tau[r - 1].clone()
    return eligible & (z > tau), torch.sigmoid(z - tau)


CachedDecoderModel._route = _mor_route_frozen if MOR_FREEZE_ROUTING else _MOR_ORIG_ROUTE


def build_phase1_lm():
    """The Phase-1 MoR model exactly as trained, with its trained weights (incl. the routing thresholds)."""
    torch.manual_seed(0)
    lm = CachedDecoderModel(VOCAB, d_model=768, n_heads=12, n_layers=8, d_ff=3072, context_len=CONTEXT_LEN)
    lm.load_state_dict(PHASE1_STATE, strict=True)
    return lm


def make_backbone(lm, C=None):
    return ForwardBackbone(lm)


def phase1_sanity(lm, docs):
    return lm_sanity_loss(lm, docs, device)
''',
        extra_settings="",
        default_targets='["attn.q_proj", "attn.k_proj", "attn.v_proj", "attn.out_proj"]',
    ),
    "canon": dict(
        title="Boundary Canon decoder (GQA + RoPE + word/sentence-boundary Canon layers), 240M words — Phase-1 Codabench 0.8054",
        checkpoint="phase1_decoder_only_gqa_kvcache_bcanon_b128_t4w3s2_AC_full_layernorm_pre_adamw_rope_ctx2048_240Mw_a1bpe_240Mw_1ep_bcanon_checkpoint.pt",
        source_nb="canon-layersv2 (Phase-1 notebook: Canon settings, 'Building blocks', Boundary Canon module, 'The model')",
        pre="NORM_TYPE = \"layernorm\"; NORM_POSITION = \"pre\"   # as in the Phase-1 run\nCONTEXT_LEN = 2048\n",
        model_src=open(P + "model_canon_src.py").read(),
        backbone_src=open(P + "p2_backbone_generic.py").read(),
        build_src='''WORD_END_TABLE, SENT_END_TABLE = build_boundary_tables(bpe_tokenizer.itos)      # derived from the tokenizer, as in Phase 1


def build_phase1_lm():
    """The Phase-1 Boundary-Canon model exactly as trained, with its trained weights."""
    torch.manual_seed(0)
    lm = CachedDecoderModel(VOCAB, d_model=768, n_heads=12, n_layers=8, d_ff=3072, context_len=CONTEXT_LEN,
                            word_end_table=WORD_END_TABLE, sent_end_table=SENT_END_TABLE)
    lm.load_state_dict(PHASE1_STATE, strict=True)
    return lm


def make_backbone(lm, C=None):
    return ForwardBackbone(lm)


def phase1_sanity(lm, docs):
    return lm_sanity_loss(lm, docs, device)
''',
        extra_settings="",
        default_targets='["attn.q_proj", "attn.k_proj", "attn.v_proj", "attn.out_proj"]',
    ),
    "lstm": dict(
        title="lightweight LSTM language model (3 layers, 768 emb, 1536 hidden, tied embeddings, 84M params) — Phase-1 Codabench 0.8237",
        checkpoint="phase1_lightweight_lstm84M_full_checkpoint.pt",
        source_nb="capston-lstm-lightmodel (Phase-1 notebook, model cell)",
        pre="CONTEXT_LEN = 2048   # (the LSTM itself has no length limit)\n",
        model_src=open(P + "model_lstm_src.py").read(),
        backbone_src=open(P + "p2_backbone_generic.py").read(),
        build_src='''def build_phase1_lm():
    """The Phase-1 LSTM exactly as trained (EMBED_DIM 768, LSTM_HIDDEN 1536, 3 layers), with its trained weights."""
    torch.manual_seed(0)
    lm = LSTMLM(VOCAB, 768, 1536, 3)
    lm.load_state_dict(PHASE1_STATE, strict=True)
    return lm


def make_backbone(lm, C=None):
    return LSTMBackbone(lm)


def phase1_sanity(lm, docs):
    return lm_sanity_loss(lm, docs, device)
''',
        extra_settings="",
        default_targets="[]   # the LSTM has no linear layers inside its recurrence: LoRA / LoRMA do not apply",
    ),
    "gdn": dict(
        title="GDN encoder-decoder (6 bidirectional Gated-DeltaNet encoder blocks + 4 GQA decoder blocks, 144.3M) — the ENCODER is used — Phase-1 Codabench 0.8010",
        checkpoint="phase1_decoder_only_gqa_kvcache_full_layernorm_pre_adamw_rope_ctx2048_160Mw_a1bpe_gdnencdec160Mw_1ep_checkpoint.pt",
        source_nb="gdn_encoder_decoder (Phase-1 notebook, cells 'Building blocks' + 'The model')",
        pre="NORM_TYPE = \"layernorm\"; NORM_POSITION = \"pre\"   # as in the Phase-1 run\nCONTEXT_LEN = 2048\nENC_MAX = 1024; ENC_MIN = 384; DEC_LEN = 384   # Phase-1 shapes (the encoder read up to 1024-token prompts)\n",
        model_src=open(P + "model_gdn_src.py").read(),
        backbone_src=open(P + "p2_backbone_generic.py").read(),
        build_src='''def build_phase1_lm():
    """The Phase-1 GDN encoder-decoder exactly as trained, with its trained weights (the decoder is dropped later)."""
    torch.manual_seed(0)
    lm = EncoderDecoderLM(VOCAB, d_model=768, n_heads=12, enc_layout=ENC_LAYOUT, n_dec_layers=N_DEC_LAYERS,
                          d_ff=3072, context_len=CONTEXT_LEN)
    lm.load_state_dict(PHASE1_STATE, strict=True)
    return lm


def make_backbone(lm, C=None):
    return GDNEncoderBackbone(lm)


def phase1_sanity(lm, docs):
    return gdn_sanity_loss(lm, docs, device)        # decoder next-token loss with the encoder reading the preceding 768 tokens
''',
        extra_settings="",
        default_targets='[".q_proj", ".k_proj", ".v_proj", ".o_proj"]   # the Gated-DeltaNet mixers (forward and backward)',
    ),
}

# ------------------------------------------------------------------------------------------------------------------
# Experiments (one notebook each). Hyperparameter values are PLACEHOLDERS for the user to set.
# ------------------------------------------------------------------------------------------------------------------
EXPERIMENTS = [
    dict(file="p2_01_moe_hc_full_ft", family="moe_hc", technique="full_ft",
         what="**Full fine-tuning**: every weight of the Phase-1 MoE is updated, plus a new token-classification head. The reference point every PEFT method is compared against.",
         lr="3e-5"),
    dict(file="p2_02_moe_hc_lorma", family="moe_hc", technique="lorma",
         what="**LoRMA (compulsory)**: the frozen weight W0 of each targeted projection is MULTIPLIED by a learned near-identity matrix: `W = ((alpha/r)·B·A + I)·W0` (LoRMA+, default) or `W = I_pi(B·A)·W0` (LoRMA_pi). Only A, B and the head are trained.",
         lr="3e-4"),
    dict(file="p2_04_moe_hc_full_ft_bidir", family="moe_hc", technique="full_ft", set={"ATTENTION": '"bidirectional"'},
         what="**Full fine-tuning with BIDIRECTIONAL attention**: identical to p2_01 except the causal mask is dropped during fine-tuning, so every token also sees the words to its RIGHT (e.g. '..Appellant', 'learned counsel for the respondent', '(PW-1)' after a name). Tests the context factor.",
         lr="3e-5"),
    dict(file="p2_05_moe_hc_full_ft_punct", family="moe_hc", technique="full_ft", set={"INPUT_MODE": '"punct"'},
         what="**Full fine-tuning with punctuation split before BPE**: identical to p2_01 except `INPUT_MODE = \"punct\"`: each word is split into letter/digit runs and single punctuation marks before your BPE, so `Haryana,` becomes `Haryana` + `,` and almost every entity edge falls on a token boundary. Tests the preprocessing / alignment factor.",
         lr="3e-5"),
    dict(file="p2_06_moe_hc_full_ft_clsw", family="moe_hc", technique="full_ft", set={"CLASS_WEIGHTING": '"inv_sqrt"'},
         what="**Full fine-tuning with class-weighted loss**: identical to p2_01 except each tag's loss weight is sqrt(total / count), so rare tags (WIT, A.COUNSEL, R.COUNSEL, JUDGE) count more. Tests the class-imbalance factor.",
         lr="3e-5"),
    dict(file="p2_03_moe_hc_lora", family="moe_hc", technique="lora",
         what="**LoRA**: the frozen W0 gets an ADDED low-rank update: `W = W0 + (alpha/r)·B·A`. Same targets/rank as the LoRMA notebook, so the two are directly comparable (additive vs multiplicative).",
         lr="3e-4"),
    dict(file="p2_07_moe_hc_lorma_clsw", family="moe_hc", technique="lorma", set={"CLASS_WEIGHTING": '"inv_sqrt"', "EPOCHS": "20", "PATIENCE": "5"},
         what="**LoRMA (compulsory) + class weights, up to 20 epochs**: LoRMA+ as in p2_02, with the class-weighted loss that helped full FT (p2_06) and twice the epoch budget (p2_02 was still improving at epoch 10).",
         lr="3e-4"),
    dict(file="p2_08_moe_hc_lora_clsw", family="moe_hc", technique="lora", set={"CLASS_WEIGHTING": '"inv_sqrt"', "EPOCHS": "20", "PATIENCE": "5"},
         what="**LoRA + class weights, up to 20 epochs**: identical settings to p2_07, so LoRA (additive) vs LoRMA (multiplicative) stays a fair comparison.",
         lr="3e-4"),
    dict(file="p2_09_moe_hc_adapter_clsw", family="moe_hc", technique="adapter", set={"CLASS_WEIGHTING": '"inv_sqrt"', "EPOCHS": "20", "PATIENCE": "5"},
         what="**Houlsby adapters + class weights**: a small bottleneck MLP (768→64→768, zero-initialised output) is inserted after the attention AND after the feed-forward of every block; the Phase-1 weights stay frozen. (`ADAPTER_TYPE = \"pfeiffer\"` = after the feed-forward only.)",
         lr="3e-4"),
    dict(file="p2_10_moe_hc_bitfit_clsw", family="moe_hc", technique="bitfit", set={"CLASS_WEIGHTING": '"inv_sqrt"', "EPOCHS": "20", "PATIENCE": "5"},
         what="**BitFit + class weights**: only the BIAS vectors of the Phase-1 model (plus the head) are trained; every weight matrix stays frozen. The sparsest method in the menu.",
         lr="1e-3"),
    dict(file="p2_11_moe_hc_prefix_clsw", family="moe_hc", technique="prefix", set={"CLASS_WEIGHTING": '"inv_sqrt"', "EPOCHS": "20", "PATIENCE": "5"},
         what="**Prefix tuning + class weights**: every attention layer gets PREFIX_LEN learned key/value vectors that all tokens can attend to; the Phase-1 weights stay frozen.",
         lr="1e-3"),
    dict(file="p2_12_moe_hc_full_ft_bilstm_clsw", family="moe_hc", technique="full_ft", set={"CLASS_WEIGHTING": '"inv_sqrt"', "HEAD_TYPE": '"bilstm"'},
         what="**Full fine-tuning + BiLSTM prediction head + class weights**: like p2_06, but the head is a bidirectional LSTM (256 per direction) over the model's hidden states, so every token gets RIGHT context (e.g. '..Appellant' after a name) without changing the causally pretrained layers (p2_04 showed that removing the causal mask hurts). Tests the prediction-head factor.",
         lr="3e-5"),
    dict(file="p2_13_moe_hc_adapter_bilstm_clsw", family="moe_hc", technique="adapter", set={"CLASS_WEIGHTING": '"inv_sqrt"', "HEAD_TYPE": '"bilstm"', "EPOCHS": "20", "PATIENCE": "5"},
         what="**Houlsby adapters + BiLSTM prediction head + class weights**: combines the best PEFT method (p2_09, 0.481 Codabench) with the best head (p2_12, 0.522). The Phase-1 weights stay frozen; only the adapters, the BiLSTM and the output layer are trained.",
         lr="3e-4"),
    dict(file="p2_14_moe_hc_lorma_bilstm_clsw", family="moe_hc", technique="lorma", set={"CLASS_WEIGHTING": '"inv_sqrt"', "HEAD_TYPE": '"bilstm"', "EPOCHS": "20", "PATIENCE": "5"},
         what="**LoRMA (compulsory) + BiLSTM prediction head + class weights**: LoRMA+ as in p2_07 (0.439 Codabench) with the BiLSTM head that helped full fine-tuning most (p2_12). Only LoRMA's A/B matrices, the BiLSTM and the output layer are trained.",
         lr="3e-4"),
]


def settings_cell(exp, fam):
    import re as _re
    cell = _settings_cell(exp, fam)
    for k, v in exp.get("set", {}).items():
        cell, n = _re.subn(rf"^{k} = [^#\n]*?(\s*#|$)", lambda m: f"{k} = {v}" + ("   " + m.group(1).strip() if m.group(1).strip() else ""), cell, count=1, flags=_re.M)
        assert n == 1, k
    return cell


def _settings_cell(exp, fam):
    t = exp["technique"]
    return f'''# =====================================================================================================
# SETTINGS -- every value below is YOUR choice (the numbers are placeholders, not tuned)
# =====================================================================================================
RUN_NAME = "{exp['file']}"            # names the results / weights / submission files
MODEL_KEY = "{exp['family']}"
PHASE1_CHECKPOINT = "{fam['checkpoint']}"
TOKENIZER_NAME = "a1bpe_tokenizer_160Mw_1ep_full_v40000_160000000w_seed42_minf1.json"

# ---- technique ---------------------------------------------------------------------------------------
TECHNIQUE = "{t}"         # full_ft | partial_ft | bitfit | lora | lorma | adapter | prompt | prefix
RANK = 16                     # lora / lorma rank r
ALPHA = 32                    # lora / lorma scale alpha (update is scaled by alpha / r; LoRMA_pi uses no scale)
LORMA_VARIANT = "plus"        # "plus" = LoRMA+ (additive rank inflation) | "pi" = LoRMA_pi (permutation inflation)
LORA_DROPOUT = 0.0
PEFT_TARGETS = {fam['default_targets']}   # linear layers adapted by lora / lorma (name suffixes)
PARTIAL_TOP_BLOCKS = 2        # partial_ft: how many top blocks are trained
ADAPTER_TYPE = "houlsby"      # adapter: "houlsby" (after attention and FFN) | "pfeiffer" (after FFN only)
ADAPTER_DIM = 64              # adapter bottleneck width
PROMPT_LEN = 20               # prompt tuning: soft tokens prepended to every window
PREFIX_LEN = 20               # prefix tuning: learned key/value vectors per layer
{fam['extra_settings']}
# ---- task modelling ----------------------------------------------------------------------------------
INPUT_MODE = "none"           # "none" = exact Phase-1 tokens | "punct" = split punctuation before BPE (see the step-0 notebook)
ATTENTION = "causal"          # "causal" = as pretrained (each token sees only its left) | "bidirectional" = drop the causal mask
HEAD_TYPE = "linear"          # "linear" | "mlp" | "bilstm" (bidirectional LSTM over the backbone states, then linear)
HEAD_LSTM_HIDDEN = 256        # bilstm head: hidden size per direction
HEAD_DROPOUT = 0.1
HEAD_NORM = False             # extra LayerNorm in the head
CLASS_WEIGHTING = "none"      # "none" | "inv_sqrt" | "inv" (loss weights from tag frequencies, against class imbalance)
DECODE = "viterbi"            # "viterbi" = best BIO-valid tag sequence | "argmax" = per-token argmax with lenient repair
MERGE = "context"             # overlapping windows: "context" = keep the window giving the token most context | "average"
BOUNDARY_REPAIR = True        # trim punctuation at span edges with the table learned on the training fold(s)

# ---- training -----------------------------------------------------------------------------------------
WINDOW = 2048                 # tokens per window (<= 2048; with prompt tuning <= 2048 - PROMPT_LEN)
TRAIN_STRIDE = 1024           # training windows overlap by WINDOW - TRAIN_STRIDE
INFER_STRIDE = 1024
BATCH = 2                     # windows per forward pass (lower to 1 if CUDA runs out of memory)
ACCUM = 4                     # gradient accumulation -> effective batch = BATCH * ACCUM windows
EPOCHS = 10                   # maximum; the best epoch on the held-out fold is kept
PATIENCE = 3                  # stop after this many epochs without held-out improvement (0 = never)
LR = {exp['lr']}                     # learning rate of every trainable parameter outside the head (backbone / LoRA / LoRMA / adapters / biases / prompt / prefix)
HEAD_LR = 1e-3                # learning rate of the new head (incl. the BiLSTM head)
WEIGHT_DECAY = 0.01
WARMUP_FRAC = 0.06
GRAD_CLIP = 1.0
AMP = "fp16"                  # "fp16" mixed precision on GPU | "none"
SEED = 0
INFER_BATCH = 4
KD_ALPHA = 0.0; KD_TEMPERATURE = 2.0     # (used only by the distillation notebook)

# ---- evaluation protocol ---------------------------------------------------------------------------------
CV_TRAIN, CV_EVAL = "fold1", "fold2"     # train on one fold, pick the epoch and report scores on the other
REFIT_ON_BOTH = True          # then retrain on fold1+fold2 for the best epoch count and predict the test set with that model
                              # (False = predict the test set with the fold1 model; saves ~half the time)
QUICK_TEST = False            # True = tiny run (2 docs per fold, 1 epoch, 512-token windows) to check the notebook end to end

C_KEYS = ["TECHNIQUE", "RANK", "ALPHA", "LORMA_VARIANT", "LORA_DROPOUT", "PEFT_TARGETS", "PARTIAL_TOP_BLOCKS", "ADAPTER_TYPE",
          "ADAPTER_DIM", "PROMPT_LEN", "PREFIX_LEN", "INPUT_MODE", "ATTENTION", "HEAD_TYPE", "HEAD_LSTM_HIDDEN", "HEAD_DROPOUT", "HEAD_NORM",
          "CLASS_WEIGHTING", "DECODE", "MERGE", "BOUNDARY_REPAIR", "WINDOW", "TRAIN_STRIDE", "INFER_STRIDE", "BATCH", "ACCUM",
          "EPOCHS", "PATIENCE", "LR", "HEAD_LR", "WEIGHT_DECAY", "WARMUP_FRAC", "GRAD_CLIP", "AMP", "SEED", "INFER_BATCH",
          "KD_ALPHA", "KD_TEMPERATURE", "CV_TRAIN", "CV_EVAL", "REFIT_ON_BOTH"]
if QUICK_TEST:
    EPOCHS, WINDOW, TRAIN_STRIDE, INFER_STRIDE, PATIENCE = 1, 512, 512, 256, 0
    RUN_NAME += "_quicktest"
C = {{k: globals()[k] for k in C_KEYS}}
print(json.dumps(C, indent=None, default=str))
'''


SETUP_CELL = '''!pip install -q -U "huggingface_hub>=1.5"
import os, json, shutil, subprocess, sys, time, zipfile, random, math, gc
from collections import Counter, defaultdict
import torch, torch.nn as nn, torch.nn.functional as F
from huggingface_hub import hf_hub_download, login, download_bucket_files, batch_bucket_files
from kaggle_secrets import UserSecretsClient

NB_START = time.time()
WORK = "/kaggle/working"
HF_DATASET_REPO = "Exploration-Lab/CS-781-Capstone"
HF_BUCKET = "kavincg23/cs781-models"
HF_CHECKPOINT_REPO = "kavincg23/cs781-checkpoints"   # old repo, read-only fallback
user_secrets = UserSecretsClient()
HF_TOKEN = user_secrets.get_secret("HF")
HF_WRITE_TOKEN = user_secrets.get_secret("HF_WRITE")
os.environ["HF_TOKEN"] = HF_TOKEN
login(token=HF_TOKEN)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", device, torch.cuda.get_device_name(0) if device.type == "cuda" else "")


def get_dataset_file(filename):
    return hf_hub_download(repo_id=HF_DATASET_REPO, filename=filename, repo_type="dataset", token=HF_TOKEN)


def fetch_model_file(name, local_path=None):
    local_path = local_path or f"{WORK}/{name}"
    if os.path.exists(local_path):
        return local_path
    part = local_path + ".part"
    try:
        download_bucket_files(HF_BUCKET, files=[(name, part)], raise_on_missing_files=True, token=HF_WRITE_TOKEN)
        os.replace(part, local_path)
        return local_path
    except Exception:
        if os.path.exists(part):
            os.remove(part)
    # fallback: the OLD model repo (the Phase-1 notebooks do the same; the tokenizer only lives there)
    p = hf_hub_download(repo_id=HF_CHECKPOINT_REPO, filename=name, repo_type="model", token=HF_WRITE_TOKEN)
    shutil.copy(p, local_path)
    print(f"(found {name} in the old repo {HF_CHECKPOINT_REPO})")
    return local_path


def upload_model_file(local_path, name=None):
    name = name or os.path.basename(local_path)
    try:
        batch_bucket_files(HF_BUCKET, add=[(local_path, name)], token=HF_WRITE_TOKEN)
        print(f"uploaded {name} to bucket {HF_BUCKET}")
    except Exception as e:
        print(f"(upload of {name} failed: {type(e).__name__}: {e}; the file is still in {WORK})")
'''

DATA_CELL = '''P2 = "Phase_2_LNER"
def load_json(p):
    with open(p, encoding="utf-8") as f:
        return json.load(f)

LABELS = [l.strip() for l in open(get_dataset_file(f"{P2}/Train/labels.txt"), encoding="utf-8") if l.strip()]
TEXTS = {f: load_json(get_dataset_file(f"{P2}/Train/{f}-text.json")) for f in ("fold1", "fold2")}
GOLDS = {f: load_json(get_dataset_file(f"{P2}/Train/{f}-gold.json")) for f in ("fold1", "fold2")}
TEST_TEXTS = load_json(get_dataset_file(f"{P2}/Test/fold3-text.json"))
for f in TEXTS:
    assert set(TEXTS[f]) == set(GOLDS[f])
    assert all(TEXTS[f][d][e[0]:e[1]] == e[3] for d in GOLDS[f] for e in GOLDS[f][d]), "gold offsets do not match the text"
if QUICK_TEST:
    for f in TEXTS:
        keep = list(TEXTS[f])[:2]
        TEXTS[f] = {k: TEXTS[f][k] for k in keep}; GOLDS[f] = {k: GOLDS[f][k] for k in keep}
    TEST_TEXTS = {k: TEST_TEXTS[k] for k in list(TEST_TEXTS)[:3]} | {k: v for k, v in TEST_TEXTS.items() if not v.strip()}
print("labels:", LABELS)
print({f: (len(TEXTS[f]), sum(len(v) for v in GOLDS[f].values())) for f in TEXTS}, "| test docs:", len(TEST_TEXTS))
'''

TOK_LOAD = '''
bpe_tokenizer = BPETokenizer.load(fetch_model_file(TOKENIZER_NAME))
VOCAB = bpe_tokenizer.get_vocab_size()
print("tokenizer:", TOKENIZER_NAME, "| vocab", VOCAB)
'''

PREP_CELL = '''t0 = time.time()
TAG_NAMES = bio_tag_names(LABELS)
DOCS = {f: prepare_docs(bpe_tokenizer, TEXTS[f], GOLDS[f], LABELS, INPUT_MODE) for f in TEXTS}
TEST_DOCS = prepare_docs(bpe_tokenizer, TEST_TEXTS, None, LABELS, INPUT_MODE)
if INPUT_MODE == "none":
    check_encode_identical(bpe_tokenizer, {**TEXTS["fold1"], **TEXTS["fold2"], **TEST_TEXTS})
    print("token ids identical to BPETokenizer.encode on every document")
GOLD3 = {f: {d: [e[:3] for e in GOLDS[f][d]] for d in GOLDS[f]} for f in GOLDS}


def trim_tables(folds):
    tx, gd, sp = {}, {}, {}
    for f in folds:
        tx.update(TEXTS[f]); gd.update(GOLDS[f]); sp.update({d: DOCS[f][d]["spans"] for d in DOCS[f]})
    return learn_trim_table(tx, gd, sp, LABELS)


TRIM_CV = trim_tables([CV_TRAIN])                 # learned on the training fold only
TRIM_ALL = trim_tables(["fold1", "fold2"])        # for the test set
mis = sum(len(gold_to_token_tags(DOCS[f][d]["spans"], GOLDS[f][d], LABELS)[1]) for f in DOCS for d in DOCS[f])
print(f"{len(TAG_NAMES)} tags; {mis} gold entities not on token boundaries (INPUT_MODE={INPUT_MODE!r}); "
      f"windows/epoch on {CV_TRAIN}: {len(window_items(DOCS[CV_TRAIN], WINDOW, TRAIN_STRIDE))}; {time.time()-t0:.0f}s")
'''

LOAD_CELL = '''t0 = time.time()
_ck = torch.load(fetch_model_file(PHASE1_CHECKPOINT), map_location="cpu", mmap=True)
PHASE1_STATE = _ck["model"] if "model" in _ck else _ck
del _ck
print(f"Phase-1 checkpoint loaded ({time.time()-t0:.0f}s): {PHASE1_CHECKPOINT}")

_lm = build_phase1_lm()
PHASE1_PARAMS = sum(p.numel() for p in _lm.parameters())
SANITY_LOSS = lm_sanity_loss(_lm, DOCS["fold1"], device)
print(f"Phase-1 model: {PHASE1_PARAMS:,} parameters; next-token loss on NER documents = {SANITY_LOSS:.3f} nats "
      f"(about 2.5-3.5 means the weights loaded correctly; ~{math.log(VOCAB):.1f} would mean random weights)")
assert SANITY_LOSS < 6.0, "the Phase-1 weights do not seem to be loaded correctly"
del _lm; gc.collect(); torch.cuda.empty_cache() if device.type == "cuda" else None


def build_ner():
    lm = build_phase1_lm()
    bb = make_backbone(lm)
    ner = NERModel(bb, bb.d_model, len(TAG_NAMES), HEAD_TYPE, HEAD_DROPOUT, HEAD_NORM, HEAD_LSTM_HIDDEN)
    info = apply_technique(ner, C)
    return ner.to(device), info
'''

CV_CELL = '''# ---- train on CV_TRAIN, choose the epoch on CV_EVAL ----
RESULTS = dict(run=RUN_NAME, model=MODEL_KEY, phase1_checkpoint=PHASE1_CHECKPOINT, phase1_params=PHASE1_PARAMS,
               sanity_lm_loss=round(SANITY_LOSS, 4), settings=C, gpu=torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu")
ner, RESULTS["technique_info"] = build_ner()
causal = ATTENTION != "bidirectional" and HEAD_TYPE != "bilstm"   # merge rule: a BiLSTM head also uses right context


def eval_fn(model):
    preds, _ = predict_docs(model, DOCS[CV_EVAL], LABELS, C, device, causal, *TRIM_CV)
    s = strict_scores(GOLD3[CV_EVAL], preds, LABELS)
    return s["macro_F1"], dict(heldout_P=round(s["macro_P"], 4), heldout_R=round(s["macro_R"], 4))


cv = train_ner(ner, DOCS[CV_TRAIN], C, device, eval_fn=eval_fn)
load_trainable_state(ner, cv["best_state"])
RESULTS["cv"] = {k: v for k, v in cv.items() if k != "best_state"}
torch.save(cv["best_state"], f"{WORK}/{RUN_NAME}_cvmodel_state.pt")
print(f"best epoch {cv['best_epoch']}: held-out strict macro-F1 {cv['best_score']:.4f} | train {cv['train_seconds']}s, "
      f"peak GPU memory {cv['peak_mem_gb']} GB, {cv['tokens_per_sec']} tokens/s")
'''

HELDOUT_CELL = '''# ---- detailed held-out evaluation of the selected model ----
preds_cv, infer_s = predict_docs(ner, DOCS[CV_EVAL], LABELS, C, device, causal, *TRIM_CV)
preds_norepair, _ = predict_docs(ner, DOCS[CV_EVAL], LABELS, dict(C, BOUNDARY_REPAIR=False), device, causal)
S = strict_scores(GOLD3[CV_EVAL], preds_cv, LABELS)
S0 = strict_scores(GOLD3[CV_EVAL], preds_norepair, LABELS)
n_tok = sum(len(d["ids"]) for d in DOCS[CV_EVAL].values())
RESULTS["heldout"] = dict(
    fold=CV_EVAL, macro_F1=round(S["macro_F1"], 4), macro_P=round(S["macro_P"], 4), macro_R=round(S["macro_R"], 4),
    macro_F1_without_boundary_repair=round(S0["macro_F1"], 4),
    per_label={l: {k: (round(v, 4) if isinstance(v, float) else v) for k, v in d.items()} for l, d in S["per_label"].items()},
    errors=error_breakdown(GOLD3[CV_EVAL], preds_cv), label_confusions=confusion_pairs(GOLD3[CV_EVAL], preds_cv),
    inference_seconds=round(infer_s, 1), inference_tokens_per_sec=round(n_tok / max(infer_s, 1e-9), 1))
print(f"held-out ({CV_EVAL}) strict macro-F1 {S['macro_F1']:.4f} (P {S['macro_P']:.4f}, R {S['macro_R']:.4f}); "
      f"without boundary repair {S0['macro_F1']:.4f}")
for l in LABELS:
    d = S["per_label"][l]
    print(f"  {l:10s} P={d['P']:.3f} R={d['R']:.3f} F1={d['F1']:.3f}  (tp {d['tp']}, fp {d['fp']}, fn {d['fn']})")
print("errors:", RESULTS["heldout"]["errors"])
print("most common label confusions:", dict(list(RESULTS["heldout"]["label_confusions"].items())[:8]))

# the official evaluator on the same predictions (must print the same macro-F1)
with open(f"{WORK}/evaluate_phase2.py", "w") as f:
    f.write(EVALUATOR_SRC)
json.dump(GOLDS[CV_EVAL], open(f"{WORK}/truth_{CV_EVAL}.json", "w"))
json.dump(preds_cv, open(f"{WORK}/{RUN_NAME}_heldout_{CV_EVAL}_pred.json", "w"))
open(f"{WORK}/labels.txt", "w").write("\\n".join(LABELS) + "\\n")
r = subprocess.run([sys.executable, f"{WORK}/evaluate_phase2.py", "--truth", f"{WORK}/truth_{CV_EVAL}.json",
                    "--prediction", f"{WORK}/{RUN_NAME}_heldout_{CV_EVAL}_pred.json", "--labels", f"{WORK}/labels.txt"],
                   capture_output=True, text=True)
print("official evaluator:", r.stdout.strip().splitlines()[-3] if r.returncode == 0 else r.stderr)
'''

FINAL_CELL = '''# ---- final model -> test predictions -> submission zip ----
if REFIT_ON_BOTH:
    del ner; gc.collect(); torch.cuda.empty_cache() if device.type == "cuda" else None
    ner, _ = build_ner()
    both = {**DOCS["fold1"], **DOCS["fold2"]}
    fit = train_ner(ner, both, dict(C, EPOCHS=cv["best_epoch"], PATIENCE=0), device, eval_fn=None)
    RESULTS["refit"] = {k: v for k, v in fit.items() if k != "best_state"}
    final_state = fit["best_state"]
else:
    final_state = cv["best_state"]
torch.save(final_state, f"{WORK}/{RUN_NAME}_final_state.pt")
test_pred, test_s = predict_docs(ner, TEST_DOCS, LABELS, C, device, causal, *TRIM_ALL)


def validate_prediction(pred, texts, labels):
    assert set(pred) == set(texts), f"id mismatch: missing {len(set(texts)-set(pred))}, extra {len(set(pred)-set(texts))}"
    for did, ents in pred.items():
        assert isinstance(ents, list)
        seen = set()
        for x in ents:
            s, e, l = x
            assert type(s) is int and type(e) is int and 0 <= s < e <= len(texts[did]) and l in labels, (did, x)
            assert (s, e, l) not in seen, ("duplicate", did, x); seen.add((s, e, l))


validate_prediction(test_pred, TEST_TEXTS, LABELS)
zip_path = f"{WORK}/submission_{RUN_NAME}.zip"
with open(f"{WORK}/prediction.json", "w", encoding="utf-8") as f:
    json.dump(test_pred, f)
with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
    z.write(f"{WORK}/prediction.json", arcname="prediction.json")        # at the zip ROOT
with zipfile.ZipFile(zip_path) as z:
    assert z.namelist() == ["prediction.json"]
RESULTS["test"] = dict(docs=len(test_pred), entities=sum(len(v) for v in test_pred.values()),
                       per_label=dict(Counter(e[2] for v in test_pred.values() for e in v)), inference_seconds=round(test_s, 1),
                       model="refit on fold1+fold2" if REFIT_ON_BOTH else f"trained on {CV_TRAIN}")
print(f"wrote {zip_path}: {RESULTS['test']['entities']} entities over {len(test_pred)} documents")
'''

SAVE_CELL = '''RESULTS["notebook_minutes"] = round((time.time() - NB_START) / 60, 1)
res_path = f"{WORK}/{RUN_NAME}_results.json"
json.dump(RESULTS, open(res_path, "w"), indent=1, default=str)
for p in (res_path, f"{WORK}/{RUN_NAME}_heldout_{CV_EVAL}_pred.json", f"{WORK}/{RUN_NAME}_cvmodel_state.pt",
          f"{WORK}/{RUN_NAME}_final_state.pt"):
    upload_model_file(p)
h, ti = RESULTS["heldout"], RESULTS["technique_info"]
print("=" * 100)
print(f"RESULT {RUN_NAME}: held-out {CV_EVAL} strict macro-F1 {h['macro_F1']} (P {h['macro_P']}, R {h['macro_R']}; "
      f"no-repair {h['macro_F1_without_boundary_repair']}) | best epoch {RESULTS['cv']['best_epoch']} | "
      f"trainable {ti['params_trainable']:,} ({ti['trainable_pct_of_backbone']}% of {PHASE1_PARAMS:,}) | "
      f"peak mem {RESULTS['cv']['peak_mem_gb']} GB | train {RESULTS['cv']['train_seconds']}s | total {RESULTS['notebook_minutes']} min")
print("per-label F1:", {l: d["F1"] for l, d in h["per_label"].items()})
print("=" * 100)
print("Submit:", f"submission_{RUN_NAME}.zip", "| send me the RESULT block above, or the file", os.path.basename(res_path))
'''


def build(exp):
    fam = FAMILIES[exp["family"]]
    cells = []
    md = lambda s: cells.append(nbf.v4.new_markdown_cell(s))
    code = lambda s: cells.append(nbf.v4.new_code_cell(s))
    md(f"""# CS-781 Phase 2 (L-NER) — `{exp['file']}`

**Phase-1 model:** {fam['title']}
**Technique:** {exp['what']}

What the notebook does (one Run All):
1. downloads the Phase-2 data, your Phase-1 tokenizer and checkpoint from the bucket; checks the weights loaded (next-token loss on NER text);
2. adds a token-classification head (BIO tags over the 12 labels) and applies the technique;
3. trains on **{'fold1'}**, scores the **strict macro-F1 on fold2 after every epoch** and keeps the best epoch (2-fold protocol: the test fold is never used);
4. full held-out report: per-label P/R/F1, boundary vs label errors, label confusions, the official `evaluate_phase2.py` on the same predictions, trainable parameters, peak GPU memory, speed;
5. retrains on fold1+fold2 for the chosen number of epochs, predicts the 35 test documents, validates the format and writes `submission_{exp['file']}.zip` (`prediction.json` at the zip root);
6. uploads the results JSON, held-out predictions and the trained weights (only the changed parameters) to the bucket.

**Kaggle:** Accelerator *GPU T4* (one GPU is used), secrets `HF` and `HF_WRITE`. Expected time ≈ 25–50 min (not measured yet). Set `QUICK_TEST = True` first for a ~5-min end-to-end check if you like.

All settings in the next cell are **your choices**; the values there are untuned placeholders.""")
    md("## 1. Settings")
    code("import json\n" + settings_cell(exp, fam))
    md("## 2. Setup: login, bucket")
    code(SETUP_CELL)
    md("## 3. Phase-2 data")
    code(DATA_CELL)
    md("## 4. Your Phase-1 BPE tokenizer (class unchanged from Phase 1), loaded from the bucket")
    code(BPE_SRC + TOK_LOAD)
    md("## 5. Character offsets, BIO tags, span decoding, boundary repair, strict scorer")
    code(ALIGN_SRC)
    md(f"## 6. Phase-1 model code (copied unchanged from {fam['source_nb']})")
    code(fam["pre"] + "\n" + fam["model_src"])
    md("## 7. NER backbone wrapper (hidden states instead of next-token logits; optional bidirectional / prefix attention)")
    code(fam["backbone_src"] + "\n\n" + fam["build_src"])
    md("## 8. NER engine: head, adaptation methods (full FT, partial FT, BitFit, LoRA, LoRMA, adapters, prompt, prefix), training, inference")
    code(CORE_SRC + "\n\nEVALUATOR_SRC = " + repr(EVAL_SRC))
    md("## 9. Tokenize the documents (with character offsets) and learn the boundary-repair tables")
    code(PREP_CELL)
    md("## 10. Load the Phase-1 checkpoint and check it")
    code(LOAD_CELL)
    md("## 11. Fine-tune on fold1, select the epoch on fold2")
    code(CV_CELL)
    md("## 12. Held-out report (for the write-up)")
    code(HELDOUT_CELL)
    md("## 13. Final model, test predictions, submission zip")
    code(FINAL_CELL)
    md("## 14. Save everything to the bucket and print the result block")
    code(SAVE_CELL)
    nb = nbf.v4.new_notebook()
    nb["cells"] = cells
    nb["metadata"] = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                      "language_info": {"name": "python"}}
    nbf.validate(nb)
    path = OUT + exp["file"] + ".ipynb"
    nbf.write(nb, path)
    return path


if __name__ == "__main__":
    for e in EXPERIMENTS:
        print("written", build(e))
