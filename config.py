# ============================================================
# config.py — MiniLLM 60M
# ============================================================
# Architecture : 60M params (10 layers, 512 dim)
# Tokenizer    : GPT2 (50257 vocab, no custom tokenizer)
# Pretrain     : Wikipedia + Books + Shakespeare (~150M tokens)
# Fine-tunes   : Multiple personalities (story, poetry, farmer...)
# ============================================================

import os
import torch

# ── Paths ────────────────────────────────────────────────────
BASE_DIR        = os.path.dirname(os.path.abspath(__file__))
PROCESSED_DIR   = os.path.join(BASE_DIR, "data", "processed")
CHECKPOINT_DIR  = os.path.join(BASE_DIR, "checkpoints")
FINETUNE_DIR    = os.path.join(BASE_DIR, "finetune")

for d in [PROCESSED_DIR, CHECKPOINT_DIR, FINETUNE_DIR]:
    os.makedirs(d, exist_ok=True)

# ── Tokenizer ────────────────────────────────────────────────
TOKENIZER_NAME  = "gpt2"
VOCAB_SIZE      = 50257
EOS_TOKEN_ID    = 50256
PAD_TOKEN_ID    = 50256

# ── Model Architecture — 60M ─────────────────────────────────
# 12 * NUM_LAYERS * EMBED_DIM^2
# 12 * 10 * 512^2 = 31M  (embedding adds ~25M) → ~60M total
EMBED_DIM       = 512
NUM_HEADS       = 8         # 512 / 8 = 64 per head
NUM_LAYERS      = 10
FFN_DIM         = 2048      # 4 × EMBED_DIM
MAX_SEQ_LEN     = 512       # good balance of context vs speed
DROPOUT         = 0.1

# ── Pretrain Settings ────────────────────────────────────────
BATCH_SIZE          = 4
GRAD_ACCUM_STEPS    = 16        # effective batch = 64
LEARNING_RATE       = 3e-4
WEIGHT_DECAY        = 0.1
NUM_EPOCHS = 5
GRAD_CLIP           = 1.0
LOG_INTERVAL        = 50
SAVE_INTERVAL       = 500
USE_FP16            = True

# ── Pretrain Data ────────────────────────────────────────────
TARGET_TOKENS       = 150_000_000   # 150M tokens
TRAIN_RATIO         = 0.95
VAL_RATIO           = 0.05

# Processed token files
TRAIN_FILE          = os.path.join(PROCESSED_DIR, "train.npy")
VAL_FILE            = os.path.join(PROCESSED_DIR, "val.npy")

# ── Fine-tune Settings ───────────────────────────────────────
FINETUNE_LR         = 1e-4
FINETUNE_EPOCHS     = 15
FINETUNE_BATCH      = 4
FINETUNE_GRAD_ACCUM = 8         # effective batch = 32

# Fine-tune data paths
FINETUNE_STORY_TRAIN    = os.path.join(PROCESSED_DIR, "story_train.jsonl")
FINETUNE_STORY_VAL      = os.path.join(PROCESSED_DIR, "story_val.jsonl")
FINETUNE_POETRY_TRAIN   = os.path.join(PROCESSED_DIR, "poetry_train.jsonl")
FINETUNE_POETRY_VAL     = os.path.join(PROCESSED_DIR, "poetry_val.jsonl")
FINETUNE_FARMER_TRAIN   = os.path.join(PROCESSED_DIR, "farmer_train.jsonl")
FINETUNE_FARMER_VAL     = os.path.join(PROCESSED_DIR, "farmer_val.jsonl")
FINETUNE_HEALTH_TRAIN   = os.path.join(PROCESSED_DIR, "health_train.jsonl")
FINETUNE_HEALTH_VAL     = os.path.join(PROCESSED_DIR, "health_val.jsonl")
FINETUNE_QA_TRAIN       = os.path.join(PROCESSED_DIR, "qa_train.jsonl")
FINETUNE_QA_VAL         = os.path.join(PROCESSED_DIR, "qa_val.jsonl")

# ── Fine-tune Prompt Formats ─────────────────────────────────
# Every personality has its own prompt format.
# At inference: feed PREFIX + user_input + RESPONSE
# Model generates from RESPONSE onward.

PROMPTS = {
    "story": {
        "prefix":   "### Task: Write a short story about the following topic.\n### Topic: ",
        "response": "\n### Story:\n",
        "end":      "\n### END"
    },
    "poetry": {
        "prefix":   "### Task: Write a poem about the following.\n### Topic: ",
        "response": "\n### Poem:\n",
        "end":      "\n### END"
    },
    "farmer": {
        "prefix":   "### Task: Give farming advice for the following situation.\n### Situation: ",
        "response": "\n### Advice:\n",
        "end":      "\n### END"
    },
    "health": {
        "prefix":   "### Task: Give basic health information for the following.\n### Symptoms: ",
        "response": "\n### Response:\n",
        "end":      "\n### END"
    },
    "qa": {
        "prefix":   "### Use the context above to answer this question: ",
        "response": "\n### Answer:\n",
        "end":      "\n### END"
    },
}

# ── Generation Settings (defaults) ──────────────────────────
MAX_NEW_TOKENS  = 300
TEMPERATURE     = 0.8
TOP_K           = 40
TOP_P           = 0.9

# ── Per-personality generation configs ───────────────────────
# temperature : higher = more creative, lower = more focused
# top_k       : how many tokens to consider at each step
# top_p       : nucleus sampling threshold
# max_tokens  : max response length
# rep_penalty : penalize repeated tokens (1.0 = disabled)
GENERATION_CONFIGS = {
    "story": {
        "temperature": 0.9,   # flowing, natural narrative
        "top_k":       45,
        "top_p":       0.95,
        "max_tokens":  400,
        "rep_penalty": 1.2,   # avoid repeating phrases
    },
    "poetry": {
        "temperature": 1.0,   # creative, expressive
        "top_k":       50,
        "top_p":       0.95,
        "max_tokens":  200,
        "rep_penalty": 1.3,   # strongly avoid repetition
    },
    "farmer": {
        "temperature": 0.5,   # focused, factual advice
        "top_k":       20,
        "top_p":       0.85,
        "max_tokens":  250,
        "rep_penalty": 1.1,
    },
    "qa": {
        "temperature": 0.3,   # precise, deterministic answers
        "top_k":       10,
        "top_p":       0.80,
        "max_tokens":  100,
        "rep_penalty": 1.1,
    },
}

# ── Device ───────────────────────────────────────────────────
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ── Sanity checks ────────────────────────────────────────────
assert EMBED_DIM % NUM_HEADS == 0, \
    f"EMBED_DIM ({EMBED_DIM}) must be divisible by NUM_HEADS ({NUM_HEADS})"

# Estimate params
estimated = 12 * NUM_LAYERS * EMBED_DIM ** 2
print(f"[Config] Device          : {DEVICE}")
print(f"[Config] Estimated params: {estimated/1e6:.0f}M")
print(f"[Config] Effective batch : {BATCH_SIZE * GRAD_ACCUM_STEPS}")
print(f"[Config] Context window  : {MAX_SEQ_LEN} tokens")
print(f"[Config] fp16            : {USE_FP16}")
INFERENCE_PROMPTS = PROMPTS
CONTEXT_WINDOW = MAX_SEQ_LEN
