# LLM-Mini

> A GPT-style language model built entirely from scratch — trained, fine-tuned, and served through a full-stack chat application on a single consumer GPU.

![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-ee4c2c?logo=pytorch)
![FastAPI](https://img.shields.io/badge/FastAPI-backend-009688?logo=fastapi)
![React](https://img.shields.io/badge/React-18-61dafb?logo=react)
![Params](https://img.shields.io/badge/Parameters-57.5M-purple)
![Tokens](https://img.shields.io/badge/Training%20Tokens-150M-orange)

---

## What is LLM-Mini?

LLM-Mini is a compact, decoder-only transformer written **from the ground up in PyTorch** — no pretrained weights, no shortcuts. Every parameter was randomly initialized and trained on ~150M tokens of real text (Wikipedia + OpenWebText), then fine-tuned into four distinct personalities, wrapped in a production-style FastAPI backend with JWT auth and SSE streaming, and served through a ChatGPT-style React frontend.

It answers one question: *can a genuinely useful language model be built end-to-end on consumer hardware?*



---

## Highlights

| Feature | Details |
| --- | --- |
| **Zero pretrained weights** | Every one of the 57.5M parameters trained from random initialization |
| **Live Wikipedia grounding** | Backend auto-fetches relevant Wikipedia summaries and injects them into prompts |
| **Four personalities** | Farmer, Storyteller, Poet, and Q&A — auto-detected from each message |
| **Full auth system** | JWT access + refresh tokens, bcrypt hashing, email verification, password reset |
| **Real-time streaming** | Tokens stream over SSE and render live, ChatGPT-style |
| **Resumable everything** | Data pipeline, pretraining, and fine-tuning all checkpoint and resume cleanly |
| **Single GPU** | Trained end-to-end on an RTX 3050 6GB with fp16 mixed precision and gradient accumulation |

---

## Architecture

### Model

A GPT-2–style decoder-only transformer, implemented from scratch in PyTorch:

```javascript
Input tokens
     |
Token Embedding (vocab_size -> embed_dim)
  +  Position Embedding (max_seq_len -> embed_dim)  +  Dropout
     |
+-----------------------------+
|   Transformer Block x 10    |
|  LayerNorm                  |
|  -> MultiHeadAttention      |  <- causal mask (upper triangular)
|  -> Residual Add            |
|  LayerNorm                  |
|  -> FeedForward (GELU)      |
|  -> Residual Add            |
+-----------------------------+
     |
Final LayerNorm -> Linear Head -> Logits
(optional) cross_entropy(logits, targets) -> loss
```

| Hyperparameter | Value |
| --- | --- |
| Embedding dim | 512 |
| Attention heads | 8 (head dim = 64) |
| Layers | 10 |
| FFN dim | 2048 (4x embed_dim) |
| Context window | 512 tokens |
| Vocab size | 50,257 (GPT-2 BPE) |
| Dropout | 0.1 |
| **Total parameters** | **57,498,112 (~57.5M)** |

Key design decisions: **weight tying** (token embedding and output head share weights, saving ~25M parameters), **pre-norm** LayerNorm for training stability, **fused QKV** projection, and causal masking via `torch.triu`.

### Tech Stack

| Layer | Technology | Purpose |
| --- | --- | --- |
| Model | PyTorch 2.x | Transformer architecture and training |
| Tokenizer | HuggingFace Transformers (GPT-2) | BPE tokenisation |
| Training data | HuggingFace Datasets | Wikipedia, OpenWebText, TinyStories, SQuAD, poetry |
| Backend | FastAPI + Uvicorn | REST API and SSE streaming |
| Database | PostgreSQL + SQLAlchemy | Users, chats, messages, tokens |
| Auth | JWT (PyJWT) + bcrypt | Token-based auth and password hashing |
| Email | SMTP (smtplib) | Verification and password reset emails |
| Frontend | React 18 + Vite | Chat UI |
| Routing | React Router v6 | Client-side navigation |
| Styling | Pure CSS (CSS variables) | Dark theme, no UI library |
| Context retrieval | Wikipedia REST API | Real-time factual grounding |
| Hardware | NVIDIA RTX 3050 6GB + CUDA | GPU training and inference |

### System Design

```javascript
+---------------------------------------------------------------+
|                       USER BROWSER                            |
|  React + Vite (port 5173)                                     |
|  +-----------+  +-------------+  +------------------------+  |
|  | Auth Pages|  |   Sidebar   |  |     Chat Window        |  |
|  |           |  | (history)   |  |  (SSE token streaming) |  |
|  +-----------+  +-------------+  +------------------------+  |
+---------------------------|-----------------------------------+
                            | HTTP / SSE
                            v
+---------------------------------------------------------------+
|                FastAPI BACKEND (port 8000)                    |
|  +-------------+  +-------------+  +-------------------+     |
|  | Auth Routes |  | Chat Routes |  |   /chat/stream    |     |
|  | /auth/*     |  | /chats/*    |  |   (SSE endpoint)  |     |
|  +------+------+  +------+------+  +---------+---------+     |
|         |                |                   |               |
|         v                v                   v               |
|  +-------------------------+     +---------------------+     |
|  |   PostgreSQL Database   |     |    Model Loader     |     |
|  | users/chats/messages/   |     |  (GPU memory cache) |     |
|  | email_tokens            |     +----------+----------+     |
|  +-------------------------+                |               |
|                                             v               |
|                                  +--------------------+      |
|                                  | Wikipedia Search   |      |
|                                  | (context inject)   |      |
|                                  +--------------------+      |
+---------------------------------------------|----------------+
                                              v
+---------------------------------------------------------------+
|                   ML INFERENCE LAYER (CUDA fp16)              |
|  +----------+ +----------+ +----------+ +----------+         |
|  |farmer_   | |story_    | |poetry_   | |qa_       |         |
|  |best.pt   | |best.pt   | |best.pt   | |best.pt   |         |
|  +----------+ +----------+ +----------+ +----------+         |
|           MiniLLM Transformer (57.5M params)                  |
+---------------------------------------------------------------+
```

---

## Personalities

LLM-Mini is fine-tuned into four task personalities, auto-selected by keyword detection:

| Personality | Task | Training Data |
| --- | --- | --- |
| farmer | Agricultural advice | 500 hand-written examples (10 categories) |
| story | Creative storytelling | TinyStories (3,000 samples) |
| poetry | Poem generation | merve/poetry + 20 manual poems |
| qa | General Q&A | SQuAD (3,000 samples) |

Per-task generation configuration:

| Task | max_tokens | temperature | top_k | top_p | rep_penalty |
| --- | --- | --- | --- | --- | --- |
| farmer | 200 | 0.70 | 40 | 0.90 | 1.3 |
| story | 300 | 0.85 | 50 | 0.92 | 1.2 |
| poetry | 150 | 0.90 | 60 | 0.95 | 1.1 |
| qa | 200 | 0.60 | 30 | 0.85 | 1.3 |

---

## Data Pipeline

| Source | Tokens | Description |
| --- | --- | --- |
| Shakespeare | ~0.34M | Complete works |
| Wikipedia (simple) | ~70.2M | wikimedia/wikipedia 20231101.simple |
| OpenWebText | ~76.4M | Web text corpus |
| **Total** | **~150M** | 95% train / 5% val split |

```bash
python data/pipeline.py   # ~6 min, fully resumable — safe to interrupt and rerun
```

---

## Training

**Pretraining** (`train.py`) — AdamW with weight decay 0.1 on weight matrices, 0.0 on biases/LayerNorm. Linear warmup then cosine decay to 0. fp16 autocast + GradScaler. Gradient accumulation x16 (effective batch 64). Saves `best.pt` / `last.pt`.

**Fine-tuning** (`finetune.py`) — loads `best.pt`, trains 15 epochs at LR 1e-4 per task, saving `{task}_best.pt` / `{task}_last.pt`.

```bash
python train.py                       # prompts to resume if last.pt exists
python finetune.py --task farmer
python finetune.py --task story
python finetune.py --task poetry
python finetune.py --task qa
```

**Generation** — autoregressive sampling with temperature scaling, repetition penalty (last 30 tokens), top-k filtering, and top-p nucleus filtering.

---

## API Overview

FastAPI, port 8000, PostgreSQL, JWT (access = 24h, refresh = 30d).

| Method | Endpoint | Auth | Description |
| --- | --- | --- | --- |
| POST | /auth/register | No | Register — sends verification email |
| POST | /auth/verify-email | No | Verify email with token |
| POST | /auth/login | No | Login -> access_token + refresh_token |
| POST | /auth/forgot-password | No | Send reset link to email |
| POST | /auth/reset-password | No | Set new password with token |
| GET | /user/profile | Yes | Get current user details |
| PUT | /user/username | Yes | Update username |
| PUT | /user/password | Yes | Change password |
| DELETE | /user/account | Yes | Delete account + all data |
| GET | /chats | Yes | List all chats (by last updated) |
| POST | /chats | Yes | Create new chat |
| GET | /chats/{id}/messages | Yes | Get all messages in chat |
| PUT | /chats/{id} | Yes | Rename chat |
| DELETE | /chats/{id} | Yes | Delete chat + messages |
| GET | /chat/stream | Optional | SSE streaming generation |

SSE event format:

```javascript
{"type": "task",  "task": "qa",    "used_wiki": true}
{"type": "token", "token": "The"}
{"type": "done"}
{"type": "error", "message": "..."}
```

---

## Project Structure

```javascript
LLM-Mini/
├── config.py                  # All hyperparameters and paths
├── train.py                   # Pretraining loop
├── finetune.py                # Fine-tuning loop (4 tasks)
├── chat.py                    # CLI chat interface
├── model/
│   └── gpt.py                 # Transformer architecture
├── data/
│   ├── pipeline.py            # Data download and tokenisation
│   └── dataset.py             # PyTorch Dataset and DataLoader
├── checkpoints/
│   ├── best.pt / last.pt
│   ├── farmer_best.pt / farmer_last.pt
│   ├── story_best.pt  / story_last.pt
│   ├── poetry_best.pt / poetry_last.pt
│   └── qa_best.pt     / qa_last.pt
├── backend/
│   ├── main.py                # FastAPI app
│   ├── model_loader.py        # Model loading and caching
│   ├── generator.py           # Streaming token generation
│   ├── wikipedia_search.py    # Wikipedia context retrieval
│   ├── models/                # SQLAlchemy + Pydantic models
│   ├── routes/                # Route definitions
│   ├── services/              # Business logic
│   ├── controllers/           # Request handlers
│   └── core/                  # DB engine and config
└── frontend/
    └── src/
        ├── main.jsx / App.jsx
        ├── api.js
        ├── context/AuthContext.jsx
        ├── pages/             # Chat, Login, Register, etc.
        └── components/        # Sidebar, ChatWindow, InputBar, etc.
```

---

## Setup & Installation

```bash
conda create -n llm-mini python=3.10 && conda activate llm-mini
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install transformers datasets fastapi uvicorn sqlalchemy psycopg2-binary
pip install python-jose bcrypt python-dotenv "pydantic[email]" requests

cd frontend && npm install && cd ..
```

**`backend/.env`**

```javascript
DATABASE_URL=postgresql://user:password@localhost:5432/minillm
JWT_SECRET=your-secret-key-here
SMTP_HOST=smtp.gmail.com
SMTP_PORT=587
SMTP_EMAIL=your@gmail.com
SMTP_PASS=your-app-password
APP_URL=http://localhost:5173
```

**`frontend/.env`**

```javascript
VITE_API_URL=http://localhost:8000
```

---

## Running the Full Stack

```bash
# Step 1 — Data pipeline (first time only, ~6 min)
python data/pipeline.py

# Step 2 — Pretrain (first time only, requires NVIDIA GPU)
python train.py

# Step 3 — Fine-tune all tasks (first time only)
python finetune.py --task farmer
python finetune.py --task story
python finetune.py --task poetry
python finetune.py --task qa

# Step 4 — Start backend
python -m uvicorn backend.main:app --reload
# Running at http://127.0.0.1:8000

# Step 5 — Start frontend
cd frontend
npm run dev
# Open http://localhost:5173
```

---

## Checkpoint Format

```javascript
{
    "epoch":        int,         # last completed epoch (0-indexed)
    "step":         int,         # global optimizer step count
    "model":        state_dict,  # model weights only (key = "model")
    "optimizer":    state_dict,
    "scheduler":    state_dict,
    "scaler":       state_dict,  # fp16 GradScaler state
    "val_losses":   list[float],
}

# Loading weights:
ckpt = torch.load("checkpoints/farmer_best.pt", map_location="cpu")
model.load_state_dict(ckpt["model"])   # note: key is "model", not "model_state_dict"
```

---

## Known Limitations

| Limitation | Future Fix |
| --- | --- |
| 57.5M params — limited output quality | Scale to 125M-350M with more training data |
| 512 token context window | Increase + use RoPE positional embeddings |
| No conversation memory | Pass last N message turns in prompt |
| Single GPU only | Add DDP / model parallelism for multi-GPU |
| IP guest limit not persistent across restarts | Redis-backed counter |
| CORS open in development | Restrict allow_origins to frontend domain |
| No user feedback loop | Thumbs up/down -> RLHF fine-tuning pipeline |
| Large GPU memory per model | INT8 / INT4 quantisation to reduce footprint |

---

## Troubleshooting

| Error | Fix |
| --- | --- |
| `No module named 'backend'` | Run from repo root: `python -m uvicorn backend.main:app --reload` |
| `cannot import MiniLLM from 'model'` | Use: `from model.gpt import MiniLLM` |
| `Unexpected key 'model' in state_dict` | Use: `model.load_state_dict(ckpt["model"])` |
| `tuple indices must be integers` | Model returns `(logits, loss)` — unpack: `logits, _ = model(ids)` |
| Frontend shows black screen | `src/` folder is empty — check component files exist |
| Connection error in frontend | Check backend on `:8000` and `VITE_API_URL=http://localhost:8000` in `frontend/.env` |
| SSE tokens not streaming (Nginx) | Add `proxy_buffering off;` to the `/chat/stream` location block |

---

## License

See the original [MiniLLM repository](https://github.com/tathagat-git/MiniLLM) for licensing terms. Credit to the original author is retained.

---

*Built from scratch — 57.5M parameters — 150M training tokens — one consumer GPU.*
