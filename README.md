# MiniLLM — Complete Project Documentation

*A from-scratch 57.5M parameter GPT-style language model — trained, fine-tuned, and deployed end-to-end on a single consumer GPU.*

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Architecture](#2-architecture)
3. [Project Structure](#3-project-structure)
4. [Configuration Reference](#4-configuration-reference)
5. [Data Pipeline](#5-data-pipeline)
6. [Pretraining](#6-pretraining)
7. [Fine-tuning](#7-fine-tuning)
8. [Inference & Generation](#8-inference--generation)
9. [Backend API](#9-backend-api)
10. [Frontend](#10-frontend)
11. [Setup & Installation](#11-setup--installation)
12. [Running the Full Stack](#12-running-the-full-stack)
13. [Checkpoint Format](#13-checkpoint-format)
14. [System Design](#14-system-design)
15. [Troubleshooting](#15-troubleshooting)

---

## 1. Project Overview

### 1.1 Motivation

Most language model projects today start from a pretrained checkpoint — GPT-2, LLaMA, Mistral — and fine-tune from there. MiniLLM was built to answer a different question: *can you build, train, and deploy a useful language model completely from scratch, on consumer hardware, with no pretrained weights at all?*

The goal was not to beat GPT-4. It was to deeply understand every layer of the stack — from tokenisation and transformer architecture, through pretraining dynamics, fine-tuning strategies, and all the way to a production-grade web interface with authentication and real-time streaming. Every line of this project was written by hand.

### 1.2 Goals & Scope

- Build a GPT-style transformer from scratch in PyTorch with no pretrained weights
- Pretrain on ~150M tokens of real text on a single consumer GPU (RTX 3050 6GB)
- Fine-tune into 4 distinct task personalities: farming advice, storytelling, poetry, Q&A
- Build a production-ready REST API with JWT auth, email verification, and SSE streaming
- Build a ChatGPT-like frontend with chat history, guest mode, and real-time token display
- Keep the full stack runnable on a single machine without cloud dependencies

**Out of scope:** RLHF, multi-GPU training, model quantisation, mobile support.

### 1.3 What Makes It Unique

- **Zero pretrained weights** — every parameter initialised randomly and trained from scratch
- **Wikipedia context injection** — backend auto-fetches relevant Wikipedia summaries and prepends them to the prompt, grounding factual responses in real knowledge
- **Task-aware generation** — four fine-tuned checkpoints with different generation configs, auto-selected by keyword detection
- **Full vertical stack** — ML research, ML engineering, backend, and frontend in one project
- **Resumable everything** — data pipeline, pretraining, and fine-tuning all checkpoint and resume gracefully

### 1.4 Tech Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
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

### 1.5 Personalities

| Personality | Task | Training Data |
|-------------|------|---------------|
| farmer | Agricultural advice | 50 hand-written examples x10 = 500 |
| story | Creative storytelling | TinyStories (3,000 samples) |
| poetry | Poem generation | merve/poetry + 20 manual poems |
| qa | General Q&A | SQuAD (3,000 samples) |

---

## 2. Architecture

### 2.1 Model — `model/gpt.py`

Decoder-only transformer (GPT-2 style) implemented from scratch in PyTorch.

```
Input tokens
     |
Token Embedding (vocab_size -> embed_dim)
  +  Position Embedding (max_seq_len -> embed_dim)  +  Dropout
     |
+-----------------------------+
|   Transformer Block x 10   |
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

**Key design decisions:**

- **Weight tying** — token embedding and output head share weights, saving ~25M parameters
- **Pre-norm** — LayerNorm before attention/FFN for training stability
- **Causal masking** — upper triangular mask via `torch.triu` inside MultiHeadAttention
- **Fused QKV** — single `Linear(embed_dim, 3*embed_dim)` computes Q, K, V together
- **No bias in attention** — qkv and out projections use `bias=False`

### 2.2 Model Size

| Parameter | Value |
|-----------|-------|
| EMBED_DIM | 512 |
| NUM_HEADS | 8 (head dim = 64) |
| NUM_LAYERS | 10 |
| FFN_DIM | 2048 (4x embed_dim) |
| MAX_SEQ_LEN | 512 tokens |
| VOCAB_SIZE | 50,257 (GPT-2 BPE) |
| DROPOUT | 0.1 |
| **Total Parameters** | **57,498,112 (~57.5M)** |

---

## 3. Project Structure

```
MiniLLM/
├── config.py                  # All hyperparameters and paths
├── train.py                   # Pretraining loop
├── finetune.py                # Fine-tuning loop (4 tasks)
├── chat.py                    # CLI chat interface
├── model/
│   └── gpt.py                 # MiniLLM transformer architecture
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

## 4. Configuration Reference

| Variable | Value | Description |
|----------|-------|-------------|
| EMBED_DIM | 512 | Embedding dimension |
| NUM_HEADS | 8 | Attention heads |
| NUM_LAYERS | 10 | Transformer blocks |
| FFN_DIM | 2048 | Feed-forward hidden size |
| MAX_SEQ_LEN | 512 | Context window |
| BATCH_SIZE | 4 | Per-step batch |
| GRAD_ACCUM_STEPS | 16 | Effective batch = 64 |
| LEARNING_RATE | 3e-4 | Peak LR (pretrain) |
| FINETUNE_LR | 1e-4 | Peak LR (fine-tune) |
| NUM_EPOCHS | 3 | Pretrain epochs |
| FINETUNE_EPOCHS | 15 | Fine-tune epochs per task |
| USE_FP16 | True | Mixed precision training |
| GRAD_CLIP | 1.0 | Gradient norm clipping |
| CONTEXT_WINDOW | 512 | Alias for MAX_SEQ_LEN (backend) |
| INFERENCE_PROMPTS | = PROMPTS | Alias used by backend/main.py |

---

## 5. Data Pipeline

| Source | Tokens | Description |
|--------|--------|-------------|
| Shakespeare | ~0.34M | Complete works |
| Wikipedia (simple) | ~70.2M | wikimedia/wikipedia 20231101.simple |
| OpenWebText | ~76.4M | Web text corpus |
| **Total** | **~150M** | 95% train / 5% val split |

```bash
python data/pipeline.py   # ~6 min, fully resumable — safe to interrupt and rerun
```

---

## 6. Pretraining

**Script:** `train.py` — AdamW optimizer with weight decay 0.1 on weight matrices, 0.0 on biases/LayerNorm. LR schedule: linear warmup then cosine decay to 0. fp16 autocast + GradScaler. Gradient accumulation x16 (effective batch 64).

| File | Saved when | Contents |
|------|-----------|---------|
| best.pt | Val loss improves | model, optimizer, scheduler, scaler, losses |
| last.pt | Every 500 steps + end of epoch | Same — for resuming |

```bash
python train.py   # prompts to resume if last.pt exists
```

---

## 7. Fine-tuning

Loads `best.pt`, trains 15 epochs at LR 1e-4. Saves `{task}_best.pt` and `{task}_last.pt` per task. Data is built automatically on first run and cached as JSONL.

| Task | Source | Samples | Notes |
|------|--------|---------|-------|
| farmer | Hand-written (10 categories) | 50 x 10 = 500 | Indian farming context |
| story | roneneldan/TinyStories | 3,000 | Filtered 150-1500 chars |
| poetry | merve/poetry + manual | ~3,000+ | Old English ages excluded |
| qa | rajpurkar/squad | 3,000 | Deduplicated, answers 5-300 chars |

```bash
python finetune.py --task farmer
python finetune.py --task story
python finetune.py --task poetry
python finetune.py --task qa
```

---

## 8. Inference & Generation

### Task Detection

| Task | Keywords |
|------|---------|
| farmer | crop, farm, soil, wheat, rice, fertilizer, irrigation, pest, harvest, seed, disease |
| story | story, tale, once, fiction, narrative, adventure, character, write a story |
| poetry | poem, poetry, verse, rhyme, haiku, sonnet, stanza, lyric, write a poem |
| qa | what, who, when, where, why, how, explain, define, meaning (default) |

### Generation Algorithm

Autoregressive sampling — one token at a time: encode prompt → temperature scaling → repetition penalty (last 30 tokens) → top-k filtering → top-p nucleus filtering → softmax sample → stop on EOS or `### END`.

### Per-task Config

| Task | max_tokens | temperature | top_k | top_p | rep_penalty |
|------|-----------|-------------|-------|-------|-------------|
| farmer | 200 | 0.70 | 40 | 0.90 | 1.3 |
| story | 300 | 0.85 | 50 | 0.92 | 1.2 |
| poetry | 150 | 0.90 | 60 | 0.95 | 1.1 |
| qa | 200 | 0.60 | 30 | 0.85 | 1.3 |

---

## 9. Backend API

**FastAPI · Port 8000 · PostgreSQL · JWT (access=24h, refresh=30d)**

| Method | Endpoint | Auth | Description |
|--------|---------|------|-------------|
| POST | /auth/register | No | Register — sends verification email |
| POST | /auth/verify-email | No | Verify email with token |
| POST | /auth/login | No | Login → access_token + refresh_token |
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

### SSE Event Format

```json
{"type": "task",  "task": "qa",    "used_wiki": true}
{"type": "token", "token": "The"}
{"type": "done"}
{"type": "error", "message": "..."}
```

---

## 10. Frontend

**React 18 + Vite · Port 5173 · React Router v6 · Pure CSS dark theme**

| Route | Page | Description |
|-------|------|-------------|
| / | Chat | Main interface — guest mode if not logged in |
| /login | Login | Email + password sign in |
| /register | Register | Create account |
| /verify-email | VerifyEmail | Handles email verification link |
| /forgot-password | ForgotPassword | Request password reset |
| /reset-password | ResetPassword | Set new password |

JWT stored in `localStorage`. SSE passes token as `?token=` query param since `EventSource` cannot send headers.

---

## 11. Setup & Installation

```bash
conda create -n myenv python=3.10 && conda activate myenv
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install transformers datasets fastapi uvicorn sqlalchemy psycopg2-binary
pip install python-jose bcrypt python-dotenv "pydantic[email]" requests

cd ~/MiniLLM/frontend && npm install
```

### `backend/.env`

```env
DATABASE_URL=postgresql://user:password@localhost:5432/minillm
JWT_SECRET=your-secret-key-here
SMTP_HOST=smtp.gmail.com
SMTP_PORT=587
SMTP_EMAIL=your@gmail.com
SMTP_PASS=your-app-password
APP_URL=http://localhost:5173
```

### `frontend/.env`

```env
VITE_API_URL=http://localhost:8000
```

---

## 12. Running the Full Stack

```bash
# Step 1 — Data pipeline (first time only, ~6 min)
python data/pipeline.py

# Step 2 — Pretrain (first time only)
python train.py

# Step 3 — Fine-tune all tasks (first time only)
python finetune.py --task farmer
python finetune.py --task story
python finetune.py --task poetry
python finetune.py --task qa

# Step 4 — Start backend
cd ~/MiniLLM
python -m uvicorn backend.main:app --reload
# Running at http://127.0.0.1:8000

# Step 5 — Start frontend
cd ~/MiniLLM/frontend
npm run dev
# Open http://localhost:5173
```

---

## 13. Checkpoint Format

```python
# All checkpoints (pretrain + fine-tune) saved as:
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

## 14. System Design

### 14.1 System Architecture

```
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

### 14.2 ML Training Pipeline

```
STAGE 1 - DATA PREPARATION (data/pipeline.py)
----------------------------------------------
Download + tokenise 3 sources -> cache as .npy
  Shakespeare  ->  0.34M tokens
  Wikipedia    ->  70.2M tokens
  OpenWebText  ->  76.4M tokens
  Total 150M   ->  train.npy (142.5M) + val.npy (7.5M)

STAGE 2 - PRETRAINING (train.py)
----------------------------------------------
MiniLLM (random init) + AdamW + cosine LR + fp16
  For each of 3 epochs:
    Forward -> (logits, loss) -> backward (accum x16)
    Every 16 steps: unscale -> clip -> step -> zero_grad
    End of epoch: eval val loss
      if best -> save best.pt
      always  -> save last.pt
Output: checkpoints/best.pt

STAGE 3 - FINE-TUNING (finetune.py --task X)
----------------------------------------------
Load best.pt -> build JSONL data -> train 15 epochs at 1e-4
  if best val -> save {task}_best.pt
  always      -> save {task}_last.pt
Run once per task: farmer, story, poetry, qa

STAGE 4 - INFERENCE (backend)
----------------------------------------------
Startup: preload_all() loads all 4 models into GPU cache
Request: generate_stream() yields tokens -> SSE events
```

### 14.3 ER Diagram — Full Schema

```
+----------------------------------------------+
|                   users                      |
+----------------------------------------------+
| id            VARCHAR(32)   PK  NOT NULL     |
| email         VARCHAR(255)  UNIQUE NOT NULL  |
| username      VARCHAR(50)   UNIQUE NOT NULL  |
| password_hash VARCHAR(255)  NOT NULL         |
| is_verified   BOOLEAN       NOT NULL         |
| created_at    DATETIME      NOT NULL         |
+------------------+---------------------------+
                   | 1
       +-----------+-----------------------------+
       | 0..*  (CASCADE)                        | 0..*  (CASCADE)
+------v---------------------------+  +---------v--------------------------+
|           chats                  |  |          email_tokens              |
+----------------------------------+  +------------------------------------+
| id         VARCHAR(32) PK        |  | id         VARCHAR(32)  PK         |
| user_id    VARCHAR(32) FK        |  | user_id    VARCHAR(32)  FK         |
| title      VARCHAR(100)          |  | token      VARCHAR(100) UNIQUE     |
|            default "New Chat"    |  | token_type VARCHAR(10)  NOT NULL   |
| created_at DATETIME NOT NULL     |  |   CHECK IN ('verify','reset')      |
| updated_at DATETIME NOT NULL     |  | expires_at DATETIME     NOT NULL   |
+-------------------+--------------+  | used       BOOLEAN  default False  |
                    | 1              +------------------------------------+
                    | 0..*  (CASCADE)
+-------------------v----------------------------------+
|                  messages                            |
+------------------------------------------------------+
| id         VARCHAR(32)   PK  NOT NULL                |
| chat_id    VARCHAR(32)   FK  NOT NULL                |
| role       VARCHAR(10)   NOT NULL                    |
|            CHECK IN ('user','assistant')             |
| content    TEXT          NOT NULL                    |
| task       VARCHAR(10)   NULL  (farmer/story/qa...)  |
| used_wiki  BOOLEAN       NOT NULL  default False     |
| created_at DATETIME      NOT NULL                    |
+------------------------------------------------------+

Cardinality:
  users  ||--o{  chats         one user -> zero or many chats
  users  ||--o{  email_tokens  one user -> zero or many tokens
  chats  ||--o{  messages      one chat -> zero or many messages

Cascade: DELETE user  -> removes chats + messages + tokens
         DELETE chat  -> removes its messages
```

### 14.4 Wikipedia Injection — Detailed Flow

```
User sends: "what causes yellow leaves in wheat"
       |
       v
build_prompt(task="farmer", user_input, prompts)
       |
       +-- task in SKIP_TASKS? (story / poetry)
       |     YES -> skip Wikipedia, return plain prompt
       |
       +-- NO -> search_wikipedia(user_input)
             |
             +-- GET wikipedia.org/w/api.php
             |     ?action=opensearch&search="..."&limit=1
             |     -> titles: ["Wheat"]
             |
             +-- GET wikipedia.org/api/rest_v1/page/summary/Wheat
             |     -> extract: "Wheat is a grass widely cultivated..."
             |
             +-- re.sub(r'\s+', ' ', extract)   # clean whitespace
             +-- truncate to 300 chars at last sentence boundary
             |
             +-- return:
                  "### Context: Wheat is a grass widely cultivated...
                   ### Task: Give farming advice for the following.
                   ### Situation: what causes yellow leaves in wheat
                   ### Advice:"

SSE event: {"type":"task","task":"farmer","used_wiki":true}
Frontend: shows Wikipedia badge on message bubble
```

### 14.5 Frontend Component Tree

```
App.jsx (React Router v6)
|
+-- /login           -> Login.jsx
+-- /register        -> Register.jsx
+-- /verify-email    -> VerifyEmail.jsx
+-- /forgot-password -> ForgotPassword.jsx
+-- /reset-password  -> ResetPassword.jsx
+-- /                -> ProtectedRoute
      +-- logged in  -> Chat.jsx
      +-- guest      -> Chat.jsx (guest=true)
            |
            +-- [state] chats, activeChatId, messages, streaming
            |
            +-- Sidebar.jsx
            |     +-- logo + New Chat button (PenSquare icon)
            |     +-- chat list
            |     |     +-- ChatItem: title | edit (Edit2) | delete (Trash2)
            |     +-- footer
            |           +-- [auth]  avatar + username + Logout (LogOut)
            |           +-- [guest] Login (LogIn) + Register (UserPlus)
            |
            +-- guest-banner div (only when !user)
            |
            +-- ChatWindow.jsx
            |     +-- [empty] empty-state: icon + hint pills
            |     +-- messages.map -> MessageBubble.jsx
            |           +-- [user]      right green bubble
            |           +-- [assistant] left grey bubble
            |                 +-- task badge: farmer/story/poetry/qa
            |                 +-- Wikipedia badge (if used_wiki=true)
            |                 +-- bubble-text (pre-wrap)
            |                 +-- blinking cursor if streaming
            |
            +-- InputBar.jsx
                  +-- textarea (rows=1, auto-grow)
                  +-- Send button (Send icon, disabled if streaming)

AuthContext.jsx (wraps entire app)
  +-- user state  (null = guest)
  +-- login(email, password)
  +-- logout()
```

### 14.6 Chat Message Lifecycle (End-to-End)

```
1.  User types "what is crop rotation?" -> presses Enter
2.  InputBar.submit() -> Chat.sendMessage(text)
3.  Optimistic UI: append userMsg + asstMsg{streaming:true} to state
4.  If no activeChatId: POST /chats -> create new chat, set activeChatId
5.  Open: new EventSource("/chat/stream?message=...&chat_id=...&token=...")
6.  Backend: validate JWT token from query param
7.  Backend: detect_task() -> "qa" (highest keyword score)
    SSE: {"type":"task","task":"qa","used_wiki":false}
8.  Backend: search_wikipedia() -> fetches and trims summary
    SSE: {"type":"task","task":"qa","used_wiki":true}
9.  Backend: get_model("qa") -> cached GPU model (no disk I/O)
10. Backend: build prompt with context + task template
11. Backend: generate_stream() loop:
      SSE: {"type":"token","token":"Crop"}
      SSE: {"type":"token","token":" rotation"}
      ... up to 200 tokens ...
      SSE: {"type":"done"}
12. Frontend onmessage:
      type=task  -> update badge on streaming bubble
      type=token -> append to bubble content (live render)
      type=done  -> streaming=false, cursor disappears
13. Backend (after done): INSERT user + assistant messages to DB
14. Backend: UPDATE chat title = first 40 chars of message
15. Frontend: GET /chats -> refresh sidebar with new title
```

### 14.7 Security Design

| Concern | Approach |
|---------|---------|
| Password storage | bcrypt with per-password salt, constant-time comparison via `bcrypt.checkpw` |
| JWT tokens | HS256 signed, access=24h, refresh=30d, payload contains only user ID + type + exp |
| Email tokens | `secrets.token_urlsafe(32)` — 256 bits entropy, single-use, verify=24h / reset=1h |
| SQL injection | SQLAlchemy ORM throughout — no raw SQL strings anywhere |
| CORS | `allow_origins=["*"]` in dev — restrict to frontend domain in production |
| TLS / HTTPS | Handled at Nginx reverse proxy level, not in the app |

### 14.8 Performance & Optimization

| Technique | Benefit |
|-----------|---------|
| fp16 mixed precision | ~2x memory reduction, faster matrix ops on NVIDIA GPU |
| torch.compile | 10-30% inference speedup via TorchInductor graph fusion |
| Model caching | All 4 models preloaded at startup — zero disk I/O per request |
| Gradient accumulation x16 | Effective batch size 64 on a 6GB GPU without OOM |
| Weight tying | Saves ~25M parameters, improves language model perplexity |
| Context truncation | Truncates to `CONTEXT_WINDOW - max_new_tokens` to ensure generation room |

### 14.9 Deployment (Docker + Nginx)

```nginx
# Nginx — critical SSE config
location /chat/stream {
    proxy_pass http://localhost:8000/chat/stream;
    proxy_buffering off;       # MUST be off — without this SSE won't stream
    proxy_cache off;
    proxy_set_header Connection '';
    chunked_transfer_encoding on;
}
```

### 14.10 Known Limitations & Future Improvements

| Limitation | Future Fix |
|-----------|-----------|
| 57.5M params — limited output quality | Scale to 125M-350M with more training data |
| 512 token context window | Increase + use RoPE positional embeddings |
| No conversation memory | Pass last N message turns in prompt |
| Single GPU only | Add DDP / model parallelism for multi-GPU |
| IP guest limit not persistent across restarts | Redis-backed counter |
| CORS open in development | Restrict allow_origins to frontend domain |
| No user feedback loop | Thumbs up/down -> RLHF fine-tuning pipeline |
| Large GPU memory per model | INT8 / INT4 quantisation to reduce footprint |

---

## 15. Troubleshooting

| Error | Fix |
|-------|-----|
| `No module named 'backend'` | Run from `~/MiniLLM` root: `python -m uvicorn backend.main:app --reload` |
| `cannot import MiniLLM from 'model'` | Use: `from model.gpt import MiniLLM` |
| `MiniLLM missing positional arguments` | Pass each arg: `vocab_size, embed_dim, num_heads, num_layers, ffn_dim, max_seq_len` |
| `Unexpected key 'model' in state_dict` | Use: `model.load_state_dict(ckpt["model"])` |
| `tuple indices must be integers` | Model returns `(logits, loss)` — unpack: `logits, _ = model(ids)` |
| `INFERENCE_PROMPTS not found` | Add to `config.py`: `INFERENCE_PROMPTS = PROMPTS` and `CONTEXT_WINDOW = MAX_SEQ_LEN` |
| Frontend shows black screen | `src/` folder is empty — create all component files manually |
| Connection error in frontend | Check backend on `:8000` and `VITE_API_URL=http://localhost:8000` in `frontend/.env` |
| SSE tokens not streaming (Nginx) | Add `proxy_buffering off` to Nginx location block for `/chat/stream` |

---

<p align="center"><sub>MiniLLM · Built from scratch · RTX 3050 6GB · 57.5M parameters · 150M training tokens</sub></p>
