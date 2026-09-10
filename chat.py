# ============================================================
# chat.py — LLM-Mini Unified Chat Interface
# ============================================================
# Automatically detects topic and routes to correct model.
# Usage:
#   python chat.py                    # auto-detect mode
#   python chat.py --task story       # force a personality
#   python chat.py --task farmer
# ============================================================

import os
import sys
import argparse
import torch
from transformers import GPT2Tokenizer
import transformers
transformers.logging.set_verbosity_error()

import config
from model.gpt import build_model


# ── Keywords for auto-detection ───────────────────────────────
TASK_KEYWORDS = {
    "farmer": [
        "crop", "plant", "soil", "farm", "wheat", "rice", "cotton",
        "tomato", "onion", "fertilizer", "pest", "irrigation", "harvest",
        "seed", "field", "kisan", "kheti", "fasal", "mango", "sugarcane",
        "disease", "insect", "spray", "manure", "yield"
    ],
    "health": [
        "fever", "pain", "headache", "cough", "cold", "medicine", "doctor",
        "symptom", "sick", "ill", "disease", "blood", "pressure", "sugar",
        "diabetes", "injury", "wound", "pregnant", "child", "baby", "ache",
        "vomit", "diarrhea", "infection", "allergy", "breath"
    ],
    "poetry": [
        "poem", "poetry", "write a poem", "shayari", "verse", "rhyme",
        "sonnet", "haiku", "lyric", "ode", "stanza", "beautiful words",
        "sad poem", "love poem", "nature poem", "emotional"
    ],
    "story": [
        "story", "tale", "once upon", "write a story", "short story",
        "fiction", "narrative", "adventure", "fairy tale", "bedtime",
        "character", "plot", "imagine", "fantasy"
    ],
    "qa": [
        "what is", "who is", "when did", "why does", "how does",
        "explain", "tell me about", "what are", "define", "describe",
        "what was", "how many", "where is", "which"
    ],
}


def detect_task(user_input: str) -> str:
    """Auto-detect which personality to use based on input."""
    text = user_input.lower()
    scores = {task: 0 for task in TASK_KEYWORDS}

    for task, keywords in TASK_KEYWORDS.items():
        for kw in keywords:
            if kw in text:
                scores[task] += 1

    best_task = max(scores, key=scores.get)
    if scores[best_task] == 0:
        return "qa"  # default
    return best_task


# ── Model loading ─────────────────────────────────────────────

_loaded_models = {}


def load_model(task: str):
    """Load fine-tuned model for a task. Cache in memory."""
    if task in _loaded_models:
        return _loaded_models[task]

    ckpt_path = os.path.join(config.CHECKPOINT_DIR, f"{task}_best.pt")

    if not os.path.exists(ckpt_path):
        print(f"[Chat] No fine-tuned model for '{task}'.")
        print(f"  Run: python finetune.py --task {task}")
        # Fall back to pretrained
        ckpt_path = os.path.join(config.CHECKPOINT_DIR, "best.pt")
        if not os.path.exists(ckpt_path):
            print("[Chat] No pretrained model found either!")
            print("  Run: python train.py")
            sys.exit(1)
        print(f"[Chat] Using pretrained model as fallback.")

    print(f"[Chat] Loading {task} model from {ckpt_path}...")
    model = build_model(config)
    ckpt  = torch.load(ckpt_path, map_location=config.DEVICE,
                       weights_only=False)
    model.load_state_dict(ckpt["model"])
    model.eval()

    _loaded_models[task] = model
    print(f"[Chat] {task} model ready.")
    return model


# ── Generation ────────────────────────────────────────────────

@torch.no_grad()
def generate(
    prompt:      str,
    model,
    tokenizer,
    max_tokens:  int   = config.MAX_NEW_TOKENS,
    temperature: float = config.TEMPERATURE,
    top_k:       int   = config.TOP_K,
    top_p:       float = config.TOP_P,
    rep_penalty: float = 1.1,
) -> str:
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(config.DEVICE)

    if input_ids.shape[1] > config.MAX_SEQ_LEN - 50:
        input_ids = input_ids[:, -(config.MAX_SEQ_LEN - 50):]

    generated  = input_ids
    output_ids = []
    end_tokens = tokenizer.encode(config.PROMPTS["story"]["end"])

    for _ in range(max_tokens):
        with torch.amp.autocast('cuda', enabled=config.USE_FP16):
            logits, _ = model(generated[:, -config.MAX_SEQ_LEN:])

        logits = logits[:, -1, :] / temperature

        # ── Repetition penalty ────────────────────────────────
        # Divide probability of already-used tokens
        # Makes model avoid repeating same words/phrases
        if rep_penalty != 1.0:
            for token_id in set(output_ids):
                logits[0, token_id] /= rep_penalty

        # ── Top-k filtering ───────────────────────────────────
        if top_k > 0:
            top_vals, _ = torch.topk(logits, top_k)
            logits = logits.masked_fill(
                logits < top_vals[:, -1:], float("-inf")
            )

        # ── Top-p (nucleus) filtering ─────────────────────────
        if top_p < 1.0:
            sorted_logits, sorted_idx = torch.sort(logits, descending=True)
            cum_probs = torch.cumsum(
                torch.softmax(sorted_logits, dim=-1), dim=-1
            )
            remove = cum_probs - torch.softmax(sorted_logits, dim=-1) > top_p
            sorted_logits[remove] = float("-inf")
            logits = torch.zeros_like(logits).scatter_(
                1, sorted_idx, sorted_logits
            )

        probs      = torch.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, 1)

        output_ids.append(next_token.item())
        generated = torch.cat([generated, next_token], dim=1)

        if next_token.item() == config.EOS_TOKEN_ID:
            break

        if len(output_ids) >= len(end_tokens):
            if output_ids[-len(end_tokens):] == end_tokens:
                break

    text = tokenizer.decode(output_ids, skip_special_tokens=True)
    for end in ["### END", "### Task", "### Question",
                "### Topic", "### Situation", "### Symptoms"]:
        if end in text:
            text = text[:text.index(end)]
    return text.strip()


# ── Format prompt ─────────────────────────────────────────────

def format_prompt(task: str, user_input: str) -> str:
    cfg = config.PROMPTS[task]
    return cfg["prefix"] + user_input.strip() + cfg["response"]


# ── Print helpers ─────────────────────────────────────────────

TASK_EMOJI = {
    "story":  "📖",
    "poetry": "🎭",
    "farmer": "🌾",
    "health": "🏥",
    "qa":     "🧠",
}

TASK_LABEL = {
    "story":  "Story",
    "poetry": "Poem",
    "farmer": "Farming Advice",
    "health": "Health Info",
    "qa":     "Answer",
}


def print_response(task: str, response: str):
    emoji = TASK_EMOJI.get(task, "🤖")
    label = TASK_LABEL.get(task, "Response")
    width = 60
    print(f"\n{'═'*width}")
    print(f"  {emoji}  LLM-Mini — {label}")
    print(f"{'─'*width}")
    # Word wrap
    words = response.split()
    line  = "  "
    for word in words:
        if len(line) + len(word) + 1 > 58:
            print(line)
            line = "  " + word + " "
        else:
            line += word + " "
    if line.strip():
        print(line)
    print(f"{'═'*width}\n")


# ── Main ──────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="LLM-Mini Chat Interface")
    parser.add_argument("--task", type=str, default=None,
                        choices=list(TASK_KEYWORDS.keys()),
                        help="Force a specific personality")
    parser.add_argument("--temperature", type=float,
                        default=config.TEMPERATURE)
    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("  🤖  LLM-Mini — Multi-Personality AI")
    print("=" * 60)
    print("  Personalities available:")
    print("  🌾 Farmer Advisor  — crop diseases, farming tips")
    print("  🏥 Health Advisor  — symptoms, basic health info")
    print("  📖 Story Generator — short creative stories")
    print("  🎭 Poetry          — poems on any topic")
    print("  🧠 General QA      — answer any question")
    print()
    print("  Type your message — topic auto-detected!")
    print("  Type 'quit' to exit.")
    print("=" * 60)

    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    # Preload forced task if specified
    if args.task:
        load_model(args.task)
        print(f"\n[Chat] Forced mode: {args.task.upper()}")

    while True:
        try:
            user_input = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n\nGoodbye! 👋")
            break

        if not user_input:
            continue

        if user_input.lower() in ["quit", "exit", "bye"]:
            print("Goodbye! 👋")
            break

        # Detect or use forced task
        task = args.task if args.task else detect_task(user_input)
        print(f"[Detected: {task}]", end=" ", flush=True)

        # Load model
        model = load_model(task)

        # Format prompt
        prompt = format_prompt(task, user_input)

        # Auto per-personality config
        gen_cfg = config.GENERATION_CONFIGS.get(task, {})
        print("Generating...", flush=True)
        response = generate(
            prompt, model, tokenizer,
            temperature = gen_cfg.get("temperature", config.TEMPERATURE),
            top_k       = gen_cfg.get("top_k",       config.TOP_K),
            top_p       = gen_cfg.get("top_p",       config.TOP_P),
            max_tokens  = gen_cfg.get("max_tokens",  config.MAX_NEW_TOKENS),
            rep_penalty = gen_cfg.get("rep_penalty", 1.1),
        )

        # Print response
        print_response(task, response)


if __name__ == "__main__":
    main()
