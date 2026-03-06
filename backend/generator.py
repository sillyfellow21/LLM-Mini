import torch, sys, os
sys.path.insert(0, os.path.expanduser("~/MiniLLM"))

TASK_KEYWORDS = {
    "farmer": ["crop","farm","soil","plant","wheat","rice","fertilizer","irrigation","pest","harvest","seed","leaf","leaves","disease","agriculture","sowing","field"],
    "story":  ["story","tale","once","fiction","write a story","tell me a story","narrative","adventure","character"],
    "poetry": ["poem","poetry","verse","rhyme","write a poem","haiku","sonnet","stanza","lyric"],
    "qa":     ["what","who","when","where","why","how","explain","define","meaning","capital","history","science"],
}

# ── Synced to chat.py CLI GENERATION_CONFIGS ─────────────────
GEN_CONFIG = {
    "farmer":  {"max_new_tokens": 200, "temperature": 0.7,  "top_k": 40, "top_p": 0.9,  "rep_penalty": 1.3},
    "story":   {"max_new_tokens": 300, "temperature": 0.85, "top_k": 50, "top_p": 0.92, "rep_penalty": 1.2},
    "poetry":  {"max_new_tokens": 150, "temperature": 0.9,  "top_k": 60, "top_p": 0.95, "rep_penalty": 1.1},
    "qa":      {"max_new_tokens": 200, "temperature": 0.6,  "top_k": 30, "top_p": 0.85, "rep_penalty": 1.3},
}

END_TOKENS = ["### END", "###", "<|endoftext|>"]


def detect_task(text):
    text_lower = text.lower()
    scores = {t: sum(1 for k in kws if k in text_lower) for t, kws in TASK_KEYWORDS.items()}
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else "qa"


def generate_stream(prompt, model, tokenizer, task):
    sys.path.insert(0, os.path.expanduser("~/MiniLLM"))
    import config

    cfg    = GEN_CONFIG.get(task, GEN_CONFIG["qa"])
    device = next(model.parameters()).device
    ids    = tokenizer.encode(prompt, return_tensors="pt").to(device)

    if ids.shape[1] > config.CONTEXT_WINDOW - cfg["max_new_tokens"]:
        ids = ids[:, -(config.CONTEXT_WINDOW - cfg["max_new_tokens"]):]

    generated = []
    past_ids  = ids.clone()

    with torch.no_grad():
        for _ in range(cfg["max_new_tokens"]):
            with torch.autocast(
                device_type=device.type if hasattr(device, "type") else "cuda",
                dtype=torch.float16, enabled=True
            ):
                # ✅ FIX 1: Unpack (logits, loss) tuple — matches chat.py
                logits, _ = model(past_ids[:, -config.CONTEXT_WINDOW:])

            logits = logits[:, -1, :] / cfg["temperature"]

            # ── Repetition penalty ─────────────────────────────
            for tid in set(generated[-30:]):
                logits[0, tid] /= cfg["rep_penalty"]

            # ── Top-k filtering ────────────────────────────────
            top_k = cfg["top_k"]
            if top_k > 0:
                vals, _ = torch.topk(logits, top_k)
                logits[logits < vals[:, -1:]] = float("-inf")

            # ✅ FIX 2: Top-p (nucleus) filtering — matches chat.py
            top_p = cfg["top_p"]
            if top_p < 1.0:
                sorted_logits, sorted_idx = torch.sort(logits, descending=True)
                cum_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                remove = cum_probs - torch.softmax(sorted_logits, dim=-1) > top_p
                sorted_logits[remove] = float("-inf")
                logits = torch.zeros_like(logits).scatter_(1, sorted_idx, sorted_logits)

            probs  = torch.softmax(logits, dim=-1)
            next_t = torch.multinomial(probs, 1)
            tid    = next_t.item()

            if tid == tokenizer.eos_token_id:
                break

            generated.append(tid)
            token_str = tokenizer.decode([tid], skip_special_tokens=True)

            # ── End marker check ───────────────────────────────
            decoded_so_far = tokenizer.decode(generated, skip_special_tokens=True)
            stop = False
            for end in END_TOKENS:
                if end in decoded_so_far:
                    stop = True
                    break
            if stop:
                break

            if token_str:
                yield token_str

            past_ids = torch.cat([past_ids, next_t], dim=1)
            if past_ids.shape[1] > config.CONTEXT_WINDOW:
                past_ids = past_ids[:, -config.CONTEXT_WINDOW:]