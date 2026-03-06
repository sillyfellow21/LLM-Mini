import torch, sys, os
sys.path.insert(0, os.path.expanduser("~/MiniLLM"))
from transformers import GPT2Tokenizer
import config
from model.gpt import MiniLLM

CHECKPOINTS = os.path.expanduser("~/MiniLLM/checkpoints")
TASKS       = ["farmer", "story", "poetry", "qa"]
_models     = {}
_tokenizer  = None

def get_tokenizer():
    global _tokenizer
    if _tokenizer is None:
        print("[Loader] Loading tokenizer...")
        _tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        _tokenizer.pad_token = _tokenizer.eos_token
    return _tokenizer

def get_model(task):
    if task not in _models:
        path = os.path.join(CHECKPOINTS, f"{task}_best.pt")
        print(f"[Loader] Loading {task} model...")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        m = MiniLLM(
            embed_dim   = config.EMBED_DIM,
            num_heads   = config.NUM_HEADS,
            num_layers  = config.NUM_LAYERS,
            ffn_dim     = config.FFN_DIM,
            max_seq_len = config.MAX_SEQ_LEN,
            vocab_size  = config.VOCAB_SIZE,
        ).to(device)
        ck = torch.load(path, map_location=device)
        m.load_state_dict(ck["model"] if "model" in ck else ck)
        m.eval()
        _models[task] = m
        print(f"[Loader] {task} ready on {device}")
    return _models[task]

def preload_all():
    for t in TASKS:
        try: get_model(t)
        except Exception as e: print(f"[Loader] {t} failed: {e}")

def get_device():
    return "cuda" if torch.cuda.is_available() else "cpu"
