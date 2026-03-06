# ============================================================
# train.py — MiniLLM Pretraining
# ============================================================
# Usage: python train.py
# ============================================================

import os
import sys
import math
import time
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

import config
from model.gpt import build_model
from data.dataset import get_dataloader


def get_scheduler(optimizer, warmup_steps, total_steps):
    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    return LambdaLR(optimizer, lr_lambda)


@torch.no_grad()
def evaluate(model, val_loader, max_batches=50):
    model.eval()
    total, count = 0.0, 0
    for i, (inp, tgt) in enumerate(val_loader):
        if i >= max_batches:
            break
        inp, tgt = inp.to(config.DEVICE), tgt.to(config.DEVICE)
        with torch.amp.autocast('cuda', enabled=config.USE_FP16):
            _, loss = model(inp, tgt)
        total += loss.item()
        count += 1
    model.train()
    return total / max(count, 1)


def save_checkpoint(model, optimizer, scheduler, scaler,
                    epoch, step, train_losses, val_losses, path):
    raw = model._orig_mod if hasattr(model, "_orig_mod") else model
    torch.save({
        "epoch":        epoch,
        "step":         step,
        "model":        raw.state_dict(),
        "optimizer":    optimizer.state_dict(),
        "scheduler":    scheduler.state_dict(),
        "scaler":       scaler.state_dict(),
        "train_losses": train_losses,
        "val_losses":   val_losses,
    }, path)
    print(f"  [Checkpoint] Saved → {path}")


def load_checkpoint(path, model, optimizer, scheduler, scaler):
    ckpt = torch.load(path, map_location=config.DEVICE, weights_only=False)
    raw  = model._orig_mod if hasattr(model, "_orig_mod") else model
    raw.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    scheduler.load_state_dict(ckpt["scheduler"])
    scaler.load_state_dict(ckpt["scaler"])
    print(f"[Resume] Epoch {ckpt['epoch']+1}, Step {ckpt['step']}, "
          f"Best val: {min(ckpt['val_losses']):.4f}")
    return ckpt["epoch"], ckpt["step"], ckpt["train_losses"], ckpt["val_losses"]


def train():
    print("=" * 55)
    print("  MiniLLM — Pretraining")
    print("=" * 55)

    train_loader = get_dataloader("train")
    val_loader   = get_dataloader("val", shuffle=False)

    total_steps  = (len(train_loader) // config.GRAD_ACCUM_STEPS) \
                   * config.NUM_EPOCHS
    warmup_steps = max(200, total_steps // 50)

    print(f"\n  Batches/epoch  : {len(train_loader):,}")
    print(f"  Optimizer steps: {total_steps:,}")
    print(f"  Warmup steps   : {warmup_steps:,}")
    print(f"  Grad accum     : {config.GRAD_ACCUM_STEPS}")
    print()

    model = build_model(config)
    model.train()

    try:
        model = torch.compile(model)
        print("[Train] torch.compile enabled")
    except Exception as e:
        print(f"[Train] torch.compile skipped: {e}")

    decay     = [p for n, p in model.named_parameters() if p.dim() >= 2]
    no_decay  = [p for n, p in model.named_parameters() if p.dim() < 2]

    optimizer = AdamW([
        {"params": decay,    "weight_decay": config.WEIGHT_DECAY},
        {"params": no_decay, "weight_decay": 0.0},
    ], lr=config.LEARNING_RATE, betas=(0.9, 0.95))

    scheduler = get_scheduler(optimizer, warmup_steps, total_steps)
    scaler    = torch.amp.GradScaler('cuda', enabled=config.USE_FP16)

    # Resume?
    start_epoch  = 0
    global_step  = 0
    train_losses = []
    val_losses   = []
    best_val     = float("inf")

    last_ckpt = os.path.join(config.CHECKPOINT_DIR, "last.pt")
    best_ckpt = os.path.join(config.CHECKPOINT_DIR, "best.pt")

    if os.path.exists(last_ckpt):
        ans = input("\n[Resume] Found last.pt. Resume? (y/n): ")
        if ans.strip().lower() == "y":
            start_epoch, global_step, train_losses, val_losses = \
                load_checkpoint(last_ckpt, model, optimizer, scheduler, scaler)
            start_epoch += 1
            best_val = min(val_losses) if val_losses else float("inf")

    # Training loop
    print("\n" + "─" * 55)
    print(f"Starting from epoch {start_epoch + 1} / {config.NUM_EPOCHS}")
    print("─" * 55 + "\n")

    for epoch in range(start_epoch, config.NUM_EPOCHS):
        epoch_start = time.time()
        epoch_loss  = 0.0
        epoch_steps = 0
        tokens_seen = 0

        optimizer.zero_grad()

        for step, (inp, tgt) in enumerate(train_loader):
            inp = inp.to(config.DEVICE, non_blocking=True)
            tgt = tgt.to(config.DEVICE, non_blocking=True)

            with torch.amp.autocast('cuda', enabled=config.USE_FP16):
                _, loss = model(inp, tgt)
                loss    = loss / config.GRAD_ACCUM_STEPS

            scaler.scale(loss).backward()
            tokens_seen += inp.numel()

            if (step + 1) % config.GRAD_ACCUM_STEPS == 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), config.GRAD_CLIP)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()

                loss_val     = loss.item() * config.GRAD_ACCUM_STEPS
                epoch_loss  += loss_val
                epoch_steps += 1
                global_step += 1

                if global_step % config.LOG_INTERVAL == 0:
                    avg     = epoch_loss / epoch_steps
                    lr      = scheduler.get_last_lr()[0]
                    elapsed = time.time() - epoch_start
                    tok_sec = tokens_seen / max(elapsed, 1)
                    print(f"  Epoch {epoch+1:02d} | Step {global_step:5d} | "
                          f"Loss {loss_val:.4f} | Avg {avg:.4f} | "
                          f"LR {lr:.2e} | {tok_sec/1000:.0f}k tok/s")

                if global_step % config.SAVE_INTERVAL == 0:
                    save_checkpoint(model, optimizer, scheduler, scaler,
                                    epoch, global_step,
                                    train_losses, val_losses, last_ckpt)

        # End of epoch
        avg_train = epoch_loss / max(1, epoch_steps)
        val_loss  = evaluate(model, val_loader)
        val_losses.append(val_loss)
        train_losses.append(avg_train)

        elapsed = time.time() - epoch_start
        print(f"\n{'═'*55}")
        print(f"  Epoch {epoch+1:02d} | "
              f"Train: {avg_train:.4f} | "
              f"Val: {val_loss:.4f} | "
              f"Time: {elapsed/60:.1f} min")
        print(f"{'═'*55}\n")

        if val_loss < best_val:
            best_val = val_loss
            save_checkpoint(model, optimizer, scheduler, scaler,
                            epoch, global_step,
                            train_losses, val_losses, best_ckpt)
            print(f"  ⭐ New best! Val Loss: {best_val:.4f}")

        save_checkpoint(model, optimizer, scheduler, scaler,
                        epoch, global_step,
                        train_losses, val_losses, last_ckpt)

    print("\n🎉 Pretraining complete!")
    print(f"  Best Val Loss : {best_val:.4f}")
    print(f"  Next step     : python finetune.py --task story")


if __name__ == "__main__":
    train()
