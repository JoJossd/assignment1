#!/usr/bin/env python3
#
# CS336 – Training loop (functional) using adapters.run_transformer_lm
#
# This version trains a Transformer LM by calling the functional forward
# defined in adapters.py: run_transformer_lm(...). All trainable parameters
# are stored in a plain Python dict mapping str -> nn.Parameter named `weights`.
# Autograd flows through these Parameters normally, and the optimizer updates them.
#
# We also reuse other helpers from adapters.py:
#   - run_get_batch                (np.memmap-friendly batch sampling)
#   - run_cross_entropy            (CE loss)
#   - run_get_lr_cosine_schedule   (warmup + cosine LR)
#   - run_gradient_clipping        (global-norm clip)
#   - get_adamw_cls                (AdamW implementation)
#   - run_save_checkpoint          (checkpointing)
#   - run_load_checkpoint          (checkpoint loading)
#
# Optional Weights & Biases logging is supported with --wandb.
#
# Example:
#

"""
cd /home/jojo/workspace/assignment1-basics

uv run python ./cs336_run/train_model.py \
  --train ./bpe_output/TinyStoriesV2-GPT4/train.bin \
  --val ./bpe_output/TinyStoriesV2-GPT4/valid.bin \
  --vocab-size 10000 \
  --context-length 256 \
  --d-model 384 \
  --n-layers 8 \
  --n-heads 6 \
  --d-ff 1536 \
  --lr 3e-4 \
  --min-lr 3e-5 \
  --batch-size 16 \
  --max-iters 2000 \
  --checkpoint ./ckpt.pt \
  --device cuda
"""
#

from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import torch
from torch import nn

# --- import helpers from the assignment files in the same directory ---
from tests.adapters import (
    run_get_batch,
    run_cross_entropy,
    run_get_lr_cosine_schedule,
    run_gradient_clipping,
    get_adamw_cls,
    run_save_checkpoint,
    run_load_checkpoint,
    run_transformer_lm,
)


def _load_memmap(path: str | Path):
    """Load a 1D token array as numpy memmap/array. Supports .npy or flat .bin (uint16/int32/int64)."""
    p = Path(path)
    if p.suffix.lower() == ".npy":
        arr = np.load(p, mmap_mode="r")
        if arr.ndim != 1:
            arr = arr.reshape(-1)
        return arr
    # try common integer dtypes for flat binaries
    for dtype in (np.uint8, np.uint16, np.int32, np.int64):
        try:
            arr = np.memmap(p, mode="r", dtype=dtype)
            if arr.size > 0:
                return arr
        except Exception:
            pass
    raise ValueError(f"Could not load dataset from {path}. Expected .npy or a flat binary of uint16/int32/int64.")


def make_parameter_dict(
    vocab_size: int, d_model: int, n_layers: int, n_heads: int, d_ff: int, device: torch.device, dtype: torch.dtype
) -> dict[str, nn.Parameter]:
    """Create a dict[str, nn.Parameter] whose keys/shapes match adapters.run_transformer_lm."""
    P: dict[str, nn.Parameter] = {}

    def p(shape, std=0.02):
        t = torch.empty(*shape, device=device, dtype=dtype)
        torch.nn.init.trunc_normal_(t, std=std)
        return torch.nn.Parameter(t)

    # token embedding + final norm + output head
    P["token_embeddings.weight"] = p((vocab_size, d_model))
    P["ln_final.weight"] = torch.nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))
    P["lm_head.weight"] = p((vocab_size, d_model))

    assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

    for layer in range(n_layers):
        pref = f"layers.{layer}."
        # attention projections (no bias)
        P[pref + "attn.q_proj.weight"] = p((d_model, d_model))
        P[pref + "attn.k_proj.weight"] = p((d_model, d_model))
        P[pref + "attn.v_proj.weight"] = p((d_model, d_model))
        P[pref + "attn.output_proj.weight"] = p((d_model, d_model))
        # norms
        P[pref + "ln1.weight"] = torch.nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))
        P[pref + "ln2.weight"] = torch.nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))
        # SwiGLU FFN (gated):
        #   expected shapes per cs336_basics.SwiGLU and tests.adapters.run_swiglu
        #   w1: (d_ff, d_model), w3: (d_ff, d_model), w2: (d_model, d_ff)
        P[pref + "ffn.w1.weight"] = p((d_ff, d_model))
        P[pref + "ffn.w3.weight"] = p((d_ff, d_model))
        P[pref + "ffn.w2.weight"] = p((d_model, d_ff))
    return P


class WeightsWrapper(nn.Module):
    """Lightweight wrapper to provide state_dict/load_state_dict/parameters for checkpointing.

    This enables saving/loading with helpers that expect a torch.nn.Module.
    """

    def __init__(self, mapping: dict[str, nn.Parameter]):
        super().__init__()
        self.mapping = mapping

    def state_dict(self, *args, **kwargs) -> dict[str, torch.Tensor]:
        # Return raw tensors for serialization
        return {k: v.detach().clone() for k, v in self.mapping.items()}

    def load_state_dict(self, state: dict[str, torch.Tensor], strict: bool = True):  # type: ignore[override]
        for k, t in state.items():
            if k in self.mapping:
                self.mapping[k].data.copy_(t)
        return None

    def parameters(self, recurse: bool = True):  # type: ignore[override]
        # Provide an iterator compatible with next(...)
        return iter(self.mapping.values())


def evaluate(args, weights, split_arr, model_kwargs, max_batches=50):
    if split_arr is None:
        return None
    losses = []
    with torch.no_grad():
        for _ in range(max_batches):
            x, y = run_get_batch(split_arr, args.batch_size, args.context_length, args.device)
            logits = run_transformer_lm(**model_kwargs, weights=dict(weights), in_indices=x)
            loss = run_cross_entropy(logits.view(-1, logits.size(-1)), y.reshape(-1))
            losses.append(loss.detach().float().item())
    return float(np.mean(losses)) if losses else None


def train(args):
    device = torch.device(args.device)
    torch.manual_seed(args.seed)

    train_data = _load_memmap(args.train)
    val_data = _load_memmap(args.val) if args.val else None

    # create Parameters that the functional model will consume
    dtype = torch.bfloat16 if args.bf16 else torch.float32
    weights = make_parameter_dict(
        vocab_size=args.vocab_size,
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        d_ff=args.d_ff,
        device=device,
        dtype=dtype,
    )

    # optimizer from adapters
    AdamW = get_adamw_cls()
    optimizer = AdamW(
        weights.values(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.999),
        eps=1e-8,
    )

    # try resume
    start_it = 0
    if args.checkpoint and Path(args.checkpoint).exists() and not args.restart:
        try:
            start_it = run_load_checkpoint(args.checkpoint, WeightsWrapper(weights), optimizer)
            print(f"Resumed from {args.checkpoint} at iteration {start_it}")
        except Exception as e:
            print(f"Could not load checkpoint: {e}")

    wb = None
    if args.wandb:
        try:
            import wandb

            wb = wandb
            wb.init(project=args.proj, name=(args.run_name or None), config=vars(args))
        except Exception as e:
            print(f"[wandb] disabled: {e}")

    _scaler_device = "cuda" if device.type == "cuda" else "cpu"
    scaler = torch.amp.GradScaler(_scaler_device, enabled=args.fp16)

    model_kwargs = dict(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        d_model=args.d_model,
        num_layers=args.n_layers,
        num_heads=args.n_heads,
        d_ff=args.d_ff,
        rope_theta=args.rope_theta,
    )

    it = start_it
    while it < args.max_iters:
        # batch
        x, y = run_get_batch(train_data, args.batch_size, args.context_length, args.device)

        # lr schedule
        lr = run_get_lr_cosine_schedule(it, args.lr, args.min_lr, args.warmup_iters, args.cosine_iters)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        # forward & loss
        if args.fp16:
            with torch.amp.autocast(_scaler_device):
                logits = run_transformer_lm(**model_kwargs, weights=dict(weights), in_indices=x)
                loss = run_cross_entropy(logits.view(-1, logits.size(-1)), y.reshape(-1))
        else:
            logits = run_transformer_lm(**model_kwargs, weights=dict(weights), in_indices=x)
            loss = run_cross_entropy(logits.view(-1, logits.size(-1)), y.reshape(-1))

        # backward
        optimizer.zero_grad(set_to_none=True)
        if args.fp16:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            run_gradient_clipping(weights.values(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            run_gradient_clipping(weights.values(), args.grad_clip)
            optimizer.step()

        it += 1

        # logging
        if it % args.log_interval == 0 or it == 1:
            item = loss.detach().float().item()
            if wb:
                wb.log({"it": it, "loss": item, "lr": lr}, step=it)
            print(f"it {it:6d} | loss {item:.4f} | lr {lr:.3e}")

        # eval
        if args.val and it % args.eval_interval == 0:
            val_loss = evaluate(args, weights, val_data, model_kwargs, max_batches=args.eval_batches)
            if val_loss is not None:
                if wb:
                    wb.log({"val_loss": val_loss}, step=it)
                print(f"eval @ it {it}: val_loss = {val_loss:.4f}")

        # checkpoint
        if args.checkpoint and (it % args.ckpt_interval == 0 or it == args.max_iters):
            try:
                run_save_checkpoint(WeightsWrapper(weights), optimizer, iteration=it, out=args.checkpoint)
                if wb:
                    wb.save(str(args.checkpoint))
                print(f"Saved checkpoint to {args.checkpoint}")
            except Exception as e:
                print(f"Failed to save checkpoint: {e}")

    # final eval
    if args.val:
        val_loss = evaluate(args, weights, val_data, model_kwargs, max_batches=args.eval_batches)
        if val_loss is not None:
            print(f"final val_loss = {val_loss:.4f}")
            if wb:
                wb.log({"val_loss_final": val_loss}, step=it)

    if wb:
        wb.finish()


def main():
    p = argparse.ArgumentParser(description="CS336 training loop (functional adapters.run_transformer_lm)")
    p.add_argument("--train", type=str, required=True, help="path to training tokens (.bin or .npy)")
    p.add_argument("--val", type=str, default="", help="path to validation tokens (.bin or .npy)")
    p.add_argument("--vocab-size", type=int, required=True)
    p.add_argument("--context-length", type=int, default=256)
    p.add_argument("--d-model", type=int, default=384)
    p.add_argument("--n-layers", type=int, default=6)
    p.add_argument("--n-heads", type=int, default=6)
    p.add_argument("--d-ff", type=int, default=1536)
    p.add_argument("--rope-theta", type=float, default=1e4)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--max-iters", type=int, default=2000)
    p.add_argument("--warmup-iters", type=int, default=200)
    p.add_argument("--cosine-iters", type=int, default=2000)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--min-lr", type=float, default=3e-5)
    p.add_argument("--weight-decay", type=float, default=0.1)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--eval-interval", type=int, default=200)
    p.add_argument("--eval-batches", type=int, default=50)
    p.add_argument("--log-interval", type=int, default=10)
    p.add_argument("--ckpt-interval", type=int, default=500)
    p.add_argument("--checkpoint", type=str, default="")
    p.add_argument("--restart", action="store_true", help="ignore existing checkpoint and start fresh")
    p.add_argument("--device", type=str, default=("cuda:0" if torch.cuda.is_available() else "cpu"))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--fp16", action="store_true")
    p.add_argument("--bf16", action="store_true")
    p.add_argument("--wandb", action="store_true")
    p.add_argument("--proj", type=str, default="cs336-basics")
    p.add_argument("--run-name", type=str, default="")
    args = p.parse_args()
    train(args)


if __name__ == "__main__":
    main()
