#!/usr/bin/env python3
#
# encode_corpus.py — CS336 A1 utility
# Encode raw text into a flat sequence of token IDs and save as .bin files.
#
# Reuses tokenizer utilities from tokenizer.py (and expects a trained BPE
# vocabulary + merges, e.g., GPT-2 format). Typical usage:
#

"""
cd /home/jojo/workspace/assignment1-basics

uv run python ./cs336_run/encode_corpus.py \
    --input ./data/TinyStoriesV2-GPT4-valid.txt \
    --vocab ./bpe_output/vocab.json \
    --merges ./bpe_output/merges.txt \
    --out ./bpe_output/TinyStoriesV2-GPT4/valid.bin

uv run python ./cs336_run/encode_corpus.py \
    --input ./data/TinyStoriesV2-GPT4-train.txt \
    --vocab ./bpe_output/vocab.json \
    --merges ./bpe_output/merges.txt \
    --out ./bpe_output/TinyStoriesV2-GPT4/train.bin
"""

#
# Optionally split into train/val in one pass:
#
#   python encode_corpus.py \
#     --input ./data/tinystories/all.txt \
#     --vocab ./gpt2_vocab.json --merges ./gpt2_merges.txt \
#     --out-prefix ./data/tinystories/tokens \
#     --val-ratio 0.01
#
# This script chooses uint16 when possible (vocab_size <= 65535), else int32.
#
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path
from collections.abc import Iterable

import numpy as np

# Local imports from your assignment repo
import cs336_basics.tokenizer as tok_mod


def _resolve_inputs(inp: str) -> list[Path]:
    p = Path(inp)
    if p.is_dir():
        # all regular files (common case: *.txt)
        return sorted([q for q in p.rglob("*") if q.is_file()])
    # allow globbing patterns
    matches = [Path(m) for m in glob.glob(inp)]
    if matches:
        return sorted(matches)
    # single file path
    if p.exists() and p.is_file():
        return [p]
    raise FileNotFoundError(f"No files found from --input={inp}")


def _iter_text(paths: list[Path]) -> Iterable[str]:
    for path in paths:
        with open(path, "r", encoding="utf-8", newline="\n", errors="ignore") as f:
            yield from f


def main():
    ap = argparse.ArgumentParser(description="Encode raw text into token IDs and save to .bin (memmap-friendly)")
    grp_tok = ap.add_argument_group("Tokenizer (required)")
    grp_tok.add_argument("--vocab", type=str, required=True, help="path to vocab.json (GPT-2 style)")
    grp_tok.add_argument("--merges", type=str, required=True, help="path to merges.txt (pair merges)")

    grp_in = ap.add_argument_group("Input")
    grp_in.add_argument("--input", type=str, required=True, help="file, directory, or glob of text files")

    grp_out = ap.add_argument_group("Output")
    mx = grp_out.add_mutually_exclusive_group(required=True)
    mx.add_argument("--out", type=str, help="output .bin file path (single dataset)")
    mx.add_argument(
        "--out-prefix", type=str, help="prefix to write {prefix}.train.bin and {prefix}.val.bin when --val-ratio>0"
    )

    ap.add_argument(
        "--val-ratio",
        type=float,
        default=0.0,
        help="if >0, split the encoded stream into train/val by ratio (val at the end)",
    )

    ap.add_argument(
        "--special", type=str, nargs="*", default=None, help="optional special tokens list (e.g. <|bos|> <|eos|>)"
    )

    args = ap.parse_args()

    # Build tokenizer from disk using helpers in tokenizer.py
    tokenizer = tok_mod.get_tokenizer_from_files(
        args.vocab,
        args.merges,
        special_tokens=args.special,
    )

    # Encode all text lines
    paths = _resolve_inputs(args.input)
    print(f"[encode_corpus] encoding {len(paths)} file(s) ...")
    ids: list[int] = []

    # Stream line-by-line to avoid loading everything into RAM at once.
    # BPE encoding is local per line (no cross-line state), so this is safe.
    n_lines = 0
    for line in _iter_text(paths):
        n_lines += 1
        ids.extend(tokenizer.encode(line))

        # light progress print
        if n_lines % 100000 == 0:
            print(f"  processed {n_lines} lines, tokens so far: {len(ids)}")

    total = len(ids)
    if total == 0:
        raise RuntimeError("No tokens produced. Are your inputs empty?")

    # Q? int32 datatype is too large?
    # Choose dtype based on vocab size if available, default to int32 if unsure.
    # Many CS336 tokenizers expose tokenizer.vocab_size or tokenizer.n_vocab. Try best-effort.
    vocab_size = getattr(tokenizer, "vocab_size", None) or getattr(tokenizer, "n_vocab", None)
    if vocab_size is not None and vocab_size <= 65535:
        dtype = np.uint8
    else:
        dtype = np.uint8

    arr = np.array(ids, dtype=dtype)

    # Single dataset output
    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        arr.tofile(out_path)
        # write a small sidecar with dtype/vocab size for convenience
        meta = {
            "dtype": str(dtype),
            "count": int(arr.size),
            "vocab_size": int(vocab_size) if vocab_size is not None else None,
        }
        with open(out_path.with_suffix(out_path.suffix + ".meta.json"), "w", encoding="utf-8") as mf:
            json.dump(meta, mf)
        print(f"[encode_corpus] wrote {arr.size} tokens -> {out_path} ({dtype})")
        return

    # Train/val split output
    if args.out_prefix:
        if args.val_ratio <= 0.0 or args.val_ratio >= 1.0:
            raise ValueError("--out-prefix requires 0 < --val-ratio < 1.0")
        split = int(round((1.0 - args.val_ratio) * arr.size))
        train_arr = arr[:split]
        val_arr = arr[split:]

        out_train = Path(args.out_prefix + ".train.bin")
        out_val = Path(args.out_prefix + ".val.bin")
        out_train.parent.mkdir(parents=True, exist_ok=True)
        train_arr.tofile(out_train)
        val_arr.tofile(out_val)

        meta = {
            "dtype": str(dtype),
            "count_total": int(arr.size),
            "count_train": int(train_arr.size),
            "count_val": int(val_arr.size),
            "val_ratio": float(args.val_ratio),
            "vocab_size": int(vocab_size) if vocab_size is not None else None,
        }
        with open(out_train.with_suffix(out_train.suffix + ".meta.json"), "w", encoding="utf-8") as mf:
            json.dump(meta, mf, indent=2)
        print(
            f"[encode_corpus] wrote train: {train_arr.size} -> {out_train}; val: {val_arr.size} -> {out_val} ({dtype})"
        )
        return


if __name__ == "__main__":
    main()
