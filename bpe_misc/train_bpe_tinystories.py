#!/usr/bin/env python3
"""Train BPE tokenizer on TinyStories dataset."""

import json
import time
import psutil
import os
from pathlib import Path

from cs336_basics.bpe_optimized import train_bpe


def format_bytes(size):
    """Convert bytes to human-readable format."""
    for unit in ["B", "KB", "MB", "GB"]:
        if size < 1024.0:
            return f"{size:.2f} {unit}"
        size /= 1024.0
    return f"{size:.2f} TB"


def get_memory_usage():
    """Get current process memory usage in bytes."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss


def main():
    # Configuration
    input_path = Path("data/TinyStoriesV2-GPT4-train.txt")
    output_dir = Path("bpe_output")
    vocab_size = 10_000
    special_tokens = ["<|endoftext|>"]

    # Create output directory
    output_dir.mkdir(exist_ok=True)

    print("=" * 80)
    print("Training BPE Tokenizer on TinyStories")
    print("=" * 80)
    print(f"Input file: {input_path}")
    print(f"Vocabulary size: {vocab_size:,}")
    print(f"Special tokens: {special_tokens}")
    print()

    # Check if input file exists
    if not input_path.exists():
        print(f"Error: Input file {input_path} not found!")
        print("Please download the TinyStories dataset first using:")
        print("  mkdir -p data && cd data")
        print("  wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt")
        return

    # Get initial memory
    initial_memory = get_memory_usage()
    print(f"Initial memory usage: {format_bytes(initial_memory)}")
    print()

    # Train BPE with timing
    print("Starting BPE training...")
    print("Using multiprocessing for faster training...")
    print()

    start_time = time.time()

    # Train with parallel processing enabled
    vocab, merges = train_bpe(
        input_path=input_path,
        vocab_size=vocab_size,
        special_tokens=special_tokens,
        use_parallel=True,  # Enable parallel processing
        n_workers=None,  # Use default (cpu_count() // 2)
    )

    end_time = time.time()
    training_time = end_time - start_time

    # Get peak memory usage
    peak_memory = get_memory_usage()
    memory_used = peak_memory - initial_memory

    print()
    print("=" * 80)
    print("Training Complete!")
    print("=" * 80)
    print(f"Training time: {training_time:.2f} seconds ({training_time / 60:.2f} minutes)")
    print(f"Peak memory usage: {format_bytes(peak_memory)}")
    print(f"Additional memory used: {format_bytes(memory_used)}")
    print()

    # Analyze vocabulary
    print("Vocabulary Statistics:")
    print(f"  Total vocabulary size: {len(vocab):,}")
    print(f"  Number of merges: {len(merges):,}")
    print(f"  Base bytes: 256")
    print(f"  Merged tokens: {len(vocab) - 256 - len(special_tokens):,}")
    print(f"  Special tokens: {len(special_tokens)}")
    print()

    # Find longest token
    longest_token_id = None
    longest_token_bytes = b""
    longest_length = 0

    for token_id, token_bytes in vocab.items():
        if len(token_bytes) > longest_length:
            longest_length = len(token_bytes)
            longest_token_bytes = token_bytes
            longest_token_id = token_id

    print("Longest Token:")
    print(f"  Token ID: {longest_token_id}")
    print(f"  Length: {longest_length} bytes")
    print(f"  Bytes (hex): {longest_token_bytes.hex()}")
    try:
        decoded = longest_token_bytes.decode("utf-8", errors="replace")
        print(f"  Decoded: {repr(decoded)}")
        print(f"  Does it make sense? ", end="")
        # Check if it's reasonable (printable, not too many special chars)
        if longest_token_bytes in [sp.encode("utf-8") for sp in special_tokens]:
            print("Yes - it's a special token")
        elif all(32 <= b < 127 or b in [9, 10, 13] for b in longest_token_bytes):
            print("Yes - it's a common text pattern")
        else:
            print("Partially - contains some non-printable bytes")
    except:
        print(f"  (Cannot decode as UTF-8)")
    print()

    # Serialize to disk
    print("Serializing to disk...")

    # Save vocabulary as JSON (convert bytes to list of ints for JSON serialization)
    vocab_path = output_dir / "vocab.json"
    vocab_serializable = {int(k): list(v) for k, v in vocab.items()}
    with open(vocab_path, "w") as f:
        json.dump(vocab_serializable, f, indent=2)
    print(f"  Vocabulary saved to: {vocab_path}")

    # Save merges as text file (one merge per line, space-separated hex strings)
    merges_path = output_dir / "merges.txt"
    with open(merges_path, "w") as f:
        for left, right in merges:
            f.write(f"{left.hex()} {right.hex()}\n")
    print(f"  Merges saved to: {merges_path}")

    # Also save a human-readable summary
    summary_path = output_dir / "training_summary.txt"
    with open(summary_path, "w") as f:
        f.write("BPE Training Summary\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Input file: {input_path}\n")
        f.write(f"Vocabulary size: {vocab_size:,}\n")
        f.write(f"Special tokens: {special_tokens}\n\n")
        f.write(f"Training time: {training_time:.2f} seconds ({training_time / 60:.2f} minutes)\n")
        f.write(f"Peak memory: {format_bytes(peak_memory)}\n")
        f.write(f"Memory used: {format_bytes(memory_used)}\n\n")
        f.write(f"Vocabulary statistics:\n")
        f.write(f"  Total size: {len(vocab):,}\n")
        f.write(f"  Merges: {len(merges):,}\n")
        f.write(f"  Base bytes: 256\n")
        f.write(f"  Merged tokens: {len(vocab) - 256 - len(special_tokens):,}\n")
        f.write(f"  Special tokens: {len(special_tokens)}\n\n")
        f.write(f"Longest token:\n")
        f.write(f"  Token ID: {longest_token_id}\n")
        f.write(f"  Length: {longest_length} bytes\n")
        f.write(f"  Bytes (hex): {longest_token_bytes.hex()}\n")
        try:
            decoded = longest_token_bytes.decode("utf-8", errors="replace")
            f.write(f"  Decoded: {repr(decoded)}\n")
        except:
            f.write(f"  (Cannot decode as UTF-8)\n")
    print(f"  Summary saved to: {summary_path}")

    print()
    print("=" * 80)
    print("All done! Files saved to:", output_dir.absolute())
    print("=" * 80)


if __name__ == "__main__":
    main()
