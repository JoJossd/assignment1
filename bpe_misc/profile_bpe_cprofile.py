#!/usr/bin/env python3
"""Profile BPE training using cProfile."""

import cProfile
import pstats
from pathlib import Path
from pstats import SortKey

from cs336_basics.bpe_optimized import train_bpe as train_bpe_optimized


def run_training():
    """Run BPE training."""
    input_path = Path("data/TinyStoriesV2-GPT4-train.txt")
    vocab_size = 10_000
    special_tokens = ["<|endoftext|>"]

    vocab, merges = train_bpe_optimized(
        input_path=input_path,
        vocab_size=vocab_size,
        special_tokens=special_tokens,
        use_parallel=True,
        n_workers=None,
    )

    return vocab, merges


def main():
    print("=" * 80)
    print("Profiling BPE Training with cProfile")
    print("=" * 80)
    print()

    # Create output directory
    output_dir = Path("bpe_output")
    output_dir.mkdir(exist_ok=True)

    # Profile the training
    profiler = cProfile.Profile()
    profiler.enable()

    vocab, merges = run_training()

    profiler.disable()

    # Save detailed stats to file
    output_file = output_dir / "bpe_profile_stats_optimized.txt"
    with open(output_file, "w") as f:
        stats = pstats.Stats(profiler, stream=f)

        f.write("=" * 80 + "\n")
        f.write("TOP 50 FUNCTIONS BY CUMULATIVE TIME\n")
        f.write("=" * 80 + "\n\n")
        stats.sort_stats(SortKey.CUMULATIVE)
        stats.print_stats(50)

        f.write("\n" + "=" * 80 + "\n")
        f.write("TOP 50 FUNCTIONS BY INTERNAL TIME\n")
        f.write("=" * 80 + "\n\n")
        stats.sort_stats(SortKey.TIME)
        stats.print_stats(50)

        f.write("\n" + "=" * 80 + "\n")
        f.write("TOP 50 FUNCTIONS BY CALL COUNT\n")
        f.write("=" * 80 + "\n\n")
        stats.sort_stats(SortKey.CALLS)
        stats.print_stats(50)

    print()
    print("=" * 80)
    print("Profiling Complete!")
    print("=" * 80)
    print(f"Detailed stats saved to: {output_file}")
    print()
    print("Top 20 functions by cumulative time:")
    print("-" * 80)

    # Print top 20 to console
    stats = pstats.Stats(profiler)
    stats.sort_stats(SortKey.CUMULATIVE)
    stats.print_stats(20)


if __name__ == "__main__":
    main()
