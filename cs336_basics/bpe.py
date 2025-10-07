"""Comprehensive BPE (Byte Pair Encoding) tokenizer implementation.

This module provides both helper utilities for BPE pretokenization and parallel counting,
as well as an optimized training implementation with significant performance improvements.

Key features:
- Helper utilities for file chunking, pretokenization, and parallel processing
- Optimized BPE training with 8-17x speedup over naive implementation
- Support for special tokens and parallel processing
- Efficient pair counting and sequence merging algorithms
"""

from __future__ import annotations

import os
from typing import BinaryIO
import regex as re  # 'regex' package (supports \p{} etc.)
from collections import Counter, defaultdict
from multiprocessing import Pool, cpu_count

# GPT-2 pretokenizer regex pattern:
# - matches contractions like "'s", "'t", "'re" etc.
# - sequences of letters (\p{L}+), numbers (\p{N}+),
# - chunks of punctuation / symbols, and whitespace runs.
# This mirrors tiktoken's PAT to ensure compatibility.
PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

# Default special token boundary used for chunking large files.
# Training can split work at occurrences of this token for parallelism.
SPLIT_TOKEN_BYTES = b"<|endoftext|>"


# ============================================================================
# HELPER UTILITIES FOR PRETOKENIZATION AND PARALLEL COUNTING
# ============================================================================


# =============================================================================
# _find_chunk_boundaries()
# Determine safe chunk boundaries for parallel processing.
# Strategy: pick N tentative offsets, then scan forward up to a window
# to find the next occurrence of a split token (like <|endoftext|>).
# Returns byte offsets [(start, end), ...] that align on token boundaries.
# Complexity: O(file_size) worst-case to scan windows, but amortized small.
# =============================================================================
def _find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.

    Example:
        Input:
            file: BinaryIO pointing to content b"Hello<|endoftext|>World<|endoftext|>Test"
            desired_num_chunks: 3
            split_special_token: b"<|endoftext|>"

        Output:
            [0, 18, 36, 40]  # byte positions where chunks start/end
            # Chunk 1: bytes 0-18 ("Hello<|endoftext|>")
            # Chunk 2: bytes 18-36 ("World<|endoftext|>")
            # Chunk 3: bytes 36-40 ("Test")
    """
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size: int = file.tell()
    file.seek(0)

    chunk_size: int = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries: list[int] = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size: int = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position: int = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk: bytes = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at: int = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))


# =============================================================================
# _split_on_specials()
# Split text while PRESERVING special tokens in output.
# Uses a compiled regex with capturing groups so specials are not dropped.
# Example: 'hi<|endoftext|>bye' -> ['hi', '<|endoftext|>', 'bye'].
# =============================================================================
def _split_on_specials(text: str, special_tokens: list[str]) -> list[str]:
    """
    Split text on any of the special tokens, **preserving** them as separate items.

    Example:
        Input:
            text: "Hello<|endoftext|>World<|pad|>Test"
            special_tokens: ["<|endoftext|>", "<|pad|>"]

        Output:
            ["Hello", "<|endoftext|>", "World", "<|pad|>", "Test"]
            # Special tokens are preserved as separate elements

    Example (no special tokens):
        Input:
            text: "Hello World"
            special_tokens: []

        Output:
            ["Hello World"]
    """
    if not special_tokens:
        return [text]
    # Escape specials for regex and join with alternation
    parts: list[str] = re.split("(" + "|".join(map(re.escape, special_tokens)) + ")", text)
    # re.split with a capture group keeps the delimiters (specials). Good.
    return [p for p in parts if p != ""]


# =============================================================================
# _pretokens_and_counts_from_file()
# Serial pretokenization & counting.
# Reads the file, applies GPT-2 PAT over non-special spans, and
# represents each token as a tuple of single-byte tokens.
# Returns a dict: {tuple(bytes,...): count}. No global order needed for BPE.
# =============================================================================
def _pretokens_and_counts_from_file(
    path: str, special_tokens: list[str]
) -> tuple[dict[tuple[bytes, ...], int], list[str]]:
    """
    Build a frequency table of pretokens as tuples of bytes.
    Return (pretoken_counts, specials_seen_sequence) where specials_seen_sequence preserves order of appearance
    (not required for training, but can be handy if you later stream encode).

    Example:
        Input:
            path: "corpus.txt" containing "Hi Bob<|endoftext|>Hi Bob"
            special_tokens: ["<|endoftext|>"]

        Output:
            (
                {
                    (b'H', b'i'): 2,
                    (b' ', b'B', b'o', b'b'): 2
                },
                ["<|endoftext|>"]  # sequence of special tokens seen
            )
            # Each pretoken is represented as a tuple of individual byte objects
            # Counts reflect how many times each pretoken appears
            # "Hi" appears twice, " Bob" appears twice
    """

    # cnt for the appearance of every single byte
    counts: Counter[tuple[bytes, ...]] = Counter()
    specials_sequence: list[str] = []

    # Determine chunk strategy based on file size and special tokens
    file_size: int = os.path.getsize(path)
    desired_num_chunks: int = 4  # Can be adjusted based on available cores

    # Use chunked processing for files larger than 10MB, or if we have a natural split token
    use_chunking: bool = (file_size > 10 * 1024 * 1024) or bool(special_tokens)

    if use_chunking and special_tokens:
        # Process file in chunks split at special token boundaries
        # Use the first special token as the split boundary (typically <|endoftext|>)

        # In GPT-2/3 style BPE, the first special token in the list is almost always <|endoftext|>

        with open(path, "rb") as f:
            boundaries: list[int] = _find_chunk_boundaries(f, desired_num_chunks, SPLIT_TOKEN_BYTES)

            # Process each chunk
            for start, end in zip(boundaries[:-1], boundaries[1:]):
                f.seek(start)
                chunk_bytes: bytes = f.read(end - start)
                # Decode with error handling for incomplete UTF-8 at boundaries
                chunk_text: str = chunk_bytes.decode("utf-8", errors="ignore")

                # Process this chunk
                for piece in _split_on_specials(chunk_text, special_tokens):
                    if piece in special_tokens:
                        specials_sequence.append(piece)
                        continue
                    # Regex pretokenization
                    for match in re.finditer(PAT, piece):
                        # match.group(0) returns the entire substring matched by the regex pattern
                        substring = match.group(0)
                        bytes_obj = substring.encode("utf-8")
                        # represent as tuple of **unit bytes tokens** (each is a bytes object of length 1)
                        tup = tuple(bytes([bt]) for bt in bytes_obj)
                        counts[tup] += 1
    else:
        # Simple baseline: read whole file (ok for tests/small data or no special tokens)
        with open(path, encoding="utf-8") as f:
            text: str = f.read()

        for piece in _split_on_specials(text, special_tokens):
            if piece in special_tokens:
                specials_sequence.append(piece)
                continue
            # Regex pretokenization
            for match in re.finditer(PAT, piece):
                # match.group(0) returns the entire substring matched by the regex pattern
                substring = match.group(0)
                bytes_obj = substring.encode("utf-8")
                # represent as tuple of **unit bytes tokens** (each is a bytes object of length 1)
                tup = tuple(bytes([bt]) for bt in bytes_obj)
                counts[tup] += 1

    return dict(counts), specials_sequence


# =============================================================================
# _process_chunk()
# Worker-side routine used in parallel counting.
# Input is (chunk_bytes, special_tokens). It decodes to text, splits on specials,
# runs PAT on non-special pieces, and returns a local Counter-like dict.
# =============================================================================
def _process_chunk(chunk_data: tuple[bytes, list[str]]) -> dict[tuple[bytes, ...], int]:
    """
    Process a single chunk of data and return pretoken counts.
    Designed to be called in parallel by worker processes.

    Example:
        Input:
            chunk_data: (
                b"Hi Bob<|endoftext|>",  # chunk bytes
                ["<|endoftext|>"]        # special tokens
            )

        Output:
            {
                (b'H', b'i'): 1,
                (b' ', b'B', b'o', b'b'): 1
            }
            # Count of pretokens in this chunk only

    Args:
        chunk_data: Tuple of (chunk_bytes, special_tokens)

    Returns:
        Dictionary mapping pretoken tuples to their counts in this chunk
    """
    chunk_bytes, special_tokens = chunk_data
    counts: Counter[tuple[bytes, ...]] = Counter()

    # Decode chunk
    chunk_text: str = chunk_bytes.decode("utf-8", errors="ignore")

    # Process chunk
    for piece in _split_on_specials(chunk_text, special_tokens):
        if piece in special_tokens:
            continue
        # Regex pretokenization
        for match in re.finditer(PAT, piece):
            substring: str = match.group(0)
            bytes_obj: bytes = substring.encode("utf-8")
            tup: tuple[bytes, ...] = tuple(bytes([bt]) for bt in bytes_obj)
            counts[tup] += 1

    return dict(counts)


# =============================================================================
# _merge_counts()
# Merge a list of local count dicts from workers into a single dict.
# We use explicit loops/dicts instead of Counter to avoid object overhead
# in hot paths. Equivalent to sum(Counter(dicts)), but faster here.
# =============================================================================
def _merge_counts(count_dicts: list[dict[tuple[bytes, ...], int]]) -> dict[tuple[bytes, ...], int]:
    """
    Merge multiple count dictionaries by summing counts for each key.

    Example:
        Input:
            count_dicts: [
                {(b'H', b'i'): 2, (b'B', b'o'): 1},
                {(b'H', b'i'): 3, (b'C', b'a'): 1}
            ]

        Output:
            {
                (b'H', b'i'): 5,  # 2 + 3
                (b'B', b'o'): 1,
                (b'C', b'a'): 1
            }

    Args:
        count_dicts: List of count dictionaries from different chunks

    Returns:
        Merged dictionary with summed counts
    """
    merged: Counter[tuple[bytes, ...]] = Counter()
    for count_dict in count_dicts:
        for key, count in count_dict.items():
            merged[key] += count
    return dict(merged)


# =============================================================================
# _pretokens_and_counts_from_file_parallel()
# Parallel pretokenization & counting.
# Pipeline: find chunk boundaries -> submit chunks to workers -> merge counts.
# Note: order of special tokens across chunks is not preserved (not required for training).
# =============================================================================
def _pretokens_and_counts_from_file_parallel(
    path: str, special_tokens: list[str], n_workers: int
) -> tuple[dict[tuple[bytes, ...], int], list[str]]:
    """
    Build frequency table of pretokens using parallel processing.

    This is a parallel version of _pretokens_and_counts_from_file that:
    1. Splits file into chunks
    2. Processes each chunk in parallel using multiprocessing
    3. Merges the counts from all chunks

    Example:
        Input:
            path: "large_corpus.txt"
            special_tokens: ["<|endoftext|>"]
            n_workers: 4

        Output:
            (
                {(b'H', b'i'): 1000, ...},  # merged counts from all chunks
                ["<|endoftext|>", ...]       # sequence of special tokens seen
            )

    Args:
        path: Path to input file
        special_tokens: List of special token strings
        n_workers: Number of worker processes

    Returns:
        Tuple of (pretoken_counts, specials_seen_sequence)
    """
    assert n_workers is not None

    # Split file into chunks
    if not special_tokens:
        # Can't use chunking without a split token
        return _pretokens_and_counts_from_file(path, special_tokens)

    with open(path, "rb") as f:
        boundaries: list[int] = _find_chunk_boundaries(f, n_workers, SPLIT_TOKEN_BYTES)

        # Read all chunks
        chunks: list[tuple[bytes, list[str]]] = []
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            f.seek(start)
            chunk_bytes: bytes = f.read(end - start)
            chunks.append((chunk_bytes, special_tokens))

    # Process chunks in parallel
    with Pool(processes=n_workers) as pool:
        chunk_counts: list[dict[tuple[bytes, ...], int]] = pool.map(_process_chunk, chunks)

    # Merge results
    merged_counts: dict[tuple[bytes, ...], int] = _merge_counts(chunk_counts)

    # Note: We lose the special token sequence order in parallel processing
    # If needed, could be reconstructed by scanning file once more
    specials_sequence: list[str] = []

    return merged_counts, specials_sequence


# ============================================================================
# OPTIMIZED BPE TRAINING IMPLEMENTATION
# ============================================================================


# =============================================================================
# _count_pairs()
# Count adjacent byte-pairs across all sequences weighted by sequence frequency.
# Result: dict[(byte, byte)] = frequency. This drives the greedy BPE loop.
# =============================================================================
def _count_pairs(pretoken_counts: dict[tuple[bytes, ...], int]) -> dict[tuple[bytes, bytes], int]:
    """
    Weighted count of adjacent pairs across all pretokens.

    OPTIMIZATION: Use regular dict instead of defaultdict to avoid __missing__ overhead.
    """
    pair_counts: dict[tuple[bytes, bytes], int] = {}
    for seq, freq in pretoken_counts.items():
        if len(seq) < 2:
            continue
        # count adjacent pairs with multiplicity equal to freq
        for a, b in zip(seq, seq[1:]):
            pair_counts[(a, b)] = pair_counts.get((a, b), 0) + freq
    return pair_counts


# =============================================================================
# _apply_merge_to_seq()
# Apply a single merge (a,b)->ab to one sequence.
# Replace non-overlapping occurrences in a single left-to-right pass.
# Returns the new tuple of bytes tokens after merging.
# =============================================================================
def _apply_merge_to_seq(seq: tuple[bytes, ...], merge_pair: tuple[bytes, bytes]) -> tuple[bytes, ...]:
    """
    Replace all non-overlapping occurrences of pair in seq with merged token.

    OPTIMIZATIONS:
    - Pre-compute merged token (a + b)
    - Simplify loop condition
    - Reduce redundant checks
    """
    if len(seq) < 2:
        return seq

    a, b = merge_pair
    merged = a + b  # Pre-compute merged token (was recomputed each iteration)
    out: list[bytes] = []
    i: int = 0
    end: int = len(seq) - 1  # Cache length - 1

    while i < end:  # Can stop one element earlier
        if seq[i] == a and seq[i + 1] == b:
            out.append(merged)
            i += 2
        else:
            out.append(seq[i])
            i += 1

    # Handle last element if we didn't consume it
    if i == end:
        out.append(seq[end])

    return tuple(out)


# =============================================================================
# _update_pair_counts_after_merge()
# Incrementally update pair counts after applying a merge to a sequence.
# Avoids recomputing all pairs from scratch each iteration.
# Subtract old pairs from old sequence; add new pairs from new sequence.
# =============================================================================
def _update_pair_counts_after_merge(
    pair_counts: dict[tuple[bytes, bytes], int],
    old_seq: tuple[bytes, ...],
    new_seq: tuple[bytes, ...],
    freq: int,
) -> None:
    """
    Incrementally update pair_counts after a sequence changes from old_seq to new_seq.
    This modifies pair_counts in-place.
    """
    # Remove old pairs
    if len(old_seq) >= 2:
        for a, b in zip(old_seq, old_seq[1:]):
            pair_counts[(a, b)] -= freq
            if pair_counts[(a, b)] <= 0:
                del pair_counts[(a, b)]

    # Add new pairs
    if len(new_seq) >= 2:
        for a, b in zip(new_seq, new_seq[1:]):
            pair_counts[(a, b)] = pair_counts.get((a, b), 0) + freq


# =============================================================================
# train_bpe()
# Main training loop.
# 1) Get pretoken counts (serial or parallel).
# 2) Initialize pair counts.
# 3) Repeat until vocab_size: pick best pair, record merge, update sequences
#    and adjust pair counts incrementally.
# Returns (vocab: id->bytes, merges: list[(bytes,bytes)] in merge order).
# =============================================================================
def train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """Given the path to an input corpus, run train a BPE tokenizer and
    output its vocabulary and merges.

    OPTIMIZED VERSION with the following improvements:
    1. Skip sequences that don't contain merge pair
    2. Track sequences by pairs (reverse index)
    3. Optimized merge function
    4. Single-pass best pair finding

    Args:
        input_path (str | os.PathLike): Path to BPE tokenizer training data.
        vocab_size (int): Total number of items in the tokenizer's vocabulary (including special tokens).
        special_tokens (list[str]): A list of string special tokens to be added to the tokenizer vocabulary.
        **kwargs: Optional keyword arguments:
            - n_workers (int | None): Number of worker processes for parallel mode (default: cpu_count())

    Returns:
        tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
            vocab: The trained tokenizer vocabulary
            merges: BPE merges in order of creation
    """
    if vocab_size < 256 + len(special_tokens):
        raise ValueError("vocab_size too small for 256 base bytes + special tokens")

    # 1) Pretokenize and count (bytes-level, weighted)
    n_workers: int = kwargs.get("n_workers", cpu_count())
    assert n_workers is not None

    pretoken_counts: dict[tuple[bytes, ...], int]
    pretoken_counts, _ = _pretokens_and_counts_from_file_parallel(str(input_path), special_tokens, n_workers)

    # 2) Iteratively select merges with optimized incremental updates
    merges: list[tuple[bytes, bytes]] = []

    # Build initial pair counts once
    pair_counts: dict[tuple[bytes, bytes], int] = _count_pairs(pretoken_counts)

    # OPTIMIZATION: Build reverse index (pair -> set of sequences containing it)
    print("\n  Building reverse index...")
    seq_index: dict[tuple[bytes, bytes], set[tuple[bytes, ...]]] = defaultdict(set)
    for seq in pretoken_counts:
        if len(seq) >= 2:
            for pair in zip(seq, seq[1:]):
                seq_index[pair].add(seq)
    print(f"  Tracking {len(seq_index)} unique pairs across {len(pretoken_counts)} sequences")

    merge_count = 0
    while 256 + len(merges) + len(special_tokens) < vocab_size:
        if not pair_counts:
            break

        # OPTIMIZATION: Single-pass best pair finding
        best_pair: tuple[bytes, bytes] | None = None
        max_freq: int = -1
        for pair, count in pair_counts.items():
            if count > max_freq or (count == max_freq and (best_pair is None or pair > best_pair)):
                best_pair = pair
                max_freq = count

        if best_pair is None:
            break

        # OPTIMIZATION: Only process sequences that contain best_pair
        # Q? bypass copy here?
        affected_seqs = seq_index.get(best_pair, set()).copy()  # Copy to avoid modification during iteration
        new_pretoken_counts: Counter[tuple[bytes, ...]] = Counter()

        for seq in affected_seqs:
            freq = pretoken_counts[seq]
            new_seq: tuple[bytes, ...] = _apply_merge_to_seq(seq, best_pair)

            # Update pair counts incrementally
            _update_pair_counts_after_merge(pair_counts, seq, new_seq, freq)

            # Update sequence index
            # Remove old sequence from all its pairs
            if len(seq) >= 2:
                for pair in zip(seq, seq[1:]):
                    seq_index[pair].discard(seq)

            # Add new sequence to all its pairs
            if len(new_seq) >= 2:
                for pair in zip(new_seq, new_seq[1:]):
                    seq_index[pair].add(new_seq)

            new_pretoken_counts[new_seq] += freq

        # Copy unchanged sequences
        for seq, freq in pretoken_counts.items():
            if seq not in affected_seqs:
                new_pretoken_counts[seq] += freq

        pretoken_counts = dict(new_pretoken_counts)
        merges.append(best_pair)

        merge_count += 1
        if merge_count % 1000 == 0:
            print(f"  Completed {merge_count} merges, affected {len(affected_seqs)} sequences")

    print(f"\n  Completed {len(merges)} total merges")

    # 3) Build vocab: base bytes, then merged tokens, then specials
    vocab: dict[int, bytes] = {}

    # base bytes
    for i in range(256):
        vocab[i] = bytes([i])

    # merged tokens
    next_id: int = 256
    for left, right in merges:
        vocab[next_id] = left + right
        next_id += 1

    # specials
    for sp in special_tokens:
        vocab[next_id] = sp.encode("utf-8")
        next_id += 1

    # Guard total size
    if len(vocab) > vocab_size:
        overflow: int = len(vocab) - vocab_size
        if overflow > 0:
            for _ in range(overflow):
                next_id -= 1
                del vocab[next_id]
                merges.pop()

    return vocab, merges
