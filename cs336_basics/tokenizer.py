import json
import regex as re  # 'regex' package (supports \p{} etc.)
import ast
from pathlib import Path
from .bpe import PAT
from collections.abc import Iterable


"""
Mental model diagram:
Preserve special tokens intact; route non-special spans through byte-BPE.

[text] --split by special-->  [chunk | SPECIAL | chunk | ...]
   chunk --PAT--> [piece][piece]...
     piece -> UTF-8 bytes -> [b'a', b'b', b' ', b'a', b'b']
        BPE loop (use rank) -> [b'ab', b' ', b'ab']
           map bytes -> ids -> [42, 3, 42]
SPECIAL -> direct lookup -> id_s
Concatenate all ids → final token id stream

"""


def _build_special_regex(special_tokens: list[str] | None) -> re.Pattern:
    if not special_tokens:
        return re.compile(r"(?!x)x")  # never matches
    parts = [re.escape(tok) for tok in sorted(special_tokens, key=len, reverse=True)]
    return re.compile(f"({'|'.join(parts)})")


# =============================================================================
# Tokenizer
# Runtime byte-level BPE tokenizer.
# Holds vocab (id->bytes), merges (ordered), and derived structures:
# - bytes_to_id: inverse vocab for fast lookup
# - rank: merge pair -> priority index (lower is earlier -> higher priority)
# - compiled regexes: specials + GPT-2 PAT for pretokenization
# =============================================================================
class Tokenizer:
    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None,
    ) -> None:
        """
        vocab: dict[int, bytes]
        merges: list[tuple[bytes, bytes]]
        special_tokens: list[str] | None
        """
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens or []

        # Build inverse vocab
        self._byte2tokenid = {b: i for i, b in self.vocab.items()}

        # Add special tokens to vocab if missing
        for tok in self.special_tokens:
            b = tok.encode("utf-8")
            if b not in self._byte2tokenid:
                new_id = max(self.vocab.keys(), default=-1) + 1
                self.vocab[new_id] = b
                self._byte2tokenid[b] = new_id

        # Merge ranks for fast lookup (lower rank = earlier in merges file = higher priority)
        self._rank = {pair: r for r, pair in enumerate(self.merges)}

        # Regex for finding special tokens
        self._special_re = _build_special_regex(self.special_tokens)
        # Regex for GPT-2 style pretokenization
        self._pat_re = re.compile(PAT)

    @classmethod
    def from_files(
        cls,
        vocab_filepath: str,
        merges_filepath: str,
        special_tokens: list[str] | None = None,
    ) -> "Tokenizer":
        vocab = _load_vocab_file(vocab_filepath)
        merges = _load_merges_file(merges_filepath)
        return cls(vocab, merges, special_tokens)

    def encode(self, text: str) -> list[int]:
        if not text:
            return []

        ids = []
        if self.special_tokens:
            # Preserve special tokens intact; route non-special spans through byte-BPE.
            # Note: _special_re is compiled with a capturing group, so split() keeps the delimiters
            for piece in self._special_re.split(text):
                if not piece:
                    continue
                if piece in self.special_tokens:
                    ids.append(self._byte2tokenid[piece.encode("utf-8")])
                else:
                    ids.extend(self._encode_plain(piece))
        else:
            ids.extend(self._encode_plain(text))
        # Final list of token IDs.
        return ids

    def encode_iterable(self, iterable: Iterable[str]) -> Iterable[int]:
        for chunk in iterable:
            yield from self.encode(chunk)

    def decode(self, ids):
        b = b"".join(self.vocab[i] for i in ids)
        return b.decode("utf-8", errors="replace")

    # ---------- internals ----------

    def _get_pairs(self, seq) -> set[tuple[bytes, bytes]]:
        return {(seq[i], seq[i + 1]) for i in range(len(seq) - 1)}

    def _encode_plain(self, text: str) -> list[int]:
        if not text:
            return []

        out_ids = []

        # Iterate GPT-2 pretokenization spans and apply BPE within each span independently
        for match in self._pat_re.finditer(text):
            piece = match.group(0)
            b = piece.encode("utf-8")
            if not b:
                continue

            # tokens is a list of one-byte bytes objects, each representing a single byte from the UTF-8 encoded piece.
            tokens = [bytes([bt]) for bt in b]

            if not self._rank:
                out_ids.extend(self._byte2tokenid[t] for t in tokens)
                continue

            pairs = self._get_pairs(tokens)
            # Greedy BPE merge loop on adjacent pairs by rank.
            while True:
                best_pair, best_rank = None, None
                for p in pairs:
                    r = self._rank.get(p)
                    if r is not None and (best_rank is None or r < best_rank):
                        best_pair, best_rank = p, r
                if best_pair is None:
                    break

                a, b = best_pair
                merged = a + b
                new_tokens = []
                i = 0
                while i < len(tokens):
                    if i < len(tokens) - 1 and tokens[i] == a and tokens[i + 1] == b:
                        new_tokens.append(merged)
                        i += 2
                    else:
                        new_tokens.append(tokens[i])
                        i += 1
                tokens = new_tokens
                if len(tokens) == 1:
                    break
                pairs = self._get_pairs(tokens)

            out_ids.extend(self._byte2tokenid[tok] for tok in tokens)

        return out_ids


# ---------- helpers ----------


# =============================================================================
# _load_vocab_file
# Helper: read a JSON vocab mapping (id->byte-string or similar)
# and normalize into id->bytes for internal use.
# =============================================================================
def _load_vocab_file(path: str) -> dict[int, bytes]:
    s = Path(path).read_text(encoding="utf-8").strip()
    out = {}

    def put(k, v):
        k = int(k)
        if isinstance(v, str):
            try:
                maybe = ast.literal_eval(v.strip())
                if isinstance(maybe, (bytes, bytearray)):
                    v_bytes = bytes(maybe)
                elif isinstance(maybe, list):
                    v_bytes = bytes(maybe)
                else:
                    v_bytes = v.encode("utf-8")
            except Exception:
                v_bytes = v.encode("utf-8")
            out[k] = v_bytes
        elif isinstance(v, list):
            out[k] = bytes(v)
        elif isinstance(v, (bytes, bytearray)):
            out[k] = bytes(v)
        else:
            raise ValueError(f"Unsupported vocab value: {v}")

    try:
        data = json.loads(s)
        if isinstance(data, dict):
            for k, v in data.items():
                put(k, v)
            return out
        if isinstance(data, list):
            for item in data:
                put(item[0], item[1])
            return out
    except Exception:
        pass

    for ln in s.splitlines():
        if not ln.strip() or ln.strip().startswith("#"):
            continue
        k, v = ln.split("\t")
        put(k.strip(), v.strip())
    return out


def _parse_token_literal(x: str) -> bytes:
    x = x.strip()
    try:
        lit = ast.literal_eval(x)
        if isinstance(lit, (bytes, bytearray)):
            return bytes(lit)
        if isinstance(lit, list):
            return bytes(lit)
        if isinstance(lit, str):
            return lit.encode("utf-8")
    except Exception:
        pass
    return x.encode("utf-8")


# =============================================================================
# _load_merges_file
# Helper: read merges file lines like 'ab cd' and return list[(b"ab", b"cd")].
# The order of this list defines merge priority (earlier -> higher priority).
# =============================================================================
def _load_merges_file(path: str) -> list[tuple[bytes, bytes]]:
    s = Path(path).read_text(encoding="utf-8").strip()
    try:
        data = json.loads(s)
        if isinstance(data, list):
            merges = []
            for item in data:
                a, b = item
                if isinstance(a, list):
                    a = bytes(a)
                elif isinstance(a, str):
                    a = a.encode("utf-8")
                if isinstance(b, list):
                    b = bytes(b)
                elif isinstance(b, str):
                    b = b.encode("utf-8")
                merges.append((a, b))
            return merges
    except Exception:
        pass

    merges = []
    for ln in s.splitlines():
        if not ln.strip() or ln.startswith("#"):
            continue
        parts = re.split(r"\s+", ln)
        a = _parse_token_literal(parts[0])
        b = _parse_token_literal(parts[1])
        merges.append((a, b))
    return merges
