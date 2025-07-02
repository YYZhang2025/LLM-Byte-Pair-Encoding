from base_tokenizer import Tokenizer
from utils import get_stats, merge
from gpt_patterns import GPT2_SPLIT_PATTERN, GPT4_SPLIT_PATTERN


import regex as re


class BPETokenizer(Tokenizer):
    def __init__(self, pattern=GPT4_SPLIT_PATTERN):
        super().__init__()

        self.pattern = pattern
        self.compiled_pattern = re.compile(self.pattern)

        # Store Special Tokens
        self.special_tokens = {}
        self.inverse_special_tokens = {}

    def train(self, text, vocab_size, verbose=False):
        assert vocab_size >= 256, "Vocabulary size must be at least 256."

        num_merges = vocab_size - 256

        text_chunks = re.findall(self.compiled_pattern, text)

        ids = [list(ch.encode("utf-8")) for ch in text_chunks]

        merges = {}
        vocab = {idx: bytes([idx]) for idx in range(256)}
        for i in range(num_merges):
            stats = {}
            for chunk_ids in ids:
                get_stats(chunk_ids, stats)

            if not stats:
                if verbose:
                    print(f"No more pairs to merge after {i} merges.")
                break

            pair = max(stats, key=stats.get)
            idx = 256 + i

            ids = [merge(chunk_ids, pair, idx) for chunk_ids in ids]

            merges[pair] = idx
            vocab[idx] = vocab[pair[0]] + vocab[pair[1]]

            if verbose:
                print(f"Merge {i + 1}/{num_merges}: {pair} -> {idx}")

        self.merges = merges
        self.vocab = vocab

    def register_special_tokens(self, special_tokens):
        # special_tokens is a dictionary of str -> int
        # example: {"<|endoftext|>": 100257}
        self.special_tokens = special_tokens
        self.inverse_special_tokens = {v: k for k, v in special_tokens.items()}

    def decode(self, ids):
        part_bytes = []
        print(self.special_tokens)
        for idx in ids:
            if idx in self.vocab:
                part_bytes.append(self.vocab[idx])
            elif idx in self.inverse_special_tokens:
                part_bytes.append(self.inverse_special_tokens[idx].encode("utf-8"))
            else:
                raise ValueError(f"Unknown token ID: {idx}")

        text_bytes = b"".join(part_bytes)
        text = text_bytes.decode("utf-8", errors="replace")
        return text

    def _encode_chunk(self, text_bytes):
        ids = list(text_bytes)

        while len(ids) >= 2:
            stats = get_stats(ids)
            pair = min(stats, key=lambda p: self.merges.get(p, float("inf")))

            if pair not in self.merges:
                break
            idx = self.merges[pair]
            ids = merge(ids, pair, idx)

        return ids

    def encode_ordinary(self, text):
        # TODO: Enable Multi-Threading
        text_chunks = re.findall(self.compiled_pattern, text)

        ids = []

        for chunk in text_chunks:
            chunk_bytes = chunk.encode("utf-8")
            chunk_ids = self._encode_chunk(list(chunk_bytes))
            ids.extend(chunk_ids)

        return ids

    def encode(self, text, allowed_special="none_raise"):
        special = None
        if allowed_special == "all":
            special = self.special_tokens
        elif allowed_special == "none":
            special = {}
        elif allowed_special == "none_raise":
            special = {}
            assert all(token not in text for token in self.special_tokens)
        elif isinstance(allowed_special, set):
            special = {
                k: v for k, v in self.special_tokens.items() if k in allowed_special
            }
        else:
            raise ValueError(f"allowed_special={allowed_special} not understood")
        if not special:
            # shortcut: if no special tokens, just use the ordinary encoding
            return self.encode_ordinary(text)

        special_pattern = "(" + "|".join(re.escape(k) for k in special) + ")"
        special_chunks = re.split(special_pattern, text)

        ids = []
        for part in special_chunks:
            if part in special:
                ids.append(special[part])
            else:
                ids.extend(self.encode_ordinary(part))
        return ids
