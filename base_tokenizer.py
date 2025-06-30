from abc import ABC

from utils import render_token


class Tokenizer(ABC):
    def __init__(self):
        self.merges = {}
        self.pattern = ""
        self.special_tokens = {}
        self.vocab = self._build_vocab()

    def train(self, text, vocab_size, verbose=False):
        raise NotImplementedError("Subclasses should implement this method.")

    def encode(self, text: str) -> list:
        raise NotImplementedError("Subclasses should implement this method.")

    def decode(self, ids: list) -> str:
        raise NotImplementedError("Subclasses should implement this method.")

    def _build_vocab(self):
        # Initial has 256 tokens, one for each byte value
        vocab = {idx: bytes([idx]) for idx in range(256)}

        for (p0, p1), idx in self.merges.items():
            vocab[idx] = vocab[p0] + vocab[p1]

        # Encode special tokens
        for special, idx in self.special_tokens.items():
            vocab[idx] = special.encode("utf-8")

        return vocab

    def save(self, file_prefix: str):
        """
        Save the tokenizer's state to a file.
        """
        model_file = f"{file_prefix}.model"

        with open(model_file, "w", encoding="utf-8") as f:
            f.write("minbpe v1\n")
            f.write(f"{self.pattern}\n")
            f.write(f"{len(self.special_tokens)}\n")

            for special, idx in self.special_tokens.items():
                f.write(f"{special} {idx}\n")

            for idx1, idx2 in self.merges.keys():
                f.write(f"{idx1} {idx2}\n")

        vocab_file = f"{file_prefix}.vocab"
        inverted_merges = {idx: pair for pair, idx in self.merges.items()}

        with open(vocab_file, "w", encoding="utf-8") as f:
            for idx, token in self.vocab.items():
                s = render_token(token)

                if idx in inverted_merges:
                    idx0, idx1 = inverted_merges[idx]
                    s0 = render_token(self.vocab[idx0])
                    s1 = render_token(self.vocab[idx1])
                    f.write(f"[{s0}][{s1}] -> [{s}] {idx}\n")
                else:
                    f.write(f"{s} {idx}\n")
        print(f"Tokenizer saved to {model_file} and {vocab_file}")

    def load(self, model_file: str):
        assert model_file.endswith(".model"), "Model file must end with .model"

        merges = {}
        special_tokens = {}
        idx = 256  # Start from 256 for new merges

        with open(model_file, "r", encoding="utf-8") as f:
            version = f.readline().strip()
            assert version == "minbpe v1", "Unsupported model version"

            self.pattern = f.readline().strip()

            num_special = int(f.readline().strip())

            for _ in range(num_special):
                line = f.readline().strip()
                special, idx_str = line.rsplit(" ", 1)
                special_tokens[special] = int(idx_str)

            # Read merges
            for line in f:
                p0, p1 = map(int, line.strip().split())
                merges[(p0, p1)] = idx
                idx += 1

        self.merges = merges
        self.special_tokens = special_tokens
        self.vocab = self._build_vocab()
