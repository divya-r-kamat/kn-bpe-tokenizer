# ============================================================
# FILE: tokenizer.py
# ============================================================
import re
from collections import Counter
from typing import List
import json


class KannadaBPETokenizer:
    def __init__(self):
        self.vocab = {}           # Maps token_id to token_str
        self.inverse_vocab = {}   # Maps token_str to token_id
        self.bpe_merges = []      # Ordered list of merge operations
        self.bpe_ranks = {}       # Maps pair to merge rank/priority

        # Regex pattern for encoding
        self.pattern = re.compile(
            r"""[\u0C80-\u0CFF]+|        # Kannada characters
                [a-zA-Z]+|               # English words
                [0-9]+|                  # Numbers
                [^\s\w\u0C80-\u0CFF]+|   # Punctuation/symbols
                \s+                      # Whitespace
            """,
            re.VERBOSE
        )

    def train(self, text: str, vocab_size: int):
        """Train the BPE tokenizer from scratch."""
        unique_chars = sorted(set(text))
        self.vocab = {i: char for i, char in enumerate(unique_chars)}
        self.inverse_vocab = {char: i for i, char in self.vocab.items()}

        print(f"Initial vocab size (unique characters): {len(self.vocab)}")

        initial_token_count = len(text)
        token_ids = [self.inverse_vocab[c] for c in text]
        
        print(f"Training on {initial_token_count} characters")
        
        num_merges = vocab_size - len(self.vocab)
        for merge_idx in range(num_merges):
            pair_freqs = self._count_pairs(token_ids)
            if not pair_freqs:
                print(f"No more pairs to merge. Stopping at vocab size {len(self.vocab)}")
                break

            best_pair = max(pair_freqs.items(), key=lambda x: x[1])[0]
            new_id = len(self.vocab)
            merged_token = self.vocab[best_pair[0]] + self.vocab[best_pair[1]]
            self.vocab[new_id] = merged_token
            self.inverse_vocab[merged_token] = new_id
            self.bpe_merges.append(best_pair)
            self.bpe_ranks[best_pair] = merge_idx
            token_ids = self._merge_pair(token_ids, best_pair, new_id)

            if (merge_idx + 1) % 1000 == 0:
                print(f"Merged {merge_idx + 1}/{num_merges} pairs, vocab size: {len(self.vocab)}")

        final_token_count = len(token_ids)
        compression_ratio = initial_token_count / final_token_count
        
        print(f"\n{'='*60}")
        print(f"Training Complete!")
        print(f"{'='*60}")
        print(f"Final vocab size: {len(self.vocab)}")
        print(f"Original characters: {initial_token_count}")
        print(f"Final BPE tokens: {final_token_count}")
        print(f"Compression ratio: {compression_ratio:.2f}x")
        print(f"{'='*60}\n")

    def _count_pairs(self, token_ids: List[int]) -> Counter:
        """Count frequency of adjacent token pairs."""
        pairs = Counter()
        for i in range(len(token_ids) - 1):
            pairs[(token_ids[i], token_ids[i + 1])] += 1
        return pairs

    def _merge_pair(self, token_ids: List[int], pair: tuple, new_id: int) -> List[int]:
        """Replace all occurrences of pair with new_id."""
        result = []
        i = 0
        while i < len(token_ids):
            if i < len(token_ids) - 1 and (token_ids[i], token_ids[i + 1]) == pair:
                result.append(new_id)
                i += 2
            else:
                result.append(token_ids[i])
                i += 1
        return result

    def _apply_bpe(self, token_str: str) -> List[int]:
        """Apply BPE merges to a string token."""
        token_ids = []
        for char in token_str:
            if char in self.inverse_vocab:
                token_ids.append(self.inverse_vocab[char])
            else:
                continue
        
        if len(token_ids) <= 1:
            return token_ids

        while len(token_ids) > 1:
            min_rank = float('inf')
            min_pos = -1
            
            for i in range(len(token_ids) - 1):
                pair = (token_ids[i], token_ids[i + 1])
                if pair in self.bpe_ranks:
                    rank = self.bpe_ranks[pair]
                    if rank < min_rank:
                        min_rank = rank
                        min_pos = i
            
            if min_pos == -1:
                break
            
            pair = (token_ids[min_pos], token_ids[min_pos + 1])
            merged_token_str = self.vocab[pair[0]] + self.vocab[pair[1]]
            new_id = self.inverse_vocab[merged_token_str]
            token_ids = token_ids[:min_pos] + [new_id] + token_ids[min_pos + 2:]

        return token_ids

    def encode(self, text: str) -> List[int]:
        """Encode text into token IDs."""
        chunks = re.findall(self.pattern, text)
        token_ids = []
        for chunk in chunks:
            token_ids.extend(self._apply_bpe(chunk))
        return token_ids

    def decode(self, token_ids: List[int]) -> str:
        """Convert token IDs back to text."""
        result = []
        for token_id in token_ids:
            if token_id in self.vocab:
                result.append(self.vocab[token_id])
        return "".join(result)

    def save_vocab(self, filepath: str):
        """Save vocabulary and merge rules to JSON file."""
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump({
                'vocab': {str(k): v for k, v in self.vocab.items()},
                'bpe_merges': [[p[0], p[1]] for p in self.bpe_merges]
            }, f, ensure_ascii=False, indent=2)
        print(f"Saved vocabulary to {filepath}")

    def load_vocab(self, filepath: str):
        """Load vocabulary and merge rules from JSON file."""
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        self.vocab = {int(k): v for k, v in data['vocab'].items()}
        self.inverse_vocab = {v: k for k, v in self.vocab.items()}
        self.bpe_merges = [tuple(pair) for pair in data['bpe_merges']]
        self.bpe_ranks = {tuple(pair): idx for idx, pair in enumerate(self.bpe_merges)}
        
        print(f"Loaded vocabulary from {filepath} (size: {len(self.vocab)})")




