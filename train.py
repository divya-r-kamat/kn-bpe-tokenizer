# ============================================================
# FILE: train.py
# ============================================================
"""
Training script for Kannada BPE Tokenizer.
Usage: python train.py
"""
from datasets import load_dataset
from tokenizer import KannadaBPETokenizer
import os


def load_training_data(filepath: str) -> str:
    """Load training data from file."""
    with open(filepath, 'r', encoding='utf-8') as f:
        return f.read()


def main():
    # Create model directory if it doesn't exist
    os.makedirs('model', exist_ok=True)
    os.makedirs('data', exist_ok=True)

    # This has translations including literary content
    dataset = load_dataset("ai4bharat/samanantar", "kn", split="train", streaming=True)

    texts = []
    for i, example in enumerate(dataset):
        if i >= 10000:
            break
        texts.append(example['tgt'])  # Kannada side
        if (i + 1) % 1000 == 0:
            print(f"Loaded {i + 1} texts...")

    full_text = '\n\n'.join(texts)
    with open('data/kannada_samanantar.txt', 'w', encoding='utf-8') as f:
        f.write(full_text)

    print(f"Saved to data/kannada_samanantar.txt")
    
    # Initialize tokenizer
    tokenizer = KannadaBPETokenizer()
    
    with open('data/kannada_samanantar.txt', 'r', encoding='utf-8') as f:
        text = f.read()
    
    # Train tokenizer
    vocab_size = 5000  # Adjust as needed
    print(f"Training tokenizer with vocab size: {vocab_size}")
    tokenizer.train(text, vocab_size=vocab_size)
    
    # Save vocabulary
    vocab_path = 'model/vocab.json'
    tokenizer.save_vocab(vocab_path)

if __name__ == "__main__":
    main()
