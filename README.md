# Custom Byte Pair Encoding (BPE) Tokenizer for Kannada

This project involves building a custom Byte Pair Encoding (BPE) tokenizer for the **Kannada language**.

### Dataset
The dataset used for this is [ai4bharat/samanantar](https://huggingface.co/datasets/ai4bharat/samanantar) This dataset provides a rich collection of parallel sentences in Indian languages, making it ideal for training language-specific tokenizers.

The Kannada portion of the `ai4bharat/samanantar` dataset is used:

```python
from datasets import load_dataset

dataset = load_dataset("ai4bharat/samanantar", "kn", split="train", streaming=True)
```
This stream-based loading allows us to efficiently train on large text corpora without loading everything into memory.

### Objective

- Language: Kannada (kn)
- Tokenizer Type: Byte Pair Encoding (BPE)
- Target Vocabulary Size: 5000
- Compression Ratio >=3.2

### Understaning Tokenization
When we interact with a large language model (LLM) like GPT, it feels like it's processing text directly — words, sentences, and meaning. But under the hood, LLMs don't actually process text as-is. They only understand numbers. Computers don't deal with text, words, or letters. They operate in numbers — 0s and 1s.

_How should we break text into manageable pieces (tokens) that the model can learn from?_

Before an LLM can make sense of "Hello, world!", it first needs to convert text into a sequence of numbers that can be fed into the model. This step is called tokenization.

For example, the sentence: **The cat sat on the mat.** might be tokenized in several ways:

- Word-level: These split text by spaces or punctuation, treating each word as a separate token.

    ["The", "cat", "sat", "on", "the", "mat", "."]
- Character-level: These break text into individual characters.

    ["T", "h", "e", " ", "c", "a", "t"," ", "s", "a", "t", " ", "o", "n", " ", "t", "h", "e", " ", "m", "a", "t","."]

- Subword-level: This is the sweet spot — instead of whole words or single characters, subword tokenizers split text into frequently occurring fragments.

    ["The", "cat", "s", "at", "on", "the", "mat", "."]

- Byte-level: Represent each character by its byte value.

Among these, subword tokenization offers the best balance — it efficiently handles both common and rare words. And the most widely used subword tokenizer in modern LLMs is **Byte Pair Encoding (BPE)**.

## Byte Pair Encoding (BPE)
The Byte Pair Encoding (BPE) algorithm didn't start in natural language processing at all — it began as a data compression technique.

It was first introduced in 1994 by Philip Gage in his paper _“A New Algorithm for Data Compression.”_
The original idea was simple but powerful: 

_Repeatedly replace the most frequent pair of bytes in a sequence with a new, unused byte._

Over time, this effectively compressed data by representing frequent patterns with shorter codes.

Years later, researchers realized this same principle could be adapted to text tokenization — compressing words into smaller, frequently occurring units. This gave rise to BPE tokenization, now a cornerstone in most modern LLM architectures.

Before diving into the Byte Pair Encoding (BPE) algorithm, it's important to understand the concept of **bytes**, as the "B" in BPE stands for **Byte**.

### What is a Byte?

A **byte** is a group of **8 bits**. Since each bit can be either `0` or `1`, a single byte can represent:
    2^8 = 256
This means a byte can encode **256 possible values**, ranging from `0` to `255`. Understanding this foundational concept is crucial because BPE operates at the byte level, especially in contexts like text encoding and compression.  

A BPE tokenizer typically uses these 256 byte values as its initial vocabulary — each representing a single-character token.

You can visualize this using OpenAI's tiktoken library:

    import tiktoken
    gpt2_tokenizer = tiktoken.get_encoding("gpt2")
    
    for i in range(300):
        decoded = gpt2_tokenizer.decode([i])
        print(f"{i}: {decoded}")

Output snippet:

    0: !
    1: "
    2: #
    ...
    255: �   # single-byte tokens up to here
    256:  t
    257:  a
    ...
    298: ent
    299:  n


Tokens 0–255 correspond to the 256 possible byte values, while tokens beyond that represent merged patterns learned through BPE training.

### BPE Training
When applied to text, the BPE algorithm follows these steps:
- Initialize Vocabulary: Start with all possible single characters (bytes).
- Count Pairs: Find the most frequent adjacent pair of symbols — for example, "t" + "h".
- Merge: Replace all occurrences of that pair with a new symbol "th".
- Repeat: Continue merging the most frequent pairs until the desired vocabulary size is reached (e.g., 5,000 tokens).
This process builds a hierarchy of symbols — from characters → subwords → complete words, capturing frequent linguistic patterns efficiently.

### The Role of Regex in Pre-Tokenization

Before the actual Byte Pair Encoding (BPE) algorithm kicks in, the text is first pre-tokenized — split into meaningful chunks such as words, numbers, punctuation, and whitespace.
This step ensures that BPE merges happen within logical boundaries (for example, it won’t merge a Kannada word with a punctuation mark).

In this project, we use a regular expression (regex) to handle Kannada text, English words, and symbols together. The regex pattern looks like this:

    self.pattern = re.compile(
        r"""[\u0C80-\u0CFF]+|        # Kannada characters
            [a-zA-Z]+|               # English words
            [0-9]+|                  # Numbers
            [^\s\w\u0C80-\u0CFF]+|   # Punctuation or symbols
            \s+                      # Whitespace
        """,
        re.VERBOSE
    )

Let’s break down what this pattern does:
- [\u0C80-\u0CFF]+ — captures sequences of Kannada characters (Unicode block for Kannada).
- [a-zA-Z]+ — matches English words.
- [0-9]+ — matches numbers.
- [^\s\w\u0C80-\u0CFF]+ — matches punctuation marks or symbols (anything not alphanumeric or Kannada).
- \s+ — captures spaces and other whitespace.

Only after this pre-tokenization step does the Byte Pair Encoding (BPE) algorithm begin merging frequent symbol pairs within these chunks — building up a vocabulary of subword units that balance compactness with linguistic richness.

# Project Structure:
    # kn-bpe-tokenizer/
    # ├── tokenizer.py           # Tokenizer class
    # ├── train.py              # Training script
    # ├── app.py                # Gradio app
    # ├── requirements.txt      # Dependencies
    # ├── model/
    # │   └── vocab.json        # Saved vocabulary
    # └── README.md
