# Custom Byte Pair Encoding (BPE) Tokenizer for Kannada Language

This project involves building a custom Byte Pair Encoding (BPE) tokenizer for the **Kannada language**.

## Dataset
The dataset used for this is [ai4bharat/samanantar](https://huggingface.co/datasets/ai4bharat/samanantar) This dataset provides a rich collection of parallel sentences in Indian languages, making it ideal for training language-specific tokenizers. Samanantar is the largest publicly available parallel corpora collection for Indic language: Assamese, Bengali, Gujarati, Hindi, Kannada, Malayalam, Marathi, Oriya, Punjabi, Tamil, Telugu. The corpus has 49.6M sentence pairs between English to Indian Languages.

The Kannada portion of the `ai4bharat/samanantar` dataset is used, text corpus has 10,000 Kannada sentences:

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

## Project Structure:
    kn-bpe-tokenizer/
    ├── tokenizer.py          # Core tokenizer implementation
    ├── train.py              # Training script
    ├── app.py                # Gradio web interface
    ├── requirements.txt      # Python dependencies
    ├── Dockerfile            # Docker configuration
    ├── docker-compose.yml    # Docker Compose configuration
    ├── .dockerignore         # Docker ignore file
    ├── model/
    │   └── vocab.json        # Trained vocabulary (generated)
    └── README.md             # This file


## Training Log

    Training tokenizer with vocab size: 5000
    Initial vocab size (unique characters): 195
    Training on 576848 characters
    Merged 1000/4805 pairs, vocab size: 1195
    Merged 2000/4805 pairs, vocab size: 2195
    Merged 3000/4805 pairs, vocab size: 3195
    Merged 4000/4805 pairs, vocab size: 4195
    
    ============================================================
    Training Complete!
    ============================================================
    Final vocab size: 5000
    Original characters: 576848
    Final BPE tokens: 153632
    Compression ratio: 3.75x
    ============================================================
    
    Saved vocabulary to model/vocab.json

### Initial Vocabulary

When training begins, the tokenizer first builds a base vocabulary of all unique characters observed in the dataset.

    Initial vocab size (unique characters): 195
    Training on 576,848 characters

This includes Kannada letters, numerals, English alphabets (if any mixed-language text is present), spaces, and punctuation marks. These 195 symbols serve as the starting tokens for the Byte Pair Encoding process.

### Iterative Merging

Next, the tokenizer performs merge operations — repeatedly identifying the most frequent pair of symbols (or subwords) and combining them into a single new token.

    Merged 1000/4805 pairs, vocab size: 1195
    Merged 2000/4805 pairs, vocab size: 2195
    Merged 3000/4805 pairs, vocab size: 3195
    Merged 4000/4805 pairs, vocab size: 4195

Each merge represents one iteration where the most common neighboring token pair (like “ಕ” + “ಾ” → “ಕಾ”) is merged to form a new subword. Over time, these merges create larger, linguistically meaningful tokens — such as suffixes, root words, and common word fragments — improving both efficiency and linguistic representation.

### Final Summary

    ============================================================
    Training Complete!
    ============================================================
    Final vocab size: 5000
    Original characters: 576848
    Final BPE tokens: 153632
    Compression ratio: 3.75x
    ============================================================

- Final vocab size: The tokenizer now has 5000 learned subword tokens.
- Compression ratio: ~3.75× reduction means that the text can now be represented much more compactly without losing linguistic detail.
- Vocabulary file: The resulting token definitions and merge rules are saved in model/vocab.json.


### Try It Out on Hugging Face!

You can explore and interact with the Kannada BPE Tokenizer directly on Hugging Face Spaces:

👉 [dkamat/kn-bpe-tokenizer](https://huggingface.co/spaces/dkamat/kn-bpe-tokenizer)

<img width="1626" height="742" alt="image" src="https://github.com/user-attachments/assets/6317b009-1c58-4999-8ef5-01eaacb574ef" />

The app is built with Gradio, providing an intuitive web interface that lets you visualize how Kannada text is tokenized using Byte Pair Encoding (BPE) in real-time. It allows you to:
- Input any Kannada text.
- View tokens and token IDs — observe how subwords and characters combine.
- Color-coded token visualization — makes patterns and merges easy to understand.
- Inspect token table — explore the mapping between token IDs and text fragments.

## References & Citation
- Dataset : This project uses the Kannada-English parallel corpus from the AI4Bharat Samanantar Dataset - https://huggingface.co/datasets/ai4bharat/samanantar

        @article{10.1162/tacl_a_00452,
            author = {Ramesh, Gowtham and Doddapaneni, Sumanth and Bheemaraj, Aravinth and Jobanputra, Mayank and AK, Raghavan and Sharma, Ajitesh and Sahoo, Sujit and Diddee, Harshita and J, Mahalakshmi and Kakwani, Divyanshu and Kumar, Navneet and Pradeep, Aswin and Nagaraj, Srihari and Deepak, Kumar and Raghavan, Vivek and Kunchukuttan, Anoop and Kumar, Pratyush and Khapra, Mitesh Shantadevi},
            title = "{Samanantar: The Largest Publicly Available Parallel Corpora Collection for 11 Indic Languages}",
            journal = {Transactions of the Association for Computational Linguistics},
            volume = {10},
            pages = {145-162},
            year = {2022},
            month = {02},
            abstract = "{We present Samanantar, the largest publicly available parallel corpora collection for Indic languages. The collection contains a total of 49.7 million sentence pairs between English and 11 Indic languages (from two language families). Specifically, we compile 12.4 million sentence pairs from existing, publicly available parallel corpora, and additionally mine 37.4 million sentence pairs from the Web, resulting in a 4× increase. We mine the parallel sentences from the Web by combining many corpora, tools, and methods: (a) Web-crawled monolingual corpora, (b) document OCR for extracting sentences from scanned documents, (c) multilingual representation models for aligning sentences, and (d) approximate nearest neighbor search for searching in a large collection of sentences. Human evaluation of samples from the newly mined corpora validate the high quality of the parallel sentences across 11 languages. Further, we extract 83.4 million sentence
                            pairs between all 55 Indic language pairs from the English-centric parallel corpus using English as the pivot language. We trained multilingual NMT models spanning all these languages on Samanantar which outperform existing models and baselines on publicly available benchmarks, such as FLORES, establishing the utility of Samanantar. Our data and models are available publicly at Samanantar and we hope they will help advance research in NMT and multilingual NLP for Indic languages.}",
            issn = {2307-387X},
            doi = {10.1162/tacl_a_00452},
            url = {https://doi.org/10.1162/tacl\_a\_00452},
            eprint = {https://direct.mit.edu/tacl/article-pdf/doi/10.1162/tacl\_a\_00452/1987010/tacl\_a\_00452.pdf},
        }



