# Tokenization Methods

Educational implementations of major subword tokenization algorithms used in modern NLP.

## Methods Implemented

### 1. Byte Pair Encoding (BPE)
**Used by:** GPT-2, GPT-3, RoBERTa

BPE iteratively merges the most frequent pair of adjacent tokens:
1. Start with character-level vocabulary
2. Count all adjacent pairs
3. Merge the most frequent pair
4. Repeat until target vocabulary size

```
"lower" → ['l', 'o', 'w', 'e', 'r']
After merges: ['low', 'er']
```

### 2. WordPiece
**Used by:** BERT, DistilBERT, Electra

Similar to BPE but uses likelihood-based scoring:
```
Score(a,b) = freq(ab) / (freq(a) × freq(b))
```

Uses `##` prefix for continuation tokens:
```
"playing" → ['play', '##ing']
```

### 3. Unigram Language Model
**Used by:** SentencePiece, T5, XLNet, ALBERT

Opposite approach - starts large and prunes:
1. Initialize with all substrings
2. Compute loss if each token is removed
3. Remove tokens with smallest loss
4. Use Viterbi algorithm for optimal tokenization

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run web dashboard
python web/app.py
# Open http://localhost:5001
```

## Web Dashboard Features

- **Train** any of the three tokenization methods
- **Compare** tokenization across methods side-by-side
- **Step-by-step visualization** of the tokenization process
- **Vocabulary exploration** with token samples

## Project Structure

```
tokenization/
├── tokenizers/
│   ├── base.py       # Base class with common interface
│   ├── bpe.py        # Byte Pair Encoding
│   ├── wordpiece.py  # WordPiece tokenizer
│   └── unigram.py    # Unigram LM tokenizer
├── web/
│   ├── app.py        # Flask application
│   ├── templates/    # HTML templates
│   └── static/       # CSS/JS
└── requirements.txt
```

## API Usage

```python
from tokenizers import BPETokenizer, WordPieceTokenizer, UnigramTokenizer

# Train BPE
bpe = BPETokenizer(vocab_size=1000)
bpe.train(["list", "of", "training", "sentences"])

# Tokenize
tokens = bpe.tokenize("Hello world")
print(tokens)  # ['Ġhello', 'Ġworld']

# Get step-by-step breakdown
tokens, steps = bpe.tokenize("Hello", return_steps=True)
for step in steps:
    print(step.description)
```

## Comparison

| Method | Approach | Tokenization | Used By |
|--------|----------|--------------|---------|
| BPE | Bottom-up merging | Greedy | GPT, RoBERTa |
| WordPiece | Likelihood scoring | Greedy longest-match | BERT |
| Unigram | Top-down pruning | Viterbi optimal | T5, XLNet |

## Educational Notes

### Why Subword Tokenization?

1. **Fixed vocabulary**: Unlike word-level, vocabulary is bounded
2. **No OOV**: Unknown words split into known subwords
3. **Morphology**: Captures word parts ("play" + "ing")
4. **Efficiency**: Common words = single tokens, rare = multiple

### Key Insight

The algorithms differ in HOW they build vocabulary:
- **BPE**: What pairs appear together most?
- **WordPiece**: What merges maximize likelihood?
- **Unigram**: What tokens contribute least to loss?
