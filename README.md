# N-gram Language Model

An educational implementation of n-gram language models with interactive visualizations.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

## Features

- 📊 **Multiple N-gram Orders**: Support for unigrams through 5-grams
- 🔧 **Various Smoothing Methods**:
  - No smoothing (MLE)
  - Laplace (Add-one) smoothing
  - Add-k smoothing
  - Good-Turing smoothing
  - Kneser-Ney smoothing (recommended)
- 📚 **Brown Corpus**: Built-in support for NLTK's Brown corpus with category selection
- 🎨 **Beautiful Terminal UI**: Rich progress bars and formatted output
- 🌐 **Web Dashboard**: Interactive training and visualization interface
- 🌳 **Tree Visualization**: D3.js-powered prediction tree exploration

## Screenshots

### Terminal Training
```
╭────────────────────────────────────╮
│  N-gram Language Model Training    │
╰────────────────────────────────────╯

╭─ Configuration ─────────────────────╮
│ Model Order (n)    3               │
│ Smoothing Method   laplace         │
│ Categories         All             │
╰─────────────────────────────────────╯

✓ Loaded 57,340 sentences (1,161,192 tokens)

Processing sentences ━━━━━━━━━━━━━ 100% 0:00:12

✓ Training complete!
```

### Web Dashboard
The web interface provides:
- Real-time training progress
- Interactive prediction tree
- Text generation
- Top word exploration

## Installation

```bash
# Clone the repository
git clone https://github.com/lenbinus/nlp.git
cd nlp

# Install dependencies
pip install -r requirements.txt

# Download NLTK data (automatic on first run)
python -c "import nltk; nltk.download('brown'); nltk.download('punkt')"
```

## Usage

### Command Line Training

```bash
# Basic trigram model with Laplace smoothing
python train.py --n 3 --smoothing laplace

# Bigram with Kneser-Ney smoothing on specific categories
python train.py --n 2 --smoothing kneser_ney --categories news fiction

# Save model and run interactive demo
python train.py --n 3 --smoothing laplace --save model.pkl --interactive

# Load pre-trained model
python train.py --load model.pkl --interactive
```

### CLI Options

| Option | Description | Default |
|--------|-------------|---------|
| `-n, --n` | N-gram order | 3 |
| `-s, --smoothing` | Smoothing method | laplace |
| `-c, --categories` | Brown corpus categories | All |
| `--min-count` | Minimum word count | 2 |
| `--max-vocab` | Maximum vocabulary size | Unlimited |
| `--save` | Save model path | None |
| `--load` | Load model path | None |
| `-i, --interactive` | Interactive demo | False |

### Web Dashboard

```bash
# Start the web server
python web/app.py

# Or with options
python web/app.py --host 0.0.0.0 --port 8080 --debug
```

Then open http://localhost:5000 in your browser.

## Project Structure

```
nlp/
├── ngram/
│   ├── __init__.py      # Package exports
│   ├── model.py         # Core NGramModel class
│   ├── smoothing.py     # Smoothing implementations
│   ├── corpus.py        # Corpus loading/preprocessing
│   └── training.py      # Terminal UI training
├── web/
│   ├── app.py           # Flask web application
│   ├── templates/
│   │   └── index.html   # Dashboard template
│   └── static/
│       ├── css/
│       │   └── style.css
│       └── js/
│           └── tree.js  # D3.js visualization
├── train.py             # CLI training script
├── requirements.txt
└── README.md
```

## Smoothing Methods Explained

### Laplace (Add-One) Smoothing
Adds 1 to all n-gram counts, ensuring no zero probabilities.

```
P(w|context) = (count(context, w) + 1) / (count(context) + V)
```

### Add-k Smoothing
Generalized Laplace with adjustable k (typically 0 < k < 1).

```
P(w|context) = (count(context, w) + k) / (count(context) + k*V)
```

### Good-Turing Smoothing
Adjusts counts based on frequency of frequencies, reallocating probability mass from seen to unseen events.

### Kneser-Ney Smoothing
State-of-the-art smoothing using absolute discounting and continuation probability. Particularly effective for lower-order distributions.

## API Reference

### NGramModel

```python
from ngram import NGramModel, SmoothingMethod

# Create model
model = NGramModel(n=3, smoothing=SmoothingMethod.KNESER_NEY)

# Train
sentences = [['the', 'cat', 'sat'], ['a', 'dog', 'ran']]
model.train(sentences)

# Get probability
prob = model.probability('sat', ('the', 'cat'))

# Get predictions
predictions = model.get_next_word_distribution(('the', 'cat'), top_k=10)

# Generate text
text = model.generate(context=['the'], max_length=20)

# Calculate perplexity
perplexity = model.perplexity(test_sentences)

# Save/Load
model.save('model.pkl')
loaded = NGramModel.load('model.pkl')
```

## Educational Notes

### What is an N-gram?

An n-gram is a contiguous sequence of n items (words, characters, etc.) from a text. For example, in "the cat sat":
- Unigrams (n=1): "the", "cat", "sat"
- Bigrams (n=2): "the cat", "cat sat"
- Trigrams (n=3): "the cat sat"

### Language Modeling

N-gram models estimate the probability of a word given its preceding context:

```
P(sat | the, cat) = count("the cat sat") / count("the cat")
```

This allows us to:
- Predict likely next words
- Score sentence likelihood
- Generate text

### The Zero Probability Problem

If an n-gram never appears in training data, its count is 0, giving probability 0. This is problematic because:
1. A single unseen n-gram makes entire sentences have probability 0
2. Perplexity becomes infinite

Smoothing methods address this by redistributing probability mass from seen to unseen events.

## License

MIT License - feel free to use for educational purposes.

## Contributing

Contributions welcome! Please feel free to submit issues and pull requests.
