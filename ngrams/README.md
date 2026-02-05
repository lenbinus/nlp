# N-gram Language Model

An educational implementation of n-gram language models with interactive visualizations.

## Features

- 📊 **Multiple N-gram Orders**: Support for unigrams through 5-grams
- 🔧 **Various Smoothing Methods**: None, Laplace, Add-k, Good-Turing, Kneser-Ney
- 📚 **Brown Corpus**: Built-in support for NLTK's Brown corpus
- 🎨 **Beautiful Terminal UI**: Rich progress bars and formatted output
- 🌐 **Web Dashboard**: Interactive training and visualization interface
- 🌳 **Tree Visualization**: D3.js-powered prediction tree exploration

## Quick Start

```bash
# From this directory
pip install -r requirements.txt

# Train from command line
python train.py --n 3 --smoothing laplace --interactive

# Or run the web dashboard
python run_server.py
# Then open http://localhost:5000
```

## CLI Options

| Option | Description | Default |
|--------|-------------|---------|
| `-n, --n` | N-gram order | 3 |
| `-s, --smoothing` | Smoothing method | laplace |
| `-c, --categories` | Brown corpus categories | All |
| `--min-count` | Minimum word count | 2 |
| `--save` | Save model path | None |
| `-i, --interactive` | Interactive demo | False |

## Smoothing Methods

- **none** - No smoothing (MLE)
- **laplace** - Add-one smoothing
- **add_k** - Add-k smoothing (k=0.5)
- **good_turing** - Good-Turing smoothing
- **kneser_ney** - Kneser-Ney smoothing (recommended)

## Project Structure

```
ngrams/
├── ngram/           # Core model implementation
│   ├── model.py     # NGramModel class
│   ├── smoothing.py # Smoothing implementations
│   ├── corpus.py    # Corpus loading
│   └── training.py  # Terminal UI
├── web/             # Web dashboard
│   ├── app.py       # Flask application
│   ├── templates/   # HTML templates
│   └── static/      # CSS/JS assets
├── train.py         # CLI training script
├── run_server.py    # Web server launcher
└── requirements.txt
```
