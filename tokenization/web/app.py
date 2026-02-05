"""
Tokenization Visualization Web Dashboard

A Flask application for training and visualizing different tokenization methods.
"""

import os
import json
import threading
from pathlib import Path
from typing import Optional, Dict, List
from flask import Flask, render_template, jsonify, request

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from tokenizers import BPETokenizer, WordPieceTokenizer, UnigramTokenizer


app = Flask(__name__)
app.config['SECRET_KEY'] = 'tokenization-visualization-secret'

# Global state
tokenizers: Dict[str, any] = {}
training_status: Dict = {
    'is_training': False,
    'progress': 0,
    'method': None,
    'message': '',
    'error': None
}
training_lock = threading.Lock()

# Sample training corpus
SAMPLE_CORPUS = [
    "the quick brown fox jumps over the lazy dog",
    "machine learning is transforming how we process natural language",
    "tokenization is a fundamental step in natural language processing",
    "byte pair encoding was originally a compression algorithm",
    "wordpiece tokenization is used by bert and other transformer models",
    "unigram language model tokenization uses probabilistic methods",
    "subword tokenization helps handle out of vocabulary words",
    "the cat sat on the mat and looked at the rat",
    "natural language processing enables computers to understand human language",
    "deep learning models require tokenized input for training",
    "transformers have revolutionized the field of nlp",
    "attention mechanisms allow models to focus on relevant parts of the input",
    "pretrained language models can be fine tuned for specific tasks",
    "vocabulary size affects model performance and memory usage",
    "rare words are often split into subword units",
    "common words typically become single tokens",
    "the embedding layer maps tokens to dense vectors",
    "sequence to sequence models use encoder decoder architecture",
    "language models learn to predict the next word in a sequence",
    "neural networks can capture complex patterns in text data"
]


def get_tokenizer_class(method: str):
    """Get tokenizer class by name."""
    classes = {
        'bpe': BPETokenizer,
        'wordpiece': WordPieceTokenizer,
        'unigram': UnigramTokenizer
    }
    return classes.get(method.lower())


def train_tokenizer_async(method: str, vocab_size: int, corpus: List[str]):
    """Train tokenizer in background thread."""
    global tokenizers, training_status
    
    try:
        with training_lock:
            training_status['is_training'] = True
            training_status['progress'] = 0
            training_status['method'] = method
            training_status['message'] = f'Initializing {method} tokenizer...'
            training_status['error'] = None
        
        # Create tokenizer
        TokenizerClass = get_tokenizer_class(method)
        if not TokenizerClass:
            raise ValueError(f"Unknown tokenizer: {method}")
        
        tokenizer = TokenizerClass(vocab_size=vocab_size)
        
        with training_lock:
            training_status['message'] = 'Training...'
            training_status['progress'] = 10
        
        # Train
        history = tokenizer.train(corpus, verbose=False)
        
        # Store tokenizer
        tokenizers[method] = tokenizer
        
        with training_lock:
            training_status['progress'] = 100
            training_status['message'] = 'Training complete!'
            training_status['is_training'] = False
        
    except Exception as e:
        with training_lock:
            training_status['error'] = str(e)
            training_status['is_training'] = False
            training_status['message'] = f'Error: {str(e)}'


@app.route('/')
def index():
    """Render main dashboard."""
    return render_template('index.html')


@app.route('/api/train', methods=['POST'])
def api_train():
    """Start tokenizer training."""
    global training_status
    
    with training_lock:
        if training_status['is_training']:
            return jsonify({'error': 'Training already in progress'}), 400
    
    data = request.json
    method = data.get('method', 'bpe')
    vocab_size = data.get('vocab_size', 500)
    custom_corpus = data.get('corpus', None)
    
    corpus = custom_corpus.split('\n') if custom_corpus else SAMPLE_CORPUS
    corpus = [line.strip() for line in corpus if line.strip()]
    
    # Start training in background
    thread = threading.Thread(
        target=train_tokenizer_async,
        args=(method, vocab_size, corpus)
    )
    thread.daemon = True
    thread.start()
    
    return jsonify({'message': f'Training {method} started'})


@app.route('/api/status')
def api_status():
    """Get training status."""
    with training_lock:
        return jsonify(training_status)


@app.route('/api/tokenizers')
def api_tokenizers():
    """Get list of trained tokenizers."""
    result = {}
    for name, tok in tokenizers.items():
        result[name] = {
            'name': tok.name,
            'description': tok.description,
            'vocab_size': len(tok.vocab),
            'is_trained': tok.is_trained
        }
    return jsonify(result)


@app.route('/api/tokenize', methods=['POST'])
def api_tokenize():
    """Tokenize text with specified method."""
    data = request.json
    text = data.get('text', '')
    method = data.get('method', 'bpe')
    
    if method not in tokenizers:
        return jsonify({'error': f'Tokenizer {method} not trained'}), 400
    
    tokenizer = tokenizers[method]
    tokens, steps = tokenizer.tokenize(text, return_steps=True)
    
    # Convert steps to serializable format
    steps_data = []
    for step in steps:
        steps_data.append({
            'type': step.step_type,
            'description': step.description,
            'tokens': step.tokens,
            'details': step.details
        })
    
    return jsonify({
        'method': method,
        'text': text,
        'tokens': tokens,
        'num_tokens': len(tokens),
        'steps': steps_data
    })


@app.route('/api/compare', methods=['POST'])
def api_compare():
    """Compare tokenization across all trained methods."""
    data = request.json
    text = data.get('text', '')
    
    results = {}
    for method, tokenizer in tokenizers.items():
        tokens = tokenizer.tokenize(text)
        results[method] = {
            'name': tokenizer.name,
            'tokens': tokens,
            'num_tokens': len(tokens),
            'token_ids': tokenizer.encode(text)
        }
    
    return jsonify(results)


@app.route('/api/vocab')
def api_vocab():
    """Get vocabulary for a tokenizer."""
    method = request.args.get('method', 'bpe')
    limit = request.args.get('limit', 100, type=int)
    
    if method not in tokenizers:
        return jsonify({'error': f'Tokenizer {method} not trained'}), 400
    
    tokenizer = tokenizers[method]
    vocab = list(tokenizer.vocab.items())[:limit]
    
    return jsonify({
        'method': method,
        'vocab_size': len(tokenizer.vocab),
        'vocab_sample': vocab
    })


@app.route('/api/training_history')
def api_training_history():
    """Get training history for visualization."""
    method = request.args.get('method', 'bpe')
    
    if method not in tokenizers:
        return jsonify({'error': f'Tokenizer {method} not trained'}), 400
    
    tokenizer = tokenizers[method]
    
    history = []
    for step in tokenizer.training_history:
        history.append({
            'iteration': step.iteration,
            'action': step.action,
            'description': step.description,
            'vocab_size': step.vocab_size,
            'details': step.details
        })
    
    return jsonify({
        'method': method,
        'name': tokenizer.name,
        'history': history
    })


@app.route('/api/info')
def api_info():
    """Get information about tokenization methods."""
    return jsonify({
        'methods': [
            {
                'id': 'bpe',
                'name': 'Byte Pair Encoding (BPE)',
                'description': 'Iteratively merges most frequent pairs. Used by GPT-2, GPT-3, RoBERTa.',
                'pros': ['Simple and effective', 'Deterministic', 'Good compression'],
                'cons': ['Greedy - may not find optimal tokenization', 'Order-dependent']
            },
            {
                'id': 'wordpiece',
                'name': 'WordPiece',
                'description': 'Uses likelihood-based scoring for merges. Used by BERT, DistilBERT.',
                'pros': ['Better handling of rare words', 'Principled scoring'],
                'cons': ['More complex training', 'Greedy tokenization']
            },
            {
                'id': 'unigram',
                'name': 'Unigram LM',
                'description': 'Starts large and prunes down. Uses Viterbi for optimal tokenization.',
                'pros': ['Optimal tokenization via Viterbi', 'Probabilistic framework'],
                'cons': ['Slower training', 'More complex implementation']
            }
        ]
    })


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Tokenization Visualization Dashboard')
    parser.add_argument('--host', default='127.0.0.1', help='Host to bind to')
    parser.add_argument('--port', type=int, default=5001, help='Port to bind to')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    
    args = parser.parse_args()
    
    print(f"\n🔤 Starting Tokenization Visualization Dashboard")
    print(f"   Open http://{args.host}:{args.port} in your browser\n")
    
    app.run(host=args.host, port=args.port, debug=args.debug, threaded=True)
