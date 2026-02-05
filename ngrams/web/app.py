"""
N-gram Language Model Web Dashboard

A Flask application providing a web interface for training n-gram models
and visualizing predictions as an interactive tree.
"""

import os
import json
import threading
import time
from pathlib import Path
from typing import Optional, Dict, List
from dataclasses import dataclass, asdict
from flask import Flask, render_template, jsonify, request, send_from_directory

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from ngram import NGramModel, SmoothingMethod
from ngram.corpus import load_brown_corpus, get_brown_categories


app = Flask(__name__)
app.config['SECRET_KEY'] = 'ngram-visualization-secret'

# Global state
model: Optional[NGramModel] = None
training_status: Dict = {
    'is_training': False,
    'progress': 0,
    'total': 100,
    'stage': 'idle',
    'message': '',
    'error': None,
    'stats': None
}
training_lock = threading.Lock()


@dataclass
class TreeNode:
    """Represents a node in the prediction tree."""
    word: str
    probability: float
    count: int = 0
    children: List['TreeNode'] = None
    
    def to_dict(self):
        result = {
            'name': self.word,
            'probability': self.probability,
            'count': self.count
        }
        if self.children:
            result['children'] = [c.to_dict() for c in self.children]
        return result


def build_prediction_tree(model: NGramModel, start_word: str, 
                          depth: int = 3, branching: int = 5,
                          context: Optional[List[str]] = None) -> TreeNode:
    """
    Build a tree of predictions starting from a word.
    
    Args:
        model: Trained n-gram model
        start_word: Starting word for the tree
        depth: How many levels deep to go
        branching: How many children per node
    
    Returns:
        Root TreeNode
    """
    n = model.n
    
    def build_node(ctx: tuple, current_depth: int) -> List[TreeNode]:
        if current_depth >= depth:
            return None
        
        predictions = model.get_next_word_distribution(ctx, top_k=branching)
        
        children = []
        for word, prob in predictions:
            if word in ('<s>', '</s>', '<UNK>'):
                continue
            
            # Get count for this word in context
            ngram = ctx + (word,)
            count = model.ngram_counts.get(ngram, 0)
            
            # Build new context for children (keep last n-1 words)
            new_ctx = (ctx + (word,))[-(n-1):] if n > 1 else tuple()
            
            node = TreeNode(
                word=word,
                probability=prob,
                count=count,
                children=build_node(new_ctx, current_depth + 1)
            )
            children.append(node)
        
        return children if children else None
    
    # Build initial context from provided context or default
    if context and len(context) > 0:
        # Use the provided context (last n-1 words for n-gram)
        if n > 1:
            initial_context = tuple(context[-(n-1):])
            # Pad with start tokens if needed
            if len(initial_context) < n - 1:
                initial_context = ('<s>',) * (n - 1 - len(initial_context)) + initial_context
        else:
            initial_context = tuple()
    else:
        # Default: use start tokens + start_word
        initial_context = ('<s>',) * (n - 2) + (start_word,) if n > 1 else tuple()
        if n == 1:
            initial_context = tuple()
    
    root = TreeNode(
        word=start_word,
        probability=1.0,
        count=sum(c for ng, c in model.ngram_counts.items() if start_word in ng),
        children=build_node(initial_context, 0)
    )
    
    return root


def train_model_async(n: int, smoothing: str, categories: Optional[List[str]],
                      min_count: int, max_vocab_size: Optional[int]):
    """Train model in background thread."""
    global model, training_status
    
    try:
        with training_lock:
            training_status['is_training'] = True
            training_status['progress'] = 0
            training_status['stage'] = 'loading'
            training_status['message'] = 'Loading Brown corpus...'
            training_status['error'] = None
        
        # Load corpus
        sentences, corpus_stats = load_brown_corpus(categories=categories)
        
        with training_lock:
            training_status['stage'] = 'loaded'
            training_status['message'] = f'Loaded {corpus_stats["num_sentences"]:,} sentences'
            training_status['progress'] = 10
        
        # Map smoothing
        smoothing_map = {
            'none': SmoothingMethod.NONE,
            'laplace': SmoothingMethod.LAPLACE,
            'add_k': SmoothingMethod.ADD_K,
            'good_turing': SmoothingMethod.GOOD_TURING,
            'kneser_ney': SmoothingMethod.KNESER_NEY
        }
        
        smoothing_method = smoothing_map.get(smoothing.lower(), SmoothingMethod.LAPLACE)
        
        # Create model
        new_model = NGramModel(n=n, smoothing=smoothing_method)
        
        # Training progress callback
        def progress_callback(current, total, stage=""):
            with training_lock:
                # Map progress to 10-95 range
                progress = 10 + int((current / max(total, 1)) * 85)
                training_status['progress'] = progress
                training_status['stage'] = stage
                training_status['message'] = f'{stage}: {current:,}/{total:,}'
        
        # Train
        with training_lock:
            training_status['stage'] = 'training'
            training_status['message'] = 'Training model...'
        
        stats = new_model.train(
            sentences,
            min_count=min_count,
            max_vocab_size=max_vocab_size,
            progress_callback=progress_callback
        )
        
        # Update global model
        model = new_model
        
        with training_lock:
            training_status['progress'] = 100
            training_status['stage'] = 'complete'
            training_status['message'] = 'Training complete!'
            training_status['stats'] = stats
            training_status['is_training'] = False
        
    except Exception as e:
        with training_lock:
            training_status['error'] = str(e)
            training_status['is_training'] = False
            training_status['stage'] = 'error'
            training_status['message'] = f'Error: {str(e)}'


@app.route('/')
def index():
    """Render main dashboard."""
    categories = get_brown_categories()
    smoothing_methods = ['none', 'laplace', 'add_k', 'good_turing', 'kneser_ney']
    return render_template('index.html', 
                          categories=categories,
                          smoothing_methods=smoothing_methods)


@app.route('/api/train', methods=['POST'])
def api_train():
    """Start model training."""
    global training_status
    
    with training_lock:
        if training_status['is_training']:
            return jsonify({'error': 'Training already in progress'}), 400
    
    data = request.json
    n = data.get('n', 3)
    smoothing = data.get('smoothing', 'laplace')
    categories = data.get('categories', None)
    min_count = data.get('min_count', 2)
    max_vocab_size = data.get('max_vocab_size', None)
    
    # Start training in background
    thread = threading.Thread(
        target=train_model_async,
        args=(n, smoothing, categories, min_count, max_vocab_size)
    )
    thread.daemon = True
    thread.start()
    
    return jsonify({'message': 'Training started'})


@app.route('/api/status')
def api_status():
    """Get training status."""
    with training_lock:
        return jsonify(training_status)


@app.route('/api/model/info')
def api_model_info():
    """Get model information."""
    if model is None:
        return jsonify({'error': 'No model trained'}), 400
    
    return jsonify({
        'is_trained': model.is_trained,
        'n': model.n,
        'smoothing': model.smoothing_method.value,
        'vocab_size': len(model.vocab),
        'stats': model.training_stats
    })


@app.route('/api/top_words')
def api_top_words():
    """Get top words from the model."""
    if model is None:
        return jsonify({'error': 'No model trained'}), 400
    
    k = request.args.get('k', 100, type=int)
    top_words = model.get_top_words(k)
    
    return jsonify([
        {'word': word, 'count': count}
        for word, count in top_words
    ])


@app.route('/api/predict')
def api_predict():
    """Get predictions for a context."""
    if model is None:
        return jsonify({'error': 'No model trained'}), 400
    
    context = request.args.get('context', '')
    k = request.args.get('k', 10, type=int)
    
    words = context.lower().split() if context else []
    n = model.n
    
    if len(words) == 0:
        context_tuple = ('<s>',) * (n - 1)
    elif len(words) < n - 1:
        context_tuple = ('<s>',) * (n - 1 - len(words)) + tuple(words)
    else:
        context_tuple = tuple(words[-(n-1):])
    
    predictions = model.get_next_word_distribution(context_tuple, top_k=k)
    
    return jsonify({
        'context': list(context_tuple),
        'predictions': [
            {'word': word, 'probability': prob}
            for word, prob in predictions
        ]
    })


@app.route('/api/tree')
def api_tree():
    """Get prediction tree for visualization."""
    if model is None:
        return jsonify({'error': 'No model trained'}), 400
    
    start_word = request.args.get('word', 'the')
    depth = request.args.get('depth', 3, type=int)
    branching = request.args.get('branching', 5, type=int)
    context_str = request.args.get('context', '')
    
    # Parse context from space-separated string
    context = context_str.lower().split() if context_str.strip() else None
    
    # Limit depth and branching for performance
    depth = min(depth, 5)
    branching = min(branching, 10)
    
    tree = build_prediction_tree(model, start_word, depth, branching, context=context)
    
    return jsonify(tree.to_dict())


@app.route('/api/generate', methods=['POST'])
def api_generate():
    """Generate text from the model."""
    if model is None:
        return jsonify({'error': 'No model trained'}), 400
    
    data = request.json
    context = data.get('context', '').split() if data.get('context') else None
    max_length = min(data.get('max_length', 20), 100)
    temperature = data.get('temperature', 1.0)
    
    generated = model.generate(
        context=context,
        max_length=max_length,
        temperature=temperature
    )
    
    return jsonify({
        'context': context,
        'generated': generated,
        'text': ' '.join(generated)
    })


@app.route('/api/perplexity', methods=['POST'])
def api_perplexity():
    """Calculate perplexity for given sentences."""
    if model is None:
        return jsonify({'error': 'No model trained'}), 400
    
    data = request.json
    sentences = data.get('sentences', [])
    
    if not sentences:
        return jsonify({'error': 'No sentences provided'}), 400
    
    # Parse sentences
    parsed = [sent.lower().split() for sent in sentences if sent.strip()]
    
    perplexity = model.perplexity(parsed)
    
    return jsonify({
        'perplexity': perplexity,
        'num_sentences': len(parsed)
    })


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='N-gram Model Web Dashboard')
    parser.add_argument('--host', default='127.0.0.1', help='Host to bind to')
    parser.add_argument('--port', type=int, default=5000, help='Port to bind to')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    
    args = parser.parse_args()
    
    print(f"\n🌐 Starting N-gram Language Model Dashboard")
    print(f"   Open http://{args.host}:{args.port} in your browser\n")
    
    app.run(host=args.host, port=args.port, debug=args.debug, threaded=True)
