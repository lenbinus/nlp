"""
N-gram Language Model Implementation

This module contains the core NGramModel class that implements n-gram
language modeling with various smoothing options.
"""

import json
import pickle
import math
from typing import Dict, List, Tuple, Optional, Generator
from collections import Counter, defaultdict
from pathlib import Path

from .smoothing import SmoothingMethod, get_smoother, Smoother
from .corpus import (
    START_TOKEN, END_TOKEN, UNK_TOKEN,
    add_sentence_markers, build_vocabulary
)


class NGramModel:
    """
    N-gram Language Model
    
    An educational implementation of n-gram language models supporting
    various smoothing techniques.
    
    Attributes:
        n: The order of the n-gram model (e.g., 2 for bigram, 3 for trigram)
        smoothing: The smoothing method to use
        vocab: Set of vocabulary words
        ngram_counts: Counter of n-gram occurrences
        context_counts: Counter of (n-1)-gram context occurrences
    """
    
    def __init__(self, n: int = 3, smoothing: SmoothingMethod = SmoothingMethod.LAPLACE,
                 smoothing_params: Optional[Dict] = None):
        """
        Initialize the n-gram model.
        
        Args:
            n: Order of the model (default: 3 for trigram)
            smoothing: Smoothing method to use
            smoothing_params: Additional parameters for the smoother
        """
        if n < 1:
            raise ValueError("n must be at least 1")
        
        self.n = n
        self.smoothing_method = smoothing
        self.smoothing_params = smoothing_params or {}
        
        # Vocabulary
        self.vocab: set = set()
        self.word_to_idx: Dict[str, int] = {}
        self.idx_to_word: Dict[int, str] = {}
        
        # Counts
        self.ngram_counts: Counter = Counter()
        self.context_counts: Counter = Counter()
        
        # For Kneser-Ney smoothing
        self.continuation_counts: Dict[str, int] = defaultdict(int)
        self.context_type_counts: Dict[Tuple, int] = defaultdict(int)
        
        # Smoother (initialized after training)
        self.smoother: Optional[Smoother] = None
        
        # Training stats
        self.is_trained = False
        self.training_stats: Dict = {}
    
    def _count_ngrams(self, sentences: List[List[str]], 
                      progress_callback=None) -> None:
        """
        Count n-grams in the training data.
        
        Args:
            sentences: List of tokenized sentences
            progress_callback: Optional callback(current, total) for progress
        """
        total = len(sentences)
        
        for idx, sent in enumerate(sentences):
            # Add sentence markers
            marked = add_sentence_markers(sent, self.n)
            
            # Update vocabulary
            self.vocab.update(sent)
            
            # Generate and count n-grams
            for i in range(len(marked) - self.n + 1):
                ngram = tuple(marked[i:i + self.n])
                context = ngram[:-1]
                word = ngram[-1]
                
                self.ngram_counts[ngram] += 1
                self.context_counts[context] += 1
                
                # For Kneser-Ney: track unique contexts each word appears in
                if self.smoothing_method == SmoothingMethod.KNESER_NEY:
                    self.continuation_counts[word] += 1
                    self.context_type_counts[context] += 1
            
            if progress_callback and (idx + 1) % 100 == 0:
                progress_callback(idx + 1, total)
        
        if progress_callback:
            progress_callback(total, total)
    
    def train(self, sentences: List[List[str]], 
              min_count: int = 1,
              max_vocab_size: Optional[int] = None,
              progress_callback=None) -> Dict:
        """
        Train the n-gram model on sentences.
        
        Args:
            sentences: List of tokenized sentences
            min_count: Minimum count for vocabulary inclusion
            max_vocab_size: Maximum vocabulary size
            progress_callback: Optional callback(current, total, stage) for progress
        
        Returns:
            Dictionary of training statistics
        """
        # Stage 1: Build vocabulary
        if progress_callback:
            progress_callback(0, 3, "Building vocabulary")
        
        self.word_to_idx, self.idx_to_word = build_vocabulary(
            sentences, min_count=min_count, max_vocab_size=max_vocab_size
        )
        self.vocab = set(self.word_to_idx.keys())
        
        # Stage 2: Count n-grams
        if progress_callback:
            progress_callback(1, 3, "Counting n-grams")
        
        def ngram_progress(current, total):
            if progress_callback:
                progress_callback(current, total, "Processing sentences")
        
        self._count_ngrams(sentences, ngram_progress)
        
        # Stage 3: Initialize smoother
        if progress_callback:
            progress_callback(2, 3, "Initializing smoother")
        
        smoother_kwargs = {
            'ngram_counts': self.ngram_counts,
            'continuation_counts': dict(self.continuation_counts),
            'context_type_counts': dict(self.context_type_counts),
            **self.smoothing_params
        }
        
        self.smoother = get_smoother(
            self.smoothing_method,
            len(self.vocab),
            **smoother_kwargs
        )
        
        self.is_trained = True
        
        # Compute training statistics
        self.training_stats = {
            'n': self.n,
            'smoothing': self.smoothing_method.value,
            'vocab_size': len(self.vocab),
            'num_sentences': len(sentences),
            'unique_ngrams': len(self.ngram_counts),
            'total_ngrams': sum(self.ngram_counts.values()),
            'unique_contexts': len(self.context_counts)
        }
        
        if progress_callback:
            progress_callback(3, 3, "Complete")
        
        return self.training_stats
    
    def probability(self, word: str, context: Tuple[str, ...]) -> float:
        """
        Calculate P(word | context) using the trained model.
        
        Args:
            word: The word to calculate probability for
            context: The context tuple (n-1 previous words)
        
        Returns:
            Probability of word given context
        """
        if not self.is_trained:
            raise RuntimeError("Model must be trained before computing probabilities")
        
        # Handle unknown words
        if word not in self.vocab:
            word = UNK_TOKEN
        
        # Handle unknown context words
        context = tuple(w if w in self.vocab else UNK_TOKEN for w in context)
        
        ngram = context + (word,)
        count = self.ngram_counts.get(ngram, 0)
        context_count = self.context_counts.get(context, 0)
        
        return self.smoother.smooth(
            count, context_count,
            word=word, context=context
        )
    
    def log_probability(self, word: str, context: Tuple[str, ...]) -> float:
        """Calculate log probability (base e) of word given context."""
        prob = self.probability(word, context)
        return math.log(prob) if prob > 0 else float('-inf')
    
    def sentence_probability(self, sentence: List[str], log: bool = True) -> float:
        """
        Calculate the probability of a sentence.
        
        Args:
            sentence: List of tokens
            log: If True, return log probability
        
        Returns:
            Probability (or log probability) of the sentence
        """
        marked = add_sentence_markers(sentence, self.n)
        
        if log:
            total = 0.0
            for i in range(self.n - 1, len(marked)):
                context = tuple(marked[i - self.n + 1:i])
                word = marked[i]
                total += self.log_probability(word, context)
            return total
        else:
            total = 1.0
            for i in range(self.n - 1, len(marked)):
                context = tuple(marked[i - self.n + 1:i])
                word = marked[i]
                total *= self.probability(word, context)
            return total
    
    def perplexity(self, sentences: List[List[str]]) -> float:
        """
        Calculate perplexity on a set of sentences.
        
        Perplexity = 2^(-1/N * sum(log2(P(w_i|context))))
        
        Args:
            sentences: List of tokenized sentences
        
        Returns:
            Perplexity score (lower is better)
        """
        total_log_prob = 0.0
        total_words = 0
        
        for sent in sentences:
            marked = add_sentence_markers(sent, self.n)
            
            for i in range(self.n - 1, len(marked)):
                context = tuple(marked[i - self.n + 1:i])
                word = marked[i]
                prob = self.probability(word, context)
                
                if prob > 0:
                    total_log_prob += math.log2(prob)
                else:
                    total_log_prob += -100  # Large negative value for zero prob
                
                total_words += 1
        
        avg_log_prob = total_log_prob / total_words if total_words > 0 else 0
        return 2 ** (-avg_log_prob)
    
    def get_next_word_distribution(self, context: Tuple[str, ...], 
                                   top_k: int = 10) -> List[Tuple[str, float]]:
        """
        Get the probability distribution over next words given context.
        
        Args:
            context: The context tuple
            top_k: Number of top words to return
        
        Returns:
            List of (word, probability) tuples, sorted by probability
        """
        if not self.is_trained:
            raise RuntimeError("Model must be trained first")
        
        # Calculate probability for each word in vocabulary
        probs = []
        for word in self.vocab:
            if word not in (START_TOKEN,):  # Don't predict start token
                prob = self.probability(word, context)
                probs.append((word, prob))
        
        # Sort by probability (descending) and take top_k
        probs.sort(key=lambda x: x[1], reverse=True)
        return probs[:top_k]
    
    def generate(self, context: Optional[List[str]] = None, 
                 max_length: int = 20,
                 temperature: float = 1.0) -> List[str]:
        """
        Generate text given a starting context.
        
        Args:
            context: Starting words (if None, starts fresh)
            max_length: Maximum number of words to generate
            temperature: Sampling temperature (higher = more random)
        
        Returns:
            List of generated tokens
        """
        import random
        
        if not self.is_trained:
            raise RuntimeError("Model must be trained first")
        
        # Initialize context
        if context is None:
            current_context = [START_TOKEN] * (self.n - 1)
        else:
            current_context = [START_TOKEN] * (self.n - 1) + context
            current_context = current_context[-(self.n - 1):]
        
        generated = []
        
        for _ in range(max_length):
            context_tuple = tuple(current_context)
            
            # Get distribution
            distribution = self.get_next_word_distribution(context_tuple, top_k=100)
            
            if not distribution:
                break
            
            # Apply temperature
            if temperature != 1.0:
                probs = [(w, p ** (1/temperature)) for w, p in distribution]
                total = sum(p for _, p in probs)
                probs = [(w, p/total) for w, p in probs]
            else:
                probs = distribution
            
            # Sample
            words, weights = zip(*probs)
            word = random.choices(words, weights=weights, k=1)[0]
            
            if word == END_TOKEN:
                break
            
            generated.append(word)
            current_context = current_context[1:] + [word]
        
        return generated
    
    def get_top_ngrams(self, top_k: int = 100) -> List[Tuple[Tuple[str, ...], int]]:
        """Get the most frequent n-grams."""
        return self.ngram_counts.most_common(top_k)
    
    def get_top_words(self, top_k: int = 100) -> List[Tuple[str, int]]:
        """Get the most frequent words based on unigram counts."""
        word_counts = Counter()
        for ngram, count in self.ngram_counts.items():
            word_counts[ngram[-1]] += count
        
        # Filter out special tokens
        filtered = [(w, c) for w, c in word_counts.items() 
                    if w not in (START_TOKEN, END_TOKEN, UNK_TOKEN)]
        filtered.sort(key=lambda x: x[1], reverse=True)
        return filtered[:top_k]
    
    def save(self, path: str) -> None:
        """Save the model to a file."""
        path = Path(path)
        
        data = {
            'n': self.n,
            'smoothing_method': self.smoothing_method.value,
            'smoothing_params': self.smoothing_params,
            'vocab': list(self.vocab),
            'word_to_idx': self.word_to_idx,
            'idx_to_word': {int(k): v for k, v in self.idx_to_word.items()},
            'ngram_counts': dict(self.ngram_counts),
            'context_counts': dict(self.context_counts),
            'continuation_counts': dict(self.continuation_counts),
            'context_type_counts': {str(k): v for k, v in self.context_type_counts.items()},
            'training_stats': self.training_stats,
            'is_trained': self.is_trained
        }
        
        with open(path, 'wb') as f:
            pickle.dump(data, f)
    
    @classmethod
    def load(cls, path: str) -> 'NGramModel':
        """Load a model from a file."""
        path = Path(path)
        
        with open(path, 'rb') as f:
            data = pickle.load(f)
        
        model = cls(
            n=data['n'],
            smoothing=SmoothingMethod(data['smoothing_method']),
            smoothing_params=data['smoothing_params']
        )
        
        model.vocab = set(data['vocab'])
        model.word_to_idx = data['word_to_idx']
        model.idx_to_word = {int(k): v for k, v in data['idx_to_word'].items()}
        model.ngram_counts = Counter(data['ngram_counts'])
        model.context_counts = Counter(data['context_counts'])
        model.continuation_counts = defaultdict(int, data['continuation_counts'])
        model.context_type_counts = defaultdict(int, {
            eval(k): v for k, v in data['context_type_counts'].items()
        })
        model.training_stats = data['training_stats']
        model.is_trained = data['is_trained']
        
        # Re-initialize smoother
        if model.is_trained:
            model.smoother = get_smoother(
                model.smoothing_method,
                len(model.vocab),
                ngram_counts=model.ngram_counts,
                continuation_counts=dict(model.continuation_counts),
                context_type_counts=dict(model.context_type_counts),
                **model.smoothing_params
            )
        
        return model
