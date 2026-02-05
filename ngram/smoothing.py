"""
Smoothing Methods for N-gram Language Models

This module implements various smoothing techniques to handle the zero-probability
problem in n-gram models.
"""

from enum import Enum
from typing import Dict, Tuple, Optional
from collections import Counter
import math


class SmoothingMethod(Enum):
    """Available smoothing methods."""
    NONE = "none"
    LAPLACE = "laplace"           # Add-one smoothing
    ADD_K = "add_k"               # Add-k smoothing (generalized Laplace)
    GOOD_TURING = "good_turing"   # Good-Turing smoothing
    KNESER_NEY = "kneser_ney"     # Kneser-Ney smoothing


class Smoother:
    """Base class for smoothing implementations."""
    
    def __init__(self, vocab_size: int):
        self.vocab_size = vocab_size
    
    def smooth(self, count: int, context_count: int, **kwargs) -> float:
        """Return smoothed probability."""
        raise NotImplementedError


class NoSmoothing(Smoother):
    """No smoothing - raw maximum likelihood estimation."""
    
    def smooth(self, count: int, context_count: int, **kwargs) -> float:
        if context_count == 0:
            return 0.0
        return count / context_count


class LaplaceSmoothing(Smoother):
    """
    Laplace (Add-One) Smoothing
    
    P(w|context) = (count(context, w) + 1) / (count(context) + V)
    
    Where V is the vocabulary size.
    """
    
    def smooth(self, count: int, context_count: int, **kwargs) -> float:
        return (count + 1) / (context_count + self.vocab_size)


class AddKSmoothing(Smoother):
    """
    Add-K Smoothing (Generalized Laplace)
    
    P(w|context) = (count(context, w) + k) / (count(context) + k*V)
    
    Where k is typically a small value (0 < k < 1).
    """
    
    def __init__(self, vocab_size: int, k: float = 0.5):
        super().__init__(vocab_size)
        self.k = k
    
    def smooth(self, count: int, context_count: int, **kwargs) -> float:
        return (count + self.k) / (context_count + self.k * self.vocab_size)


class GoodTuringSmoothing(Smoother):
    """
    Good-Turing Smoothing
    
    Adjusts counts based on the frequency of frequencies.
    For count r, adjusted count r* = (r+1) * N_{r+1} / N_r
    
    Where N_r is the number of n-grams that occur exactly r times.
    """
    
    def __init__(self, vocab_size: int, ngram_counts: Counter):
        super().__init__(vocab_size)
        # Build frequency of frequencies
        self.freq_of_freq = Counter(ngram_counts.values())
        self.total_ngrams = sum(ngram_counts.values())
        self._cache = {}
    
    def _adjusted_count(self, r: int) -> float:
        """Calculate adjusted count r* for count r."""
        if r in self._cache:
            return self._cache[r]
        
        if r == 0:
            # Probability mass for unseen events
            n1 = self.freq_of_freq.get(1, 1)
            result = n1 / self.total_ngrams if self.total_ngrams > 0 else 0
        else:
            nr = self.freq_of_freq.get(r, 0)
            nr1 = self.freq_of_freq.get(r + 1, 0)
            
            if nr == 0:
                result = r
            else:
                result = (r + 1) * nr1 / nr if nr1 > 0 else r
        
        self._cache[r] = result
        return result
    
    def smooth(self, count: int, context_count: int, **kwargs) -> float:
        if context_count == 0:
            return 1 / self.vocab_size
        
        adjusted = self._adjusted_count(count)
        return adjusted / context_count


class KneserNeySmoothing(Smoother):
    """
    Kneser-Ney Smoothing
    
    One of the most effective smoothing methods, using absolute discounting
    and a novel lower-order distribution based on continuation probability.
    
    P_KN(w|context) = max(count(context,w) - d, 0) / count(context) 
                      + λ(context) * P_continuation(w)
    
    Where d is the discount and λ is the interpolation weight.
    """
    
    def __init__(self, vocab_size: int, discount: float = 0.75,
                 continuation_counts: Optional[Dict[str, int]] = None,
                 context_type_counts: Optional[Dict[Tuple, int]] = None):
        super().__init__(vocab_size)
        self.discount = discount
        self.continuation_counts = continuation_counts or {}
        self.context_type_counts = context_type_counts or {}
        self.total_continuations = sum(self.continuation_counts.values()) if self.continuation_counts else 1
    
    def smooth(self, count: int, context_count: int, 
               word: str = None, context: Tuple = None, **kwargs) -> float:
        if context_count == 0:
            # Fall back to continuation probability
            if word and word in self.continuation_counts:
                return self.continuation_counts[word] / self.total_continuations
            return 1 / self.vocab_size
        
        # First term: discounted probability
        first_term = max(count - self.discount, 0) / context_count
        
        # Interpolation weight λ
        num_types = self.context_type_counts.get(context, 1)
        lambda_weight = (self.discount * num_types) / context_count
        
        # Continuation probability for the word
        if word and word in self.continuation_counts:
            p_continuation = self.continuation_counts[word] / self.total_continuations
        else:
            p_continuation = 1 / self.vocab_size
        
        return first_term + lambda_weight * p_continuation


def get_smoother(method: SmoothingMethod, vocab_size: int, **kwargs) -> Smoother:
    """Factory function to create the appropriate smoother."""
    if method == SmoothingMethod.NONE:
        return NoSmoothing(vocab_size)
    elif method == SmoothingMethod.LAPLACE:
        return LaplaceSmoothing(vocab_size)
    elif method == SmoothingMethod.ADD_K:
        k = kwargs.get('k', 0.5)
        return AddKSmoothing(vocab_size, k=k)
    elif method == SmoothingMethod.GOOD_TURING:
        ngram_counts = kwargs.get('ngram_counts', Counter())
        return GoodTuringSmoothing(vocab_size, ngram_counts)
    elif method == SmoothingMethod.KNESER_NEY:
        return KneserNeySmoothing(
            vocab_size,
            discount=kwargs.get('discount', 0.75),
            continuation_counts=kwargs.get('continuation_counts'),
            context_type_counts=kwargs.get('context_type_counts')
        )
    else:
        raise ValueError(f"Unknown smoothing method: {method}")
