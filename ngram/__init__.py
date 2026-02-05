"""
N-gram Language Model Package

An educational implementation of n-gram language models with various
smoothing techniques and interactive visualizations.
"""

from .model import NGramModel
from .smoothing import SmoothingMethod
from .corpus import load_brown_corpus, preprocess_text

__version__ = "0.1.0"
__all__ = ["NGramModel", "SmoothingMethod", "load_brown_corpus", "preprocess_text"]
