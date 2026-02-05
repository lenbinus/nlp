"""
Corpus Loading and Preprocessing

This module handles loading the Brown corpus and preprocessing text
for n-gram model training.
"""

import re
import string
from typing import List, Generator, Optional, Tuple
from pathlib import Path
import nltk
from nltk.corpus import brown


# Special tokens
START_TOKEN = "<s>"
END_TOKEN = "</s>"
UNK_TOKEN = "<UNK>"


def ensure_nltk_data():
    """Download required NLTK data if not present."""
    try:
        nltk.data.find('corpora/brown')
    except LookupError:
        print("Downloading Brown corpus...")
        nltk.download('brown', quiet=True)
    
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        print("Downloading punkt tokenizer...")
        nltk.download('punkt', quiet=True)


def preprocess_text(text: str, lowercase: bool = True, 
                    remove_punctuation: bool = False) -> List[str]:
    """
    Preprocess raw text into a list of tokens.
    
    Args:
        text: Raw input text
        lowercase: Whether to lowercase the text
        remove_punctuation: Whether to remove punctuation
    
    Returns:
        List of preprocessed tokens
    """
    if lowercase:
        text = text.lower()
    
    if remove_punctuation:
        text = text.translate(str.maketrans('', '', string.punctuation))
    
    # Basic tokenization - split on whitespace
    tokens = text.split()
    
    return tokens


def add_sentence_markers(tokens: List[str], n: int) -> List[str]:
    """
    Add start and end markers to a sentence for n-gram training.
    
    Args:
        tokens: List of tokens in the sentence
        n: The n in n-gram (determines number of start markers)
    
    Returns:
        Tokens with start and end markers
    """
    # Add (n-1) start tokens and 1 end token
    return [START_TOKEN] * (n - 1) + tokens + [END_TOKEN]


def load_brown_corpus(categories: Optional[List[str]] = None,
                      lowercase: bool = True,
                      min_sentence_length: int = 3) -> Tuple[List[List[str]], dict]:
    """
    Load the Brown corpus and return preprocessed sentences.
    
    Args:
        categories: Optional list of Brown corpus categories to load
                   (e.g., ['news', 'fiction', 'science_fiction'])
                   If None, loads all categories.
        lowercase: Whether to lowercase the text
        min_sentence_length: Minimum number of words in a sentence
    
    Returns:
        Tuple of (list of sentences as token lists, corpus statistics dict)
    """
    ensure_nltk_data()
    
    # Get sentences from Brown corpus
    if categories:
        sents = brown.sents(categories=categories)
    else:
        sents = brown.sents()
    
    processed_sentences = []
    total_tokens = 0
    
    for sent in sents:
        # Convert to list and optionally lowercase
        tokens = [w.lower() if lowercase else w for w in sent]
        
        if len(tokens) >= min_sentence_length:
            processed_sentences.append(tokens)
            total_tokens += len(tokens)
    
    stats = {
        'num_sentences': len(processed_sentences),
        'total_tokens': total_tokens,
        'categories': categories or brown.categories()
    }
    
    return processed_sentences, stats


def get_brown_categories() -> List[str]:
    """Return list of available Brown corpus categories."""
    ensure_nltk_data()
    return brown.categories()


def build_vocabulary(sentences: List[List[str]], 
                     min_count: int = 1,
                     max_vocab_size: Optional[int] = None) -> Tuple[dict, dict]:
    """
    Build vocabulary from sentences.
    
    Args:
        sentences: List of tokenized sentences
        min_count: Minimum count for a word to be included
        max_vocab_size: Maximum vocabulary size (keeps most frequent)
    
    Returns:
        Tuple of (word_to_idx, idx_to_word) dictionaries
    """
    from collections import Counter
    
    # Count all words
    word_counts = Counter()
    for sent in sentences:
        word_counts.update(sent)
    
    # Filter by min_count
    words = [(w, c) for w, c in word_counts.items() if c >= min_count]
    
    # Sort by frequency (descending)
    words.sort(key=lambda x: x[1], reverse=True)
    
    # Limit vocabulary size
    if max_vocab_size:
        words = words[:max_vocab_size - 3]  # Reserve space for special tokens
    
    # Build mappings (special tokens first)
    word_to_idx = {UNK_TOKEN: 0, START_TOKEN: 1, END_TOKEN: 2}
    idx_to_word = {0: UNK_TOKEN, 1: START_TOKEN, 2: END_TOKEN}
    
    for idx, (word, _) in enumerate(words, start=3):
        word_to_idx[word] = idx
        idx_to_word[idx] = word
    
    return word_to_idx, idx_to_word


def sentences_to_ngrams(sentences: List[List[str]], n: int) -> Generator[Tuple[str, ...], None, None]:
    """
    Generate n-grams from sentences.
    
    Args:
        sentences: List of tokenized sentences
        n: The n in n-gram
    
    Yields:
        N-gram tuples
    """
    for sent in sentences:
        # Add sentence markers
        marked = add_sentence_markers(sent, n)
        
        # Generate n-grams
        for i in range(len(marked) - n + 1):
            yield tuple(marked[i:i + n])
