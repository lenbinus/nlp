"""
Byte Pair Encoding (BPE) Tokenizer

BPE is a subword tokenization algorithm that iteratively merges the most
frequent pair of consecutive tokens. Originally a compression algorithm,
it was adapted for NLP and is used by GPT, RoBERTa, and many other models.

Algorithm:
1. Start with character-level vocabulary
2. Count all adjacent pairs of tokens
3. Merge the most frequent pair into a new token
4. Repeat until desired vocabulary size is reached
"""

import re
from collections import Counter, defaultdict
from typing import List, Dict, Tuple, Optional
from .base import BaseTokenizer, TokenizationStep, TrainingStep


class BPETokenizer(BaseTokenizer):
    """
    Byte Pair Encoding Tokenizer
    
    This implementation follows the original BPE algorithm with some
    modifications for handling word boundaries (using Ġ for spaces).
    """
    
    def __init__(self, vocab_size: int = 1000):
        super().__init__(vocab_size)
        self.merges: List[Tuple[str, str]] = []  # Ordered list of merge operations
        self.word_freqs: Dict[str, int] = {}
    
    @property
    def name(self) -> str:
        return "Byte Pair Encoding (BPE)"
    
    @property
    def description(self) -> str:
        return """BPE iteratively merges the most frequent pair of adjacent tokens.
        
Starting with characters, it builds up a vocabulary of common subwords.
Used by GPT-2, GPT-3, RoBERTa, and many other models.

Key insight: Common words become single tokens, rare words are broken into subwords."""
    
    def _get_word_freqs(self, texts: List[str]) -> Dict[str, int]:
        """
        Get word frequencies from texts.
        Words are split into characters with </w> marking word end.
        """
        word_freqs = Counter()
        
        for text in texts:
            # Simple word tokenization
            words = text.lower().split()
            for word in words:
                # Add space prefix (Ġ convention) and split into chars
                word_chars = ' '.join(list('Ġ' + word))
                word_freqs[word_chars] += 1
        
        return dict(word_freqs)
    
    def _get_pair_freqs(self, word_freqs: Dict[str, int]) -> Counter:
        """Count frequencies of adjacent pairs."""
        pairs = Counter()
        
        for word, freq in word_freqs.items():
            symbols = word.split()
            for i in range(len(symbols) - 1):
                pair = (symbols[i], symbols[i + 1])
                pairs[pair] += freq
        
        return pairs
    
    def _merge_pair(self, word_freqs: Dict[str, int], 
                    pair: Tuple[str, str]) -> Dict[str, int]:
        """Merge all occurrences of a pair in the vocabulary."""
        new_word_freqs = {}
        bigram = ' '.join(pair)
        replacement = ''.join(pair)
        
        for word, freq in word_freqs.items():
            new_word = word.replace(bigram, replacement)
            new_word_freqs[new_word] = freq
        
        return new_word_freqs
    
    def train(self, texts: List[str], verbose: bool = False) -> List[TrainingStep]:
        """
        Train BPE on a corpus of texts.
        
        Args:
            texts: Training texts
            verbose: Print progress
        
        Returns:
            Training history for visualization
        """
        self.training_history = []
        
        # Step 1: Get word frequencies
        self.word_freqs = self._get_word_freqs(texts)
        
        # Initialize vocabulary with characters
        self.vocab = {'<pad>': 0, '<unk>': 1, '<s>': 2, '</s>': 3}
        chars = set()
        for word in self.word_freqs.keys():
            chars.update(word.split())
        
        for char in sorted(chars):
            self.vocab[char] = len(self.vocab)
        
        self.training_history.append(TrainingStep(
            iteration=0,
            action='initialize',
            description=f'Initialized with {len(chars)} character tokens',
            vocab_size=len(self.vocab),
            details={'characters': sorted(chars)}
        ))
        
        if verbose:
            print(f"Initial vocab size: {len(self.vocab)}")
        
        # Step 2: Iteratively merge pairs
        iteration = 0
        while len(self.vocab) < self.vocab_size:
            # Get pair frequencies
            pair_freqs = self._get_pair_freqs(self.word_freqs)
            
            if not pair_freqs:
                break
            
            # Find most frequent pair
            best_pair = pair_freqs.most_common(1)[0]
            pair, freq = best_pair
            
            if freq < 2:
                break
            
            # Merge the pair
            self.merges.append(pair)
            new_token = ''.join(pair)
            self.vocab[new_token] = len(self.vocab)
            
            # Update word frequencies
            self.word_freqs = self._merge_pair(self.word_freqs, pair)
            
            iteration += 1
            
            self.training_history.append(TrainingStep(
                iteration=iteration,
                action='merge',
                description=f'Merged "{pair[0]}" + "{pair[1]}" → "{new_token}" (freq: {freq})',
                vocab_size=len(self.vocab),
                details={
                    'pair': pair,
                    'new_token': new_token,
                    'frequency': freq
                }
            ))
            
            if verbose and iteration % 100 == 0:
                print(f"Iteration {iteration}: vocab size = {len(self.vocab)}, "
                      f"merged {pair} (freq={freq})")
        
        self._build_inverse_vocab()
        self.is_trained = True
        
        if verbose:
            print(f"Training complete. Final vocab size: {len(self.vocab)}")
        
        return self.training_history
    
    def tokenize(self, text: str, return_steps: bool = False):
        """
        Tokenize text using learned BPE merges.
        
        Args:
            text: Input text
            return_steps: Return tokenization steps for visualization
        
        Returns:
            List of tokens (and optionally steps)
        """
        if not self.is_trained:
            raise RuntimeError("Tokenizer must be trained first")
        
        steps = []
        tokens = []
        
        # Split into words
        words = text.lower().split()
        
        for word in words:
            # Start with characters
            word_tokens = list('Ġ' + word)
            
            if return_steps:
                steps.append(TokenizationStep(
                    step_type='split',
                    description=f'Split "{word}" into characters',
                    tokens=word_tokens.copy(),
                    details={'word': word}
                ))
            
            # Apply merges in order
            for merge_idx, (a, b) in enumerate(self.merges):
                i = 0
                new_tokens = []
                merged = False
                
                while i < len(word_tokens):
                    if i < len(word_tokens) - 1 and word_tokens[i] == a and word_tokens[i + 1] == b:
                        new_tokens.append(a + b)
                        merged = True
                        i += 2
                    else:
                        new_tokens.append(word_tokens[i])
                        i += 1
                
                if merged:
                    word_tokens = new_tokens
                    if return_steps:
                        steps.append(TokenizationStep(
                            step_type='merge',
                            description=f'Applied merge: "{a}" + "{b}" → "{a+b}"',
                            tokens=word_tokens.copy(),
                            details={'merge': (a, b), 'merge_index': merge_idx}
                        ))
            
            tokens.extend(word_tokens)
        
        if return_steps:
            steps.append(TokenizationStep(
                step_type='final',
                description='Final tokenization',
                tokens=tokens,
                details={'num_tokens': len(tokens)}
            ))
            return tokens, steps
        
        return tokens
    
    def detokenize(self, tokens: List[str]) -> str:
        """Convert tokens back to text."""
        text = ''.join(tokens)
        # Remove Ġ markers and add spaces
        text = text.replace('Ġ', ' ')
        return text.strip()
