"""
WordPiece Tokenizer

WordPiece is a subword tokenization algorithm used by BERT and related models.
Unlike BPE which merges the most frequent pairs, WordPiece uses a likelihood-based
scoring to decide which pairs to merge.

Algorithm:
1. Start with character-level vocabulary  
2. Score each potential merge by: freq(ab) / (freq(a) * freq(b))
3. Merge the pair with highest score
4. Repeat until desired vocabulary size

Key difference from BPE: Uses "##" prefix to indicate continuation tokens.
"""

import re
from collections import Counter, defaultdict
from typing import List, Dict, Tuple, Optional
from .base import BaseTokenizer, TokenizationStep, TrainingStep


class WordPieceTokenizer(BaseTokenizer):
    """
    WordPiece Tokenizer
    
    Implements the WordPiece algorithm used by BERT.
    Uses "##" prefix for continuation subwords.
    """
    
    def __init__(self, vocab_size: int = 1000, unk_token: str = '[UNK]',
                 max_word_length: int = 100):
        super().__init__(vocab_size)
        self.unk_token = unk_token
        self.max_word_length = max_word_length
        self.word_freqs: Dict[str, int] = {}
    
    @property
    def name(self) -> str:
        return "WordPiece"
    
    @property
    def description(self) -> str:
        return """WordPiece uses likelihood-based scoring to build vocabulary.

Score(a,b) = freq(ab) / (freq(a) × freq(b))

This favors merging rare pairs that appear together frequently.
Uses "##" prefix for continuation tokens (e.g., "play" + "##ing").
Used by BERT, DistilBERT, and Electra."""
    
    def _get_word_freqs(self, texts: List[str]) -> Dict[str, int]:
        """Get word frequencies from texts."""
        word_freqs = Counter()
        
        for text in texts:
            words = text.lower().split()
            word_freqs.update(words)
        
        return dict(word_freqs)
    
    def _get_alphabet(self, word_freqs: Dict[str, int]) -> set:
        """Get all characters in the corpus."""
        alphabet = set()
        for word in word_freqs.keys():
            for i, char in enumerate(word):
                if i == 0:
                    alphabet.add(char)
                else:
                    alphabet.add(f'##{char}')
        return alphabet
    
    def _split_word(self, word: str) -> List[str]:
        """Split word into WordPiece format."""
        return [word[0]] + [f'##{c}' for c in word[1:]]
    
    def _compute_pair_scores(self, splits: Dict[str, List[str]], 
                            word_freqs: Dict[str, int]) -> Dict[Tuple[str, str], float]:
        """
        Compute scores for all adjacent pairs.
        Score = freq(pair) / (freq(first) * freq(second))
        """
        # Count individual token frequencies
        token_freqs = Counter()
        pair_freqs = Counter()
        
        for word, freq in word_freqs.items():
            split = splits[word]
            for token in split:
                token_freqs[token] += freq
            for i in range(len(split) - 1):
                pair = (split[i], split[i + 1])
                pair_freqs[pair] += freq
        
        # Compute scores
        scores = {}
        for pair, freq in pair_freqs.items():
            score = freq / (token_freqs[pair[0]] * token_freqs[pair[1]])
            scores[pair] = score
        
        return scores
    
    def _merge_pair(self, splits: Dict[str, List[str]], 
                    pair: Tuple[str, str]) -> Dict[str, List[str]]:
        """Merge a pair in all words."""
        new_splits = {}
        
        for word, split in splits.items():
            new_split = []
            i = 0
            while i < len(split):
                if i < len(split) - 1 and split[i] == pair[0] and split[i + 1] == pair[1]:
                    new_split.append(pair[0] + pair[1].replace('##', ''))
                    i += 2
                else:
                    new_split.append(split[i])
                    i += 1
            new_splits[word] = new_split
        
        return new_splits
    
    def train(self, texts: List[str], verbose: bool = False) -> List[TrainingStep]:
        """
        Train WordPiece on a corpus.
        
        Args:
            texts: Training texts
            verbose: Print progress
        
        Returns:
            Training history for visualization
        """
        self.training_history = []
        
        # Get word frequencies
        self.word_freqs = self._get_word_freqs(texts)
        
        # Initialize with special tokens
        self.vocab = {
            '[PAD]': 0, '[UNK]': 1, '[CLS]': 2, '[SEP]': 3, '[MASK]': 4
        }
        
        # Add alphabet
        alphabet = sorted(self._get_alphabet(self.word_freqs))
        for char in alphabet:
            if char not in self.vocab:
                self.vocab[char] = len(self.vocab)
        
        self.training_history.append(TrainingStep(
            iteration=0,
            action='initialize',
            description=f'Initialized with {len(alphabet)} character tokens',
            vocab_size=len(self.vocab),
            details={'alphabet_size': len(alphabet)}
        ))
        
        if verbose:
            print(f"Initial vocab size: {len(self.vocab)}")
        
        # Create initial splits
        splits = {word: self._split_word(word) for word in self.word_freqs}
        
        # Iteratively merge pairs
        iteration = 0
        while len(self.vocab) < self.vocab_size:
            # Compute pair scores
            scores = self._compute_pair_scores(splits, self.word_freqs)
            
            if not scores:
                break
            
            # Find best pair (highest score)
            best_pair = max(scores, key=scores.get)
            best_score = scores[best_pair]
            
            # Create new token
            new_token = best_pair[0] + best_pair[1].replace('##', '')
            self.vocab[new_token] = len(self.vocab)
            
            # Merge pair
            splits = self._merge_pair(splits, best_pair)
            
            iteration += 1
            
            self.training_history.append(TrainingStep(
                iteration=iteration,
                action='merge',
                description=f'Merged "{best_pair[0]}" + "{best_pair[1]}" → "{new_token}" (score: {best_score:.4f})',
                vocab_size=len(self.vocab),
                details={
                    'pair': best_pair,
                    'new_token': new_token,
                    'score': best_score
                }
            ))
            
            if verbose and iteration % 100 == 0:
                print(f"Iteration {iteration}: vocab size = {len(self.vocab)}, "
                      f"merged {best_pair} (score={best_score:.4f})")
        
        self._build_inverse_vocab()
        self.is_trained = True
        
        if verbose:
            print(f"Training complete. Final vocab size: {len(self.vocab)}")
        
        return self.training_history
    
    def tokenize(self, text: str, return_steps: bool = False):
        """
        Tokenize text using WordPiece algorithm.
        
        Uses greedy longest-match-first strategy.
        """
        if not self.is_trained:
            raise RuntimeError("Tokenizer must be trained first")
        
        steps = []
        output_tokens = []
        
        words = text.lower().split()
        
        for word in words:
            if len(word) > self.max_word_length:
                output_tokens.append(self.unk_token)
                continue
            
            tokens = []
            start = 0
            
            if return_steps:
                steps.append(TokenizationStep(
                    step_type='start_word',
                    description=f'Processing word: "{word}"',
                    tokens=list(word),
                    details={'word': word}
                ))
            
            while start < len(word):
                end = len(word)
                found = False
                
                while start < end:
                    substr = word[start:end]
                    if start > 0:
                        substr = '##' + substr
                    
                    if substr in self.vocab:
                        tokens.append(substr)
                        found = True
                        
                        if return_steps:
                            steps.append(TokenizationStep(
                                step_type='match',
                                description=f'Found token: "{substr}"',
                                tokens=tokens.copy(),
                                details={'matched': substr, 'position': (start, end)}
                            ))
                        break
                    
                    end -= 1
                
                if not found:
                    tokens.append(self.unk_token)
                    if return_steps:
                        steps.append(TokenizationStep(
                            step_type='unknown',
                            description=f'Unknown character at position {start}',
                            tokens=tokens.copy(),
                            details={'position': start}
                        ))
                    start += 1
                else:
                    start = end
            
            output_tokens.extend(tokens)
        
        if return_steps:
            steps.append(TokenizationStep(
                step_type='final',
                description='Final tokenization',
                tokens=output_tokens,
                details={'num_tokens': len(output_tokens)}
            ))
            return output_tokens, steps
        
        return output_tokens
    
    def detokenize(self, tokens: List[str]) -> str:
        """Convert tokens back to text."""
        text = ''
        for i, token in enumerate(tokens):
            if token.startswith('##'):
                text += token[2:]
            elif i > 0:
                text += ' ' + token
            else:
                text += token
        return text
