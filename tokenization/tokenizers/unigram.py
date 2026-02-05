"""
Unigram Language Model Tokenizer

Unigram is a subword tokenization algorithm that starts with a large vocabulary
and iteratively removes tokens to maximize the likelihood of the training data.
This is the opposite approach from BPE/WordPiece.

Algorithm:
1. Start with a large vocabulary (all substrings up to max length)
2. For each token, compute loss if removed (how much likelihood decreases)
3. Remove tokens with smallest loss (least impact)
4. Repeat until desired vocabulary size

Used by: SentencePiece, T5, XLNet, ALBERT
"""

import math
from collections import Counter, defaultdict
from typing import List, Dict, Tuple, Optional
from .base import BaseTokenizer, TokenizationStep, TrainingStep


class UnigramTokenizer(BaseTokenizer):
    """
    Unigram Language Model Tokenizer
    
    Implements the Unigram algorithm used by SentencePiece.
    Uses negative log likelihood for scoring.
    """
    
    def __init__(self, vocab_size: int = 1000, max_piece_length: int = 16,
                 unk_token: str = '<unk>', initial_vocab_size: int = None):
        super().__init__(vocab_size)
        self.max_piece_length = max_piece_length
        self.unk_token = unk_token
        self.initial_vocab_size = initial_vocab_size or vocab_size * 4
        self.token_probs: Dict[str, float] = {}  # Log probabilities
    
    @property
    def name(self) -> str:
        return "Unigram"
    
    @property
    def description(self) -> str:
        return """Unigram starts with a LARGE vocabulary and prunes it down.

For each token, compute: Loss = -sum(log P(corpus with token removed))
Remove tokens with smallest loss (least useful).

Uses dynamic programming (Viterbi) for optimal tokenization.
Used by SentencePiece, T5, XLNet, and ALBERT."""
    
    def _get_initial_vocab(self, texts: List[str]) -> Dict[str, int]:
        """Build initial large vocabulary from substrings."""
        substring_freqs = Counter()
        
        for text in texts:
            words = text.lower().split()
            for word in words:
                word = '▁' + word  # SentencePiece space marker
                # Add all substrings
                for start in range(len(word)):
                    for end in range(start + 1, min(start + self.max_piece_length + 1, len(word) + 1)):
                        substring = word[start:end]
                        substring_freqs[substring] += 1
        
        # Keep most frequent substrings
        vocab = dict(substring_freqs.most_common(self.initial_vocab_size))
        return vocab
    
    def _compute_token_probs(self, token_freqs: Dict[str, int]) -> Dict[str, float]:
        """Compute log probabilities for each token."""
        total = sum(token_freqs.values())
        return {token: math.log(freq / total) for token, freq in token_freqs.items()}
    
    def _tokenize_word_viterbi(self, word: str, token_probs: Dict[str, float]) -> Tuple[List[str], float]:
        """
        Find optimal tokenization using Viterbi algorithm.
        
        Returns best tokenization and its log probability.
        """
        n = len(word)
        
        # best[i] = (best_log_prob, best_tokenization) for word[:i]
        best = [(0.0, [])] + [(float('-inf'), []) for _ in range(n)]
        
        for end in range(1, n + 1):
            for start in range(max(0, end - self.max_piece_length), end):
                piece = word[start:end]
                if piece in token_probs:
                    prob = best[start][0] + token_probs[piece]
                    if prob > best[end][0]:
                        best[end] = (prob, best[start][1] + [piece])
        
        return best[n][1], best[n][0]
    
    def _compute_loss(self, words: List[str], word_freqs: Dict[str, int],
                      token_probs: Dict[str, float]) -> float:
        """Compute total negative log likelihood."""
        total_loss = 0.0
        
        for word, freq in word_freqs.items():
            _, log_prob = self._tokenize_word_viterbi(word, token_probs)
            if log_prob > float('-inf'):
                total_loss -= log_prob * freq
        
        return total_loss
    
    def train(self, texts: List[str], verbose: bool = False) -> List[TrainingStep]:
        """
        Train Unigram tokenizer.
        
        Starts with large vocabulary and prunes down.
        """
        self.training_history = []
        
        # Step 1: Build initial large vocabulary
        initial_freqs = self._get_initial_vocab(texts)
        
        # Add special tokens
        self.vocab = {'<pad>': 0, '<unk>': 1, '<s>': 2, '</s>': 3}
        for token in sorted(initial_freqs.keys()):
            if token not in self.vocab:
                self.vocab[token] = len(self.vocab)
        
        current_freqs = initial_freqs.copy()
        
        self.training_history.append(TrainingStep(
            iteration=0,
            action='initialize',
            description=f'Initialized with {len(current_freqs)} substring tokens',
            vocab_size=len(self.vocab),
            details={'initial_size': len(current_freqs)}
        ))
        
        if verbose:
            print(f"Initial vocab size: {len(self.vocab)}")
        
        # Build word list with frequencies
        word_freqs = Counter()
        for text in texts:
            for word in text.lower().split():
                word_freqs['▁' + word] += 1
        
        # Iteratively prune vocabulary
        iteration = 0
        while len(current_freqs) > self.vocab_size - 4:  # Reserve space for special tokens
            # Compute current token probabilities
            token_probs = self._compute_token_probs(current_freqs)
            
            # Compute loss for removing each token
            losses = {}
            for token in list(current_freqs.keys()):
                if len(token) == 1:  # Keep single characters
                    continue
                
                # Try removing this token
                test_freqs = current_freqs.copy()
                del test_freqs[token]
                test_probs = self._compute_token_probs(test_freqs)
                
                # Check if corpus is still tokenizable
                can_tokenize = True
                loss_increase = 0.0
                
                for word, freq in list(word_freqs.items())[:100]:  # Sample for speed
                    tokens, log_prob = self._tokenize_word_viterbi(word, test_probs)
                    if not tokens or log_prob == float('-inf'):
                        can_tokenize = False
                        break
                    
                    orig_tokens, orig_log_prob = self._tokenize_word_viterbi(word, token_probs)
                    loss_increase += (orig_log_prob - log_prob) * freq
                
                if can_tokenize:
                    losses[token] = loss_increase
            
            if not losses:
                break
            
            # Remove tokens with smallest loss (batch removal for speed)
            num_to_remove = max(1, len(current_freqs) // 10)
            tokens_to_remove = sorted(losses.keys(), key=lambda t: losses[t])[:num_to_remove]
            
            for token in tokens_to_remove:
                del current_freqs[token]
                if token in self.vocab:
                    del self.vocab[token]
            
            iteration += 1
            
            if tokens_to_remove:
                self.training_history.append(TrainingStep(
                    iteration=iteration,
                    action='prune',
                    description=f'Removed {len(tokens_to_remove)} tokens with lowest loss',
                    vocab_size=len(self.vocab),
                    details={
                        'removed_sample': tokens_to_remove[:5],
                        'num_removed': len(tokens_to_remove)
                    }
                ))
            
            if verbose and iteration % 10 == 0:
                print(f"Iteration {iteration}: vocab size = {len(self.vocab)}")
        
        # Rebuild final vocabulary with proper indices
        self.vocab = {'<pad>': 0, '<unk>': 1, '<s>': 2, '</s>': 3}
        for token in sorted(current_freqs.keys()):
            self.vocab[token] = len(self.vocab)
        
        # Store final token probabilities
        self.token_probs = self._compute_token_probs(current_freqs)
        
        self._build_inverse_vocab()
        self.is_trained = True
        
        if verbose:
            print(f"Training complete. Final vocab size: {len(self.vocab)}")
        
        return self.training_history
    
    def tokenize(self, text: str, return_steps: bool = False):
        """
        Tokenize text using Viterbi algorithm for optimal tokenization.
        """
        if not self.is_trained:
            raise RuntimeError("Tokenizer must be trained first")
        
        steps = []
        output_tokens = []
        
        words = text.lower().split()
        
        for word in words:
            word_with_space = '▁' + word
            
            if return_steps:
                steps.append(TokenizationStep(
                    step_type='start_word',
                    description=f'Processing: "{word}" → "{word_with_space}"',
                    tokens=[word_with_space],
                    details={'word': word}
                ))
            
            # Use Viterbi to find best tokenization
            tokens, log_prob = self._tokenize_word_viterbi(word_with_space, self.token_probs)
            
            if not tokens:
                tokens = [self.unk_token]
                if return_steps:
                    steps.append(TokenizationStep(
                        step_type='unknown',
                        description=f'Could not tokenize "{word}", using <unk>',
                        tokens=[self.unk_token],
                        details={'word': word}
                    ))
            else:
                if return_steps:
                    prob = math.exp(log_prob)
                    steps.append(TokenizationStep(
                        step_type='viterbi',
                        description=f'Viterbi: {tokens} (prob: {prob:.6f})',
                        tokens=tokens,
                        details={'log_prob': log_prob, 'prob': prob}
                    ))
            
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
        text = ''.join(tokens)
        # Remove ▁ markers and convert to spaces
        text = text.replace('▁', ' ')
        return text.strip()
