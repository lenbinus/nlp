"""
Base Tokenizer Class

Defines the interface for all tokenizer implementations.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field
import json
import pickle
from pathlib import Path


@dataclass
class TokenizationStep:
    """Records a single step in the tokenization process for visualization."""
    step_type: str  # 'split', 'merge', 'lookup', etc.
    description: str
    tokens: List[str]
    details: Dict = field(default_factory=dict)


@dataclass 
class TrainingStep:
    """Records a single step in vocabulary training for visualization."""
    iteration: int
    action: str  # 'merge', 'split', 'prune', etc.
    description: str
    vocab_size: int
    details: Dict = field(default_factory=dict)


class BaseTokenizer(ABC):
    """
    Abstract base class for tokenizers.
    
    All tokenizer implementations should inherit from this class
    and implement the required methods.
    """
    
    def __init__(self, vocab_size: int = 1000):
        """
        Initialize the tokenizer.
        
        Args:
            vocab_size: Target vocabulary size
        """
        self.vocab_size = vocab_size
        self.vocab: Dict[str, int] = {}  # token -> id
        self.inverse_vocab: Dict[int, str] = {}  # id -> token
        self.is_trained = False
        self.training_history: List[TrainingStep] = []
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Return the name of the tokenizer."""
        pass
    
    @property
    @abstractmethod
    def description(self) -> str:
        """Return a description of how the tokenizer works."""
        pass
    
    @abstractmethod
    def train(self, texts: List[str], verbose: bool = False) -> List[TrainingStep]:
        """
        Train the tokenizer on a corpus.
        
        Args:
            texts: List of training texts
            verbose: Whether to print progress
        
        Returns:
            List of training steps for visualization
        """
        pass
    
    @abstractmethod
    def tokenize(self, text: str, return_steps: bool = False) -> List[str] | Tuple[List[str], List[TokenizationStep]]:
        """
        Tokenize a string into subword tokens.
        
        Args:
            text: Input text to tokenize
            return_steps: If True, also return tokenization steps
        
        Returns:
            List of tokens, optionally with step history
        """
        pass
    
    @abstractmethod
    def detokenize(self, tokens: List[str]) -> str:
        """
        Convert tokens back to text.
        
        Args:
            tokens: List of tokens
        
        Returns:
            Reconstructed text
        """
        pass
    
    def encode(self, text: str) -> List[int]:
        """
        Tokenize and convert to token IDs.
        
        Args:
            text: Input text
        
        Returns:
            List of token IDs
        """
        tokens = self.tokenize(text)
        return [self.vocab.get(t, self.vocab.get('<unk>', 0)) for t in tokens]
    
    def decode(self, ids: List[int]) -> str:
        """
        Convert token IDs back to text.
        
        Args:
            ids: List of token IDs
        
        Returns:
            Reconstructed text
        """
        tokens = [self.inverse_vocab.get(i, '<unk>') for i in ids]
        return self.detokenize(tokens)
    
    def get_vocab(self) -> Dict[str, int]:
        """Return the vocabulary dictionary."""
        return self.vocab.copy()
    
    def save(self, path: str) -> None:
        """Save tokenizer to file."""
        path = Path(path)
        data = {
            'name': self.name,
            'vocab_size': self.vocab_size,
            'vocab': self.vocab,
            'is_trained': self.is_trained,
            'training_history': [
                {'iteration': s.iteration, 'action': s.action, 
                 'description': s.description, 'vocab_size': s.vocab_size,
                 'details': s.details}
                for s in self.training_history
            ]
        }
        with open(path, 'wb') as f:
            pickle.dump(data, f)
    
    def _build_inverse_vocab(self) -> None:
        """Build inverse vocabulary mapping."""
        self.inverse_vocab = {v: k for k, v in self.vocab.items()}
