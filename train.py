#!/usr/bin/env python3
"""
N-gram Language Model Training Script

Train an n-gram model on the Brown corpus with beautiful terminal output.

Usage:
    python train.py --n 3 --smoothing laplace --save model.pkl
    python train.py --n 2 --smoothing kneser_ney --categories news fiction
    python train.py --interactive  # Run interactive demo after training
"""

import argparse
import sys
from pathlib import Path

from ngram import NGramModel, SmoothingMethod
from ngram.training import train_model_cli, interactive_demo
from ngram.corpus import get_brown_categories, load_brown_corpus


def main():
    parser = argparse.ArgumentParser(
        description="Train an n-gram language model on the Brown corpus",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --n 3 --smoothing laplace
  %(prog)s --n 2 --smoothing kneser_ney --categories news fiction
  %(prog)s --n 4 --smoothing good_turing --save models/trigram.pkl
  %(prog)s --interactive  # Interactive demo after training

Available smoothing methods:
  none        - No smoothing (MLE)
  laplace     - Add-one smoothing
  add_k       - Add-k smoothing (k=0.5 by default)
  good_turing - Good-Turing smoothing
  kneser_ney  - Kneser-Ney smoothing (recommended)

Brown corpus categories:
  adventure, belles_lettres, editorial, fiction, government,
  hobbies, humor, learned, lore, mystery, news, religion,
  reviews, romance, science_fiction
        """
    )
    
    parser.add_argument(
        '-n', '--n',
        type=int,
        default=3,
        help='Order of the n-gram model (default: 3 for trigram)'
    )
    
    parser.add_argument(
        '-s', '--smoothing',
        type=str,
        default='laplace',
        choices=['none', 'laplace', 'add_k', 'good_turing', 'kneser_ney'],
        help='Smoothing method (default: laplace)'
    )
    
    parser.add_argument(
        '-c', '--categories',
        type=str,
        nargs='+',
        default=None,
        help='Brown corpus categories to use (default: all)'
    )
    
    parser.add_argument(
        '--min-count',
        type=int,
        default=2,
        help='Minimum word count for vocabulary (default: 2)'
    )
    
    parser.add_argument(
        '--max-vocab',
        type=int,
        default=None,
        help='Maximum vocabulary size (default: unlimited)'
    )
    
    parser.add_argument(
        '--save',
        type=str,
        default=None,
        help='Path to save the trained model'
    )
    
    parser.add_argument(
        '--load',
        type=str,
        default=None,
        help='Path to load a pre-trained model'
    )
    
    parser.add_argument(
        '-i', '--interactive',
        action='store_true',
        help='Run interactive demo after training'
    )
    
    parser.add_argument(
        '--list-categories',
        action='store_true',
        help='List available Brown corpus categories and exit'
    )
    
    args = parser.parse_args()
    
    # List categories and exit
    if args.list_categories:
        print("Available Brown corpus categories:")
        for cat in get_brown_categories():
            print(f"  - {cat}")
        return 0
    
    # Load or train model
    if args.load:
        from rich.console import Console
        console = Console()
        
        with console.status(f"[cyan]Loading model from {args.load}..."):
            model = NGramModel.load(args.load)
        console.print(f"[green]✓[/green] Model loaded from: [bold]{args.load}[/bold]")
    else:
        # Train new model
        model = train_model_cli(
            n=args.n,
            smoothing=args.smoothing,
            categories=args.categories,
            min_count=args.min_count,
            max_vocab_size=args.max_vocab,
            save_path=args.save
        )
    
    # Interactive demo
    if args.interactive:
        interactive_demo(model)
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
