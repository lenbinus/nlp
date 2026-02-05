"""
Training Module with Rich Terminal UI

This module provides training functionality with beautiful terminal
progress bars and status displays using the Rich library.
"""

import time
from typing import Optional, List, Dict
from rich.console import Console
from rich.progress import (
    Progress, SpinnerColumn, TextColumn, BarColumn,
    TaskProgressColumn, TimeElapsedColumn, TimeRemainingColumn
)
from rich.panel import Panel
from rich.table import Table
from rich.live import Live
from rich.layout import Layout
from rich import box

from .model import NGramModel
from .smoothing import SmoothingMethod
from .corpus import load_brown_corpus, get_brown_categories


console = Console()


class TrainingProgress:
    """Manages training progress display."""
    
    def __init__(self):
        self.current_stage = ""
        self.current_progress = 0
        self.total_progress = 0
        self.stats = {}
    
    def update(self, current: int, total: int, stage: str = ""):
        self.current_progress = current
        self.total_progress = total
        if stage:
            self.current_stage = stage


def create_stats_table(stats: Dict) -> Table:
    """Create a Rich table displaying training statistics."""
    table = Table(box=box.ROUNDED, show_header=True, header_style="bold cyan")
    table.add_column("Metric", style="green")
    table.add_column("Value", style="yellow", justify="right")
    
    for key, value in stats.items():
        # Format the key nicely
        display_key = key.replace('_', ' ').title()
        
        # Format the value
        if isinstance(value, float):
            display_value = f"{value:,.4f}"
        elif isinstance(value, int):
            display_value = f"{value:,}"
        elif isinstance(value, list):
            display_value = f"{len(value)} items"
        else:
            display_value = str(value)
        
        table.add_row(display_key, display_value)
    
    return table


def train_model_cli(
    n: int = 3,
    smoothing: str = "laplace",
    categories: Optional[List[str]] = None,
    min_count: int = 2,
    max_vocab_size: Optional[int] = None,
    save_path: Optional[str] = None
) -> NGramModel:
    """
    Train an n-gram model with beautiful terminal output.
    
    Args:
        n: Order of the n-gram model
        smoothing: Smoothing method name
        categories: Brown corpus categories to use
        min_count: Minimum word count for vocabulary
        max_vocab_size: Maximum vocabulary size
        save_path: Path to save the trained model
    
    Returns:
        Trained NGramModel
    """
    # Map smoothing name to enum
    smoothing_map = {
        'none': SmoothingMethod.NONE,
        'laplace': SmoothingMethod.LAPLACE,
        'add_k': SmoothingMethod.ADD_K,
        'good_turing': SmoothingMethod.GOOD_TURING,
        'kneser_ney': SmoothingMethod.KNESER_NEY
    }
    
    smoothing_method = smoothing_map.get(smoothing.lower(), SmoothingMethod.LAPLACE)
    
    # Print header
    console.print()
    console.print(Panel.fit(
        "[bold blue]N-gram Language Model Training[/bold blue]",
        border_style="blue"
    ))
    console.print()
    
    # Print configuration
    config_table = Table(box=box.SIMPLE, show_header=False)
    config_table.add_column("Setting", style="cyan")
    config_table.add_column("Value", style="white")
    config_table.add_row("Model Order (n)", str(n))
    config_table.add_row("Smoothing Method", smoothing_method.value)
    config_table.add_row("Categories", ", ".join(categories) if categories else "All")
    config_table.add_row("Min Word Count", str(min_count))
    config_table.add_row("Max Vocab Size", str(max_vocab_size) if max_vocab_size else "Unlimited")
    
    console.print(Panel(config_table, title="[bold]Configuration[/bold]", border_style="green"))
    console.print()
    
    # Load corpus
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        console=console
    ) as progress:
        
        # Stage 1: Load corpus
        task = progress.add_task("[cyan]Loading Brown corpus...", total=None)
        sentences, corpus_stats = load_brown_corpus(categories=categories)
        progress.update(task, completed=100, total=100)
        progress.remove_task(task)
        
        console.print(f"[green]✓[/green] Loaded {corpus_stats['num_sentences']:,} sentences "
                     f"({corpus_stats['total_tokens']:,} tokens)")
        console.print()
        
        # Create model
        model = NGramModel(n=n, smoothing=smoothing_method)
        
        # Stage 2: Train with progress
        train_task = progress.add_task(
            "[cyan]Training model...", 
            total=len(sentences)
        )
        
        def update_progress(current, total, stage=""):
            progress.update(train_task, completed=current, description=f"[cyan]{stage}")
        
        # Train
        stats = model.train(
            sentences,
            min_count=min_count,
            max_vocab_size=max_vocab_size,
            progress_callback=update_progress
        )
        
        progress.remove_task(train_task)
    
    console.print()
    console.print("[green]✓[/green] Training complete!")
    console.print()
    
    # Display statistics
    console.print(Panel(
        create_stats_table(stats),
        title="[bold]Training Statistics[/bold]",
        border_style="yellow"
    ))
    
    # Save model if path provided
    if save_path:
        console.print()
        with console.status("[cyan]Saving model..."):
            model.save(save_path)
        console.print(f"[green]✓[/green] Model saved to: [bold]{save_path}[/bold]")
    
    # Show some examples
    console.print()
    console.print(Panel.fit("[bold]Sample Predictions[/bold]", border_style="magenta"))
    
    # Get top words and show predictions
    top_words = model.get_top_words(10)
    
    for word, count in top_words[:5]:
        if word not in ('<s>', '</s>', '<UNK>'):
            context = ('<s>',) * (n-1) if n > 1 else ()
            if n > 1:
                context = context[:-1] + (word,)
            else:
                context = (word,)
            
            predictions = model.get_next_word_distribution(context, top_k=5)
            pred_str = ", ".join([f"{w} ({p:.3f})" for w, p in predictions])
            console.print(f"  After '[bold]{word}[/bold]': {pred_str}")
    
    console.print()
    
    return model


def evaluate_model_cli(model: NGramModel, test_sentences: List[List[str]]) -> Dict:
    """
    Evaluate a model with terminal output.
    
    Args:
        model: Trained NGramModel
        test_sentences: Test sentences
    
    Returns:
        Dictionary of evaluation metrics
    """
    console.print()
    console.print(Panel.fit("[bold blue]Model Evaluation[/bold blue]", border_style="blue"))
    console.print()
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        console=console
    ) as progress:
        
        task = progress.add_task("[cyan]Computing perplexity...", total=len(test_sentences))
        
        # Calculate perplexity
        perplexity = model.perplexity(test_sentences)
        
        progress.update(task, completed=len(test_sentences))
    
    # Display results
    results = {
        'perplexity': perplexity,
        'test_sentences': len(test_sentences)
    }
    
    console.print()
    console.print(Panel(
        create_stats_table(results),
        title="[bold]Evaluation Results[/bold]",
        border_style="green"
    ))
    
    return results


def interactive_demo(model: NGramModel):
    """Run an interactive demo of the model."""
    console.print()
    console.print(Panel.fit(
        "[bold magenta]Interactive Demo[/bold magenta]\n"
        "Enter a word or phrase to see predictions.\n"
        "Type 'quit' to exit.",
        border_style="magenta"
    ))
    console.print()
    
    n = model.n
    
    while True:
        try:
            user_input = console.input("[bold cyan]Enter context:[/bold cyan] ")
            
            if user_input.lower() in ('quit', 'exit', 'q'):
                break
            
            words = user_input.lower().split()
            
            if len(words) == 0:
                context = ('<s>',) * (n - 1)
            elif len(words) < n - 1:
                context = ('<s>',) * (n - 1 - len(words)) + tuple(words)
            else:
                context = tuple(words[-(n-1):])
            
            predictions = model.get_next_word_distribution(context, top_k=10)
            
            console.print()
            console.print(f"[yellow]Context:[/yellow] {' '.join(context)}")
            console.print("[yellow]Top predictions:[/yellow]")
            
            for i, (word, prob) in enumerate(predictions, 1):
                bar_length = int(prob * 50)
                bar = "█" * bar_length + "░" * (50 - bar_length)
                console.print(f"  {i:2}. {word:15} {bar} {prob:.4f}")
            
            console.print()
            
            # Also generate a sentence
            generated = model.generate(context=list(context), max_length=15)
            console.print(f"[green]Generated:[/green] {' '.join(context)} [bold]{' '.join(generated)}[/bold]")
            console.print()
            
        except KeyboardInterrupt:
            break
    
    console.print("\n[yellow]Goodbye![/yellow]")
