import click
import logging
from bsort.train import train as train_pipeline
from bsort.inference import run_inference
from bsort.utils.logging import setup_logger

@click.group()
@click.option('--verbose', is_flag=True, help="Enable verbose logging.")
def main(verbose):
    """Bottlecap Sorter CLI"""
    level = logging.DEBUG if verbose else logging.INFO
    setup_logger(level=level)

@main.command()
@click.option('--data-dir', default='dataset/processed', help='Path to processed dataset directory.')
@click.option('--epochs', default=10, help='Number of epochs (placeholder for heuristic).')
@click.option('--test-size', default=0.2, help='Validation set size ratio.')
@click.option('--wandb', 'use_wandb', is_flag=True, help='Enable Weights & Biases logging.')
def train(data_dir, epochs, test_size, use_wandb):
    """Run the training/evaluation pipeline."""
    train_pipeline(data_dir=data_dir, test_size=test_size, use_wandb=use_wandb)

@main.command()
@click.option('--image-path', required=True, help='Path to image for inference.')
def infer(image_path):
    """Run inference on a single image."""
    run_inference(image_path)

if __name__ == '__main__':
    main()
