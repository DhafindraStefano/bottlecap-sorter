import logging

import click

from bsort.inference import run_inference
from bsort.train import train as train_pipeline
from bsort.utils.config import load_config
from bsort.utils.logging import setup_logger


@click.group()
@click.option("--verbose", is_flag=True, help="Enable verbose logging.")
def main(verbose):
    """
    Bottlecap Sorter CLI

    Args:
        verbose (bool): Enable verbose logging.
    """
    level = logging.DEBUG if verbose else logging.INFO
    setup_logger(level=level)


@main.command()
@click.option("--config", default="configs/settings.yaml", help="Path to config file.")
@click.option("--data-dir", help="Path to processed dataset directory.")
@click.option("--epochs", type=int, help="Number of epochs.")
@click.option("--test-size", type=float, help="Validation set size ratio.")
@click.option(
    "--wandb", "use_wandb", is_flag=True, help="Enable Weights & Biases logging."
)
def train(config, data_dir, epochs, test_size, use_wandb):
    """
    Run the training/evaluation pipeline.

    Args:
        config (str): Path to config file.
        data_dir (str): Path to processed dataset directory.
        epochs (int): Number of epochs.
        test_size (float): Validation set size ratio.
        use_wandb (bool): Enable Weights & Biases logging.
    """
    # Load config
    cfg = load_config(config)

    # CLI args override config
    data_cfg = cfg.get("data", {})
    train_cfg = cfg.get("train", {})
    log_cfg = cfg.get("logging", {})

    final_data_dir = data_dir or data_cfg.get("data_dir", "dataset/processed")
    # final_epochs = epochs or train_cfg.get("epochs", 10)
    final_test_size = test_size or data_cfg.get("test_size", 0.2)
    final_use_wandb = use_wandb or log_cfg.get("use_wandb", False)

    train_pipeline(
        data_dir=final_data_dir, test_size=final_test_size, use_wandb=final_use_wandb
    )


@main.command()
@click.option("--config", default="configs/settings.yaml", help="Path to config file.")
@click.option("--image-path", required=True, help="Path to image for inference.")
def infer(config, image_path): # pylint: disable=unused-argument
    """
    Run inference on a single image.

    Args:
        config (str): Path to config file.
        image_path (str): Path to image for inference.
    """
    # Config might not be strictly needed for inference yet, but good to have for future model paths etc.
    # cfg = load_config(config)
    run_inference(image_path)


if __name__ == "__main__":
    main() # pylint: disable=no-value-for-parameter
