import logging
import sys
import wandb

def setup_logger(name: str = "bsort", level: int = logging.INFO) -> logging.Logger:
    """Configures and returns a logger."""
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        
    return logger

def init_wandb(project_name: str = "bottlecap-classifier", config: dict = None):
    """Initializes a Weights & Biases run."""
    wandb.init(project=project_name, config=config)
