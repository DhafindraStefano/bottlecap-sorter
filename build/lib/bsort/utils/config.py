import yaml
import pathlib
from typing import Any, Dict

def load_config(config_path: str) -> Dict[str, Any]:
    """
    Loads configuration from a YAML file.

    Args:
        config_path (str): Path to the YAML configuration file.

    Returns:
        Dict[str, Any]: Configuration dictionary.
    """
    path = pathlib.Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(path, 'r') as f:
        config = yaml.safe_load(f)

    return config
