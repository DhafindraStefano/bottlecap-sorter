# Bottlecap Sorter - ML Pipeline

This project implements a Machine Learning pipeline for sorting bottlecaps by color (Light Blue, Dark Blue, Others). It includes data loading, model training, evaluation, and a CLI for easy interaction.

## Project Structure

- `src/bsort`: Core package source code.
    - `data.py`: Data loading and preprocessing.
    - `model.py`: Model architecture definition.
    - `train.py`: Training loop implementation.
    - `cli.py`: Command-line interface entry point.
    - `utils/`: Utility functions for logging, metrics, and visualization.
- `notebooks/`: Jupyter notebooks for experimentation.
- `tests/`: Unit tests.
- `dataset/`: Data directory.

## Installation

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd bottlecap-sorter
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python3 -m venv .venv
    source .venv/bin/activate
    ```

3.  **Install the package in editable mode:**
    ```bash
    pip install -e .
    ```

## Performance

The inference speed was benchmarked on a standard machine (M1/M2/Intel Mac).

- **Average Inference Time**: ~0.024 ms/frame
- **Target**: < 10 ms/frame

The heuristic model is extremely lightweight and easily meets the real-time requirements for edge devices like Raspberry Pi 5.

## Usage

The project provides a CLI tool `bsort`.

### Training
To train the model:
```bash
bsort train --data-dir dataset/processed --epochs 10
```

### Inference
To run inference on an image:
```bash
bsort infer --image-path path/to/image.jpg
```

## Development

- **Run tests:** `pytest`
- **Linting:** `flake8 src/bsort`
