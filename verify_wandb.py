import wandb
import sys

print(f"WandB Version: {wandb.__version__}")
print(f"Python Version: {sys.version}")

try:
    print("Attempting wandb.init()...")
    wandb.init(project="test-project")
    print("Successfully initialized wandb run.")
    wandb.finish()
except Exception as e:
    print(f"Failed to initialize wandb: {e}")
