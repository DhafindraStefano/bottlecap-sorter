import sys
import pathlib

# Add src to path so we can import bsort
sys.path.append(str(pathlib.Path.cwd() / "src"))

from bsort.train import train

if __name__ == "__main__":
    print("Running Core Logic Verification...")
    try:
        # Run training pipeline on the processed dataset
        # We disable wandb for verification to avoid login prompts/errors
        train(data_dir="dataset/processed", test_size=0.2, use_wandb=False)
        print("\nVerification Successful!")
    except Exception as e:
        print(f"\nVerification Failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
