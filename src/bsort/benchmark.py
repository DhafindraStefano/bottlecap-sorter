import time
import numpy as np
import cv2
from bsort.model import ColorThresholdModel

def benchmark_inference(n_runs: int = 1000):
    """
    Benchmarks the inference speed of the ColorThresholdModel.
    
    Args:
        n_runs: Number of iterations to run.
    """
    # Create a dummy image (100x100)
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    
    model = ColorThresholdModel()
    
    # Warmup
    for _ in range(10):
        model.predict(img)
        
    start_time = time.perf_counter()
    
    for _ in range(n_runs):
        model.predict(img)
        
    end_time = time.perf_counter()
    
    total_time = end_time - start_time
    avg_time_ms = (total_time / n_runs) * 1000
    
    print(f"Benchmark Results:")
    print(f"Total Runs: {n_runs}")
    print(f"Total Time: {total_time:.4f}s")
    print(f"Average Time per Frame: {avg_time_ms:.4f}ms")
    
    return avg_time_ms

if __name__ == "__main__":
    benchmark_inference()
