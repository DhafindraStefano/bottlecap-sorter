import pytest
import numpy as np
import pathlib
from bsort.data import load_label, extract_roi

def test_load_label(tmp_path):
    """Test loading YOLO format labels from a file."""
    # Create a dummy label file
    label_file = tmp_path / "test_label.txt"
    content = "0 0.5 0.5 0.2 0.2\n1 0.1 0.1 0.1 0.1"
    label_file.write_text(content)
    
    labels = load_label(label_file)
    
    assert len(labels) == 2
    # Check first label
    assert labels[0] == (0, 0.5, 0.5, 0.2, 0.2)
    # Check second label
    assert labels[1] == (1, 0.1, 0.1, 0.1, 0.1)

def test_load_label_nonexistent():
    """Test loading a non-existent label file."""
    path = pathlib.Path("nonexistent.txt")
    labels = load_label(path)
    assert labels == []

def test_extract_roi():
    """Test extracting ROI from an image."""
    # Create a dummy 100x100 image
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    
    # Label: Class 0, Center (50, 50), Width 20, Height 20
    # Expected ROI: x1=40, y1=40, x2=60, y2=60 (20x20)
    label = (0, 0.5, 0.5, 0.2, 0.2)
    
    roi = extract_roi(img, label)
    
    assert roi is not None
    assert roi.shape == (20, 20, 3)

def test_extract_roi_out_of_bounds():
    """Test extracting ROI that is completely out of bounds (should return None or handle gracefully)."""
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    # Center at 1.5, 1.5 (way outside)
    label = (0, 1.5, 1.5, 0.2, 0.2)
    
    roi = extract_roi(img, label)
    
    # Depending on implementation, it might return None or empty array
    # My implementation returns None if x1>=x2 or y1>=y2 after clipping
    assert roi is None
