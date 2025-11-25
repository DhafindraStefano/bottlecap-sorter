import cv2
import pathlib
import numpy as np
from sklearn.model_selection import train_test_split
from typing import List, Tuple, Optional

def load_image(path: pathlib.Path) -> Optional[np.ndarray]:
    """Loads an image from the given path."""
    return cv2.imread(str(path))

def load_label(path: pathlib.Path) -> List[Tuple[int, float, float, float, float]]:
    """
    Parses a YOLO format label file.
    
    Returns:
        List of tuples: (class_id, x_center, y_center, width, height)
    """
    if not path.exists():
        return []
        
    with open(path, 'r') as f:
        lines = f.readlines()
        
    labels = []
    for line in lines:
        try:
            parts = list(map(float, line.strip().split()))
            if len(parts) >= 5:
                cls_id = int(parts[0])
                xc, yc, w, h = parts[1:5]
                labels.append((cls_id, xc, yc, w, h))
        except ValueError:
            continue
            
    return labels

def get_all_image_paths(data_dir: pathlib.Path) -> List[pathlib.Path]:
    """Returns a list of all .jpg images in the directory."""
    return list(data_dir.glob('*.jpg'))

def split_dataset(image_paths: List[pathlib.Path], test_size: float = 0.2, random_state: int = 42):
    """Splits the dataset into training and validation sets."""
    if not image_paths:
        return [], []
    return train_test_split(image_paths, test_size=test_size, random_state=random_state)

def extract_roi(image: np.ndarray, label: Tuple[int, float, float, float, float]) -> Optional[np.ndarray]:
    """Extracts the Region of Interest (ROI) from the image based on the label."""
    img_h, img_w = image.shape[:2]
    _, xc, yc, w, h = label
    
    x1 = int((xc - w/2) * img_w)
    y1 = int((yc - h/2) * img_h)
    x2 = int((xc + w/2) * img_w)
    y2 = int((yc + h/2) * img_h)
    
    # Clip to image bounds
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(img_w, x2), min(img_h, y2)
    
    if x1 >= x2 or y1 >= y2:
        return None
        
    return image[y1:y2, x1:x2]
