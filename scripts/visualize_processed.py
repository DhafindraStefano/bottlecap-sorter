import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import random
import os

# Define paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = PROJECT_ROOT / 'dataset'
PROCESSED_DIR = DATASET_DIR / 'processed'
ARTIFACTS_DIR = PROJECT_ROOT / 'artifacts'

def load_image(path):
    """Load an image from path and convert to RGB."""
    img = cv2.imread(str(path))
    if img is None:
        raise ValueError(f"Could not load image: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def load_label(path):
    """Load YOLO labels from text file."""
    labels = []
    if not path.exists():
        return labels
    
    with open(path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 5:
                # class_id, x_center, y_center, width, height
                labels.append([int(parts[0])] + [float(x) for x in parts[1:]])
    return labels

def draw_bbox(image, labels, class_names=None):
    """Draw bounding boxes on the image."""
    img_h, img_w = image.shape[:2]
    img_vis = image.copy()
    
    # Colors for different classes (RGB)
    colors = {
        0: (0, 255, 255),    # Light Blue (Cyan)
        1: (0, 0, 255),      # Dark Blue (Blue)
        2: (255, 0, 0)       # Others (Red)
    }
    
    for label in labels:
        cls_id, xc, yc, w, h = label
        
        # Convert normalized coordinates to pixel coordinates
        x1 = int((xc - w/2) * img_w)
        y1 = int((yc - h/2) * img_h)
        x2 = int((xc + w/2) * img_w)
        y2 = int((yc + h/2) * img_h)
        
        # Clamp coordinates
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(img_w, x2), min(img_h, y2)
        
        color = colors.get(cls_id, (0, 255, 0))
        
        # Draw rectangle
        cv2.rectangle(img_vis, (x1, y1), (x2, y2), color, 2)
        
        # Draw label
        if class_names:
            name = class_names.get(cls_id, str(cls_id))
            text = f"{name}"
            (text_w, text_h), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(img_vis, (x1, y1 - text_h - baseline), (x1 + text_w, y1), color, -1)
            cv2.putText(img_vis, text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
    return img_vis

def visualize_processed_images(num_images=5):
    """Visualize random processed images with their labels."""
    image_files = list(PROCESSED_DIR.glob('*.jpg'))
    
    if not image_files:
        print(f"No images found in {PROCESSED_DIR}")
        return

    # Select random images
    selected_files = random.sample(image_files, min(num_images, len(image_files)))
    
    class_names = {0: 'Light Blue', 1: 'Dark Blue', 2: 'Others'}
    
    plt.figure(figsize=(15, 5 * len(selected_files)))
    
    for i, img_path in enumerate(selected_files):
        img = load_image(img_path)
        label_path = img_path.with_suffix('.txt')
        labels = load_label(label_path)
        
        img_vis = draw_bbox(img, labels, class_names)
        
        plt.subplot(len(selected_files), 1, i + 1)
        plt.imshow(img_vis)
        plt.title(f"Image: {img_path.name}")
        plt.axis('off')
        
    output_path = ARTIFACTS_DIR / 'processed_visualization.png'
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"Visualization saved to {output_path}")

if __name__ == "__main__":
    visualize_processed_images()
