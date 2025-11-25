import pathlib
import cv2
import logging
from typing import Optional
from bsort.model import ColorThresholdModel
from bsort.data import load_image

logger = logging.getLogger("bsort")

def run_inference(image_path: str):
    """
    Runs inference on a single image.
    
    Args:
        image_path: Path to the input image.
    """
    path = pathlib.Path(image_path)
    if not path.exists():
        logger.error(f"Image not found: {image_path}")
        return

    # Load Image
    img = load_image(path)
    if img is None:
        logger.error(f"Failed to load image: {image_path}")
        return

    # Initialize Model
    model = ColorThresholdModel()

    # Predict
    # Note: We assume the input image IS the ROI (cropped bottlecap)
    # because we don't have an object detector yet.
    pred_cls = model.predict(img)
    
    class_names = {0: 'Light Blue', 1: 'Dark Blue', 2: 'Others'}
    result = class_names.get(pred_cls, "Unknown")
    
    print(f"Image: {path.name}")
    print(f"Prediction: {result}")
    
    return result
