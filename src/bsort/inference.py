import logging
import pathlib
from typing import Optional



from bsort.data import load_image
from bsort.model import ColorThresholdModel

logger = logging.getLogger("bsort")


def run_inference(image_path: str):
    """
    Runs inference on a single image.

    Args:
        image_path (str): Path to the input image.

    Returns:
        Optional[str]: The predicted class name (Light Blue, Dark Blue, Others),
        or None if image loading fails.
    """
    path = pathlib.Path(image_path)
    if not path.exists():
        logger.error("Image not found: %s", image_path)
        return

    # Load Image
    img = load_image(path)
    if img is None:
        logger.error("Failed to load image: %s", image_path)
        return

    # Initialize Model
    model = ColorThresholdModel()

    # Predict
    # Note: We assume the input image IS the ROI (cropped bottlecap)
    # because we don't have an object detector yet.
    pred_cls = model.predict(img)

    class_names = {0: "Light Blue", 1: "Dark Blue", 2: "Others"}
    result = class_names.get(pred_cls, "Unknown")

    print(f"Image: {path.name}")
    print(f"Prediction: {result}")

    return result
