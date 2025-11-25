import pathlib
from typing import List, Optional, Tuple

import cv2
import numpy as np
from sklearn.model_selection import train_test_split


def load_image(path: pathlib.Path) -> Optional[np.ndarray]:
    """
    Loads an image from the given path.

    Args:
        path (pathlib.Path): Path to the image file.

    Returns:
        Optional[np.ndarray]: Loaded image as numpy array, or None if loading fails.
    """
    return cv2.imread(str(path)) # pylint: disable=no-member


def load_label(path: pathlib.Path) -> List[Tuple[int, float, float, float, float]]:
    """
    Parses a YOLO format label file.

    Args:
        path (pathlib.Path): Path to the label file.

    Returns:
        List[Tuple[int, float, float, float, float]]: List of labels, where each label is
        (class_id, x_center, y_center, width, height).
    """
    if not path.exists():
        return []

    with open(path, "r", encoding="utf-8") as f:
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
    """
    Returns a list of all .jpg images in the directory.

    Args:
        data_dir (pathlib.Path): Directory to search for images.

    Returns:
        List[pathlib.Path]: List of paths to .jpg images.
    """
    return list(data_dir.glob("*.jpg"))


def get_primary_class(image_path: pathlib.Path) -> int:
    """
    Helper to get the class ID for stratification.
    Assumes the first label in the file is the primary object.
    """
    label_path = image_path.with_suffix(".txt")
    labels = load_label(label_path)
    if labels:
        return int(labels[0][0])
    return -1


def split_dataset(
    image_paths: List[pathlib.Path], test_size: float = 0.2, random_state: int = 42
):
    """
    Splits the dataset into training and validation sets.

    Args:
        image_paths (List[pathlib.Path]): List of image paths.
        test_size (float): Proportion of the dataset to include in the test split.
        random_state (int): Random seed for reproducibility.

    Returns:
        Tuple[List[pathlib.Path], List[pathlib.Path]]: Training and validation file paths.
    """
    if not image_paths:
        return [], []

    # Try to get labels for stratification
    labels = [get_primary_class(p) for p in image_paths]

    # Filter valid data (images with labels)
    valid_data = [(img, lbl) for img, lbl in zip(image_paths, labels) if lbl != -1]

    if not valid_data:
        # Fallback to random split if no labels found
        return train_test_split(
            image_paths, test_size=test_size, random_state=random_state
        )

    X_valid, y_valid = zip(*valid_data)

    try:
        return train_test_split(
            X_valid, test_size=test_size, random_state=random_state, stratify=y_valid
        )
    except ValueError:
        # Fallback if stratification fails (e.g. single sample class)
        return train_test_split(
            image_paths, test_size=test_size, random_state=random_state
        )


def extract_roi(
    image: np.ndarray, label: Tuple[int, float, float, float, float]
) -> Optional[np.ndarray]:
    """
    Extracts the Region of Interest (ROI) from the image based on the label.

    Args:
        image (np.ndarray): The source image.
        label (Tuple[int, float, float, float, float]): The label containing bounding box info.

    Returns:
        Optional[np.ndarray]: The extracted ROI, or None if invalid.
    """
    img_h, img_w = image.shape[:2]
    _, xc, yc, w, h = label

    x1 = int((xc - w / 2) * img_w)
    y1 = int((yc - h / 2) * img_h)
    x2 = int((xc + w / 2) * img_w)
    y2 = int((yc + h / 2) * img_h)

    # Clip to image bounds
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(img_w, x2), min(img_h, y2)

    if x1 >= x2 or y1 >= y2:
        return None

    return image[y1:y2, x1:x2]
