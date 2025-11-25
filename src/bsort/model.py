from typing import Protocol

import cv2
import numpy as np


class BottlecapModel(Protocol):
    def predict(self, image_roi: np.ndarray) -> int: ...


class ColorThresholdModel:
    """
    Heuristic model based on HSV color thresholding.
    """

    def __init__(self):
        # Light Blue Thresholds
        self.lb_lower = np.array([85, 50, 50])
        self.lb_upper = np.array([100, 255, 255])

        # Dark Blue Thresholds
        self.db_lower = np.array([100, 50, 30])
        self.db_upper = np.array([135, 255, 255])

    def predict(self, image_roi: np.ndarray) -> int:
        """
        Predicts the class of the bottlecap ROI.

        Args:
            image_roi (np.ndarray): The region of interest containing the bottlecap.

        Returns:
            int: 0 (Light Blue), 1 (Dark Blue), or 2 (Others)
        """
        if image_roi is None or image_roi.size == 0:
            return 2  # Default to Others if invalid

        hsv_roi = cv2.cvtColor(image_roi, cv2.COLOR_BGR2HSV)

        mask_lb = cv2.inRange(hsv_roi, self.lb_lower, self.lb_upper)
        mask_db = cv2.inRange(hsv_roi, self.db_lower, self.db_upper)

        count_lb = cv2.countNonZero(mask_lb)
        count_db = cv2.countNonZero(mask_db)

        total_pixels = image_roi.shape[0] * image_roi.shape[1]
        threshold = total_pixels * 0.1  # 10% threshold

        if count_lb > count_db and count_lb > threshold:
            return 0  # Light Blue
        elif count_db > count_lb and count_db > threshold:
            return 1  # Dark Blue
        else:
            return 2  # Others
