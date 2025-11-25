import pytest
import numpy as np
import cv2
from bsort.model import ColorThresholdModel

@pytest.fixture
def model():
    return ColorThresholdModel()

def create_solid_color_image(hsv_color, size=(50, 50)):
    """Creates a BGR image of a solid HSV color."""
    # Create HSV image
    img_hsv = np.full((size[0], size[1], 3), hsv_color, dtype=np.uint8)
    # Convert to BGR
    img_bgr = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR)
    return img_bgr

def test_predict_light_blue(model):
    """Test prediction for Light Blue color."""
    # Light Blue is approx H=90-100. Let's pick H=95, S=200, V=200
    lb_color = (95, 200, 200)
    img = create_solid_color_image(lb_color)
    
    pred = model.predict(img)
    assert pred == 0 # Light Blue

def test_predict_dark_blue(model):
    """Test prediction for Dark Blue color."""
    # Dark Blue is approx H=100-135. Let's pick H=120, S=200, V=100
    db_color = (120, 200, 100)
    img = create_solid_color_image(db_color)
    
    pred = model.predict(img)
    assert pred == 1 # Dark Blue

def test_predict_others(model):
    """Test prediction for Other color (e.g., Red)."""
    # Red is approx H=0 or H=170+. Let's pick H=0, S=200, V=200
    red_color = (0, 200, 200)
    img = create_solid_color_image(red_color)
    
    pred = model.predict(img)
    assert pred == 2 # Others

def test_predict_empty_image(model):
    """Test prediction for empty/None image."""
    assert model.predict(None) == 2
    assert model.predict(np.array([])) == 2
