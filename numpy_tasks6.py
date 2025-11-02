import numpy as np
import matplotlib.pyplot as plt

def draw_rectangle(a, b, m, n, rectangle_color, background_color):
    image = np.full((n, m, 3), background_color, dtype=np.uint8)
    x_start = (m - a) // 2
    y_start = (n - b) // 2
    x_end = x_start + a
    y_end = y_start + b  
    image[y_start:y_end, x_start:x_end] = rectangle_color
    
    return image

def draw_ellipse(a, b, m, n, ellipse_color, background_color):
    image = np.full((n, m, 3), background_color, dtype=np.uint8)
    x0, y0 = m // 2, n // 2
    x = np.arange(m)
    y = np.arange(n)
    X, Y = np.meshgrid(x, y)
    ellipse_mask = ((X - x0) / a)**2 + ((Y - y0) / b)**2 <= 1
    image[ellipse_mask] = ellipse_color
    
    return image

#Тесты
import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from main import draw_rectangle, draw_ellipse

def test_rectangle_basic():
    result = draw_rectangle(100, 50, 200, 100, (255, 0, 0), (0, 0, 0))
    assert result.shape == (100, 200, 3)
    assert np.array_equal(result[25, 50], [255, 0, 0])

def test_ellipse_basic():
    result = draw_ellipse(40, 20, 100, 100, (0, 0, 255), (255, 255, 255))
    assert result.shape == (100, 100, 3)
    assert np.array_equal(result[50, 50], [0, 0, 255])

def test_rectangle_colors():
    result = draw_rectangle(50, 50, 100, 100, (0, 255, 0), (0, 0, 255))
    assert np.array_equal(result[25, 25], [0, 255, 0])
    assert np.array_equal(result[0, 0], [0, 0, 255])

def test_ellipse_circle():
    result = draw_ellipse(30, 30, 100, 100, (255, 0, 0), (0, 0, 0))
    assert np.array_equal(result[50, 50], [255, 0, 0])

#Запускать через pytest numpy_tasks6.py -v