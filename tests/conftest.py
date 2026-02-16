"""Shared test fixtures — synthetic test images."""

from __future__ import annotations

import numpy as np
import pytest


@pytest.fixture
def white_square_100():
    """100x100 white square on 200x200 black background.

    Known area: 100*100 = 10000 px^2.
    Square spans [50, 150) in both x and y.
    """
    img = np.zeros((200, 200), dtype=np.uint8)
    img[50:150, 50:150] = 255
    return img


@pytest.fixture
def checkerboard_50():
    """200x200 checkerboard with 50px pitch.

    4x4 grid of 50x50 cells alternating black/white.
    Useful for FFT grid detection tests.
    """
    img = np.zeros((200, 200), dtype=np.uint8)
    for i in range(4):
        for j in range(4):
            if (i + j) % 2 == 0:
                img[i * 50 : (i + 1) * 50, j * 50 : (j + 1) * 50] = 255
    return img


@pytest.fixture
def gradient_edge_image():
    """200x200 image with a sharp vertical edge at x=100.

    Left half = 0, right half = 255. Provides a well-defined
    sub-pixel edge for edge detection testing.
    """
    img = np.zeros((200, 200), dtype=np.uint8)
    img[:, 100:] = 255
    return img


@pytest.fixture
def tilted_rect_image():
    """300x300 image with a white rectangle drawn at a known perspective.

    The rectangle corners in the image are at known pixel positions,
    and its real-world size is 10x8 (arbitrary units).
    """
    img = np.zeros((300, 300), dtype=np.uint8)
    # Draw a slightly trapezoidal shape (simulating perspective)
    pts = np.array([
        [80, 60],   # TL
        [220, 70],  # TR
        [210, 230], # BR
        [90, 220],  # BL
    ], dtype=np.int32)
    import cv2
    cv2.fillPoly(img, [pts], 255)
    return img, pts.tolist()
