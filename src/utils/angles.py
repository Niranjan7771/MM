"""
Kinematic math utilities for joint angle computation and spatial analysis.

Provides 2D and 3D angle calculation using trigonometric and vector
algebra, Euclidean distance helpers, and midpoint computation used
throughout the pose and face analysis pipelines.
"""

import math
import numpy as np


def calculate_angle_2d(a, b, c):
    """
    Calculate the angle at vertex B formed by points A-B-C in 2D.

    Uses the atan2 method for robust angle computation that correctly
    handles all quadrants. Returns the interior angle in degrees [0, 180].

    Parameters
    ----------
    a : tuple (x, y) -- first endpoint
    b : tuple (x, y) -- vertex (joint center)
    c : tuple (x, y) -- second endpoint

    Returns
    -------
    float : angle in degrees [0, 180]
    """
    ax, ay = a[0], a[1]
    bx, by = b[0], b[1]
    cx, cy = c[0], c[1]

    angle = math.degrees(
        math.atan2(cy - by, cx - bx) - math.atan2(ay - by, ax - bx)
    )

    if angle < 0:
        angle += 360
    if angle > 180:
        angle = 360 - angle

    return angle


def calculate_angle_3d(a, b, c):
    """
    Calculate the angle at vertex B formed by points A-B-C in 3D.

    Uses dot-product / cross-product (vector algebra) for true spatial
    angle computation. Returns the interior angle in degrees [0, 180].

    Parameters
    ----------
    a : tuple (x, y, z) -- first endpoint
    b : tuple (x, y, z) -- vertex (joint center)
    c : tuple (x, y, z) -- second endpoint

    Returns
    -------
    float : angle in degrees [0, 180]
    """
    ba = np.array([a[0] - b[0], a[1] - b[1], a[2] - b[2]], dtype=np.float64)
    bc = np.array([c[0] - b[0], c[1] - b[1], c[2] - b[2]], dtype=np.float64)

    dot = np.dot(ba, bc)
    mag_ba = np.linalg.norm(ba)
    mag_bc = np.linalg.norm(bc)

    if mag_ba == 0 or mag_bc == 0:
        return 0.0

    cos_angle = np.clip(dot / (mag_ba * mag_bc), -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_angle)))


def euclidean_distance_2d(a, b):
    """Return the Euclidean distance between two 2D points."""
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


def euclidean_distance_3d(a, b):
    """Return the Euclidean distance between two 3D points."""
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2)


def midpoint(a, b):
    """
    Return the midpoint between two points.
    Works for any dimension (2D or 3D) based on input length.
    """
    return tuple((a[i] + b[i]) / 2.0 for i in range(min(len(a), len(b))))
