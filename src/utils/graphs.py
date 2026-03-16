"""
Real-time scrolling line graph renderer.

Draws live mini-graphs directly on the video frame using OpenCV,
showing the temporal evolution of analytics signals like joint angles,
EAR, MAR, and hand openness. Each graph is a scrolling strip-chart
with labeled axes and color-coded traces.
"""

import cv2
import numpy as np
from collections import deque


# Graph visual parameters
GRAPH_WIDTH = 260
GRAPH_HEIGHT = 80
GRAPH_BG_COLOR = (15, 15, 15)
GRAPH_BORDER_COLOR = (60, 60, 60)
GRAPH_ALPHA = 0.6
GRAPH_LINE_THICKNESS = 2
GRAPH_BUFFER_SIZE = 120  # frames of history


class LiveGraph:
    """
    A single scrolling line graph for one signal.

    Parameters
    ----------
    label : str
        Display name shown on the graph.
    color : tuple (B, G, R)
        Line color.
    y_min : float
        Minimum expected value (bottom of graph).
    y_max : float
        Maximum expected value (top of graph).
    unit : str
        Unit label (e.g., 'deg', '', 'px').
    """

    def __init__(self, label, color, y_min=0, y_max=180, unit=''):
        self.label = label
        self.color = color
        self.y_min = y_min
        self.y_max = y_max
        self.unit = unit
        self._buffer = deque(maxlen=GRAPH_BUFFER_SIZE)

    def push(self, value):
        """Add a new data point."""
        if value is not None:
            self._buffer.append(float(value))

    def draw(self, img, x, y, w=GRAPH_WIDTH, h=GRAPH_HEIGHT):
        """
        Render the graph at position (x, y) on the image.

        Returns the y position after the graph (for stacking).
        """
        # Draw semi-transparent background
        overlay = img.copy()
        cv2.rectangle(overlay, (x, y), (x + w, y + h), GRAPH_BG_COLOR, cv2.FILLED)
        cv2.rectangle(overlay, (x, y), (x + w, y + h), GRAPH_BORDER_COLOR, 1)
        cv2.addWeighted(overlay, GRAPH_ALPHA, img, 1 - GRAPH_ALPHA, 0, img)

        # Label
        cv2.putText(img, self.label, (x + 5, y + 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1, cv2.LINE_AA)

        # Current value
        if self._buffer:
            current = self._buffer[-1]
            val_text = f"{current:.1f}{self.unit}"
            cv2.putText(img, val_text, (x + w - 70, y + 14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, self.color, 1, cv2.LINE_AA)

        # Draw the line graph
        if len(self._buffer) >= 2:
            data = list(self._buffer)
            n = len(data)
            point_spacing = w / GRAPH_BUFFER_SIZE

            # Margin inside the graph area
            margin_top = 20
            margin_bottom = 5
            plot_h = h - margin_top - margin_bottom

            points = []
            for i, val in enumerate(data):
                # Map value to pixel y (inverted: higher value = higher on screen)
                clamped = max(self.y_min, min(self.y_max, val))
                ratio = (clamped - self.y_min) / (self.y_max - self.y_min) if self.y_max != self.y_min else 0.5
                px = int(x + (GRAPH_BUFFER_SIZE - n + i) * point_spacing)
                py = int(y + margin_top + plot_h * (1.0 - ratio))
                points.append((px, py))

            # Draw the line segments
            for i in range(1, len(points)):
                cv2.line(img, points[i - 1], points[i], self.color,
                         GRAPH_LINE_THICKNESS, cv2.LINE_AA)

        # Y-axis labels (min/max)
        cv2.putText(img, f"{self.y_max:.0f}", (x + 2, y + 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.25, (100, 100, 100), 1, cv2.LINE_AA)
        cv2.putText(img, f"{self.y_min:.0f}", (x + 2, y + h - 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.25, (100, 100, 100), 1, cv2.LINE_AA)

        return y + h + 3


class GraphPanel:
    """
    Manages a collection of LiveGraph instances and renders them
    stacked vertically at a configurable screen position.
    """

    def __init__(self):
        self._graphs = {}
        self._enabled = True

        # Pre-configure default graphs
        self._graphs['right_elbow'] = LiveGraph(
            'R.Elbow', (0, 180, 255), y_min=0, y_max=180, unit='deg',
        )
        self._graphs['left_elbow'] = LiveGraph(
            'L.Elbow', (255, 180, 0), y_min=0, y_max=180, unit='deg',
        )
        self._graphs['ear'] = LiveGraph(
            'EAR (Blink)', (100, 255, 100), y_min=0.0, y_max=0.5, unit='',
        )
        self._graphs['mar'] = LiveGraph(
            'MAR (Mouth)', (100, 200, 255), y_min=0.0, y_max=1.0, unit='',
        )

    @property
    def enabled(self):
        return self._enabled

    @enabled.setter
    def enabled(self, value):
        self._enabled = value

    def update(self, angles, face_data):
        """
        Push new data points to all graphs.

        Parameters
        ----------
        angles : dict from PoseTracker
        face_data : dict from FaceTracker
        """
        if 'right_elbow' in angles and angles['right_elbow'] is not None:
            self._graphs['right_elbow'].push(angles['right_elbow'])
        if 'left_elbow' in angles and angles['left_elbow'] is not None:
            self._graphs['left_elbow'].push(angles['left_elbow'])
        if face_data:
            self._graphs['ear'].push(face_data.get('ear_avg'))
            self._graphs['mar'].push(face_data.get('mar'))

    def draw(self, img, x, y):
        """
        Draw all graphs stacked vertically starting at (x, y).

        Returns the y position after all graphs.
        """
        if not self._enabled:
            return y

        for graph in self._graphs.values():
            y = graph.draw(img, x, y)

        return y
