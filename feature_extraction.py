import numpy as np
from collections import deque
from typing import Tuple, List
import random
import math
from typing import Iterator
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

class FeatureExtractor:
    """Extracts features from a data stream for anomaly detection."""

    def __init__(self, window_size: int):
        """
        Initialize the FeatureExtractor.

        Args:
            window_size (int): The size of the sliding window for feature extraction(a fixed number of past data points).
        """
        self.window = deque(maxlen=window_size)

    def extract(self, x: float) -> List[float]:
        """
        Extract features from the current data point and window.

        Args:
            x (float): The current data point.

        Returns:
            List[float]: A list of extracted features.
        """
        self.window.append(x)

        if len(self.window) < self.window.maxlen:
            return [x]  # Return raw value if window not full

        return [
            x,  # Raw value
            np.mean(self.window),  # Moving average
            np.std(self.window),  # Standard deviation
            (x - np.mean(self.window)) / (np.std(self.window) or 1),  # Z-score
            np.percentile(self.window, 75) - np.percentile(self.window, 25),  # IQR
            np.polyfit(range(len(self.window)), self.window, 1)[0]  # Trend (slope)
        ]