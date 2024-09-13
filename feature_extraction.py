import numpy as np
from collections import deque
from typing import Tuple, List
import random
import math
from typing import Iterator
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

class FeatureExtractor:
    """
    A class used to extract features from a stream of data points for anomaly detection.

    This class maintains a sliding window of the most recent data points and extracts statistical
    features such as the mean, standard deviation, Z-score, interquartile range (IQR), and the trend (slope) 
    from the data in the window. These features are helpful for detecting anomalies in real-time data streams.

    Attributes:
    -----------
    window : deque
        A deque object that holds the most recent data points up to the specified window size. 
        It automatically discards the oldest data point when a new one is added after reaching its max length.

    Methods:
    --------
    __init__(window_size: int):
        Initializes the FeatureExtractor with a specific window size.
        
    extract(x: float) -> List[float]:
        Appends the current data point to the window and extracts key features such as the moving average,
        standard deviation, Z-score, interquartile range (IQR), and trend (slope).
    """

    def __init__(self, window_size: int):
        """
        Initializes the FeatureExtractor with a sliding window of a fixed size.

        Args:
            window_size (int): The number of past data points to consider for feature extraction.
                               This size determines how many data points will be used to calculate the
                               statistical features.
        """
        # Initialize a deque with a fixed size to store recent data points
        self.window = deque(maxlen=window_size)

    def extract(self, x: float) -> List[float]:
        """
        Extracts statistical features from the current data point and the recent data in the sliding window.

        The function calculates various features based on the window of recent data points and the current 
        data point. If the window is not yet full (i.e., has fewer points than `window_size`), the raw value 
        is returned. Once the window is full, it computes the following features:
        
        1. **Current Value**: The raw value of the current data point.
        2. **Moving Average**: The mean of the data points in the window.
        3. **Standard Deviation**: A measure of how spread out the data points in the window are.
        4. **Z-Score**: The number of standard deviations the current value is from the mean.
        5. **Interquartile Range (IQR)**: The range between the 75th percentile (Q3) and 25th percentile (Q1), used to detect outliers.
        6. **Trend (Slope)**: The slope of the linear trend in the data points, indicating whether the data shows an increasing, decreasing, or stable trend.
        """
        # Add the current data point to the sliding window
        self.window.append(x)

        # If the window is not full, return the raw value of the current data point
        if len(self.window) < self.window.maxlen:
            return [x]

        # Extract the key statistical features once the window is full
        return [
            x,  # Current data point (raw value)
            np.mean(self.window),  # Moving average of the window
            np.std(self.window),  # Standard deviation of the window
            (x - np.mean(self.window)) / (np.std(self.window) or 1),  # Z-score: Standardized measure of how far `x` is from the mean
            np.percentile(self.window, 75) - np.percentile(self.window, 25),  # IQR: Difference between the 75th and 25th percentiles (Q3 - Q1)
            np.polyfit(range(len(self.window)), self.window, 1)[0]  # Trend (Slope): Linear trend over the window using linear regression
        ]
