import numpy as np
from collections import deque
from typing import Tuple
from feature_extraction import FeatureExtractor

class AnomalyDetector:
    """Detects anomalies in a data stream using adaptive thresholding."""

    def __init__(self, window_size: int = 50, initial_z_threshold: float = 3):
        """
        Initialize the AnomalyDetector.

        Args:
            window_size (int): The size of the sliding window for feature extraction.
            initial_z_threshold (float): The initial Z-score threshold for anomaly detection.
            
        """
        self.feature_extractor = FeatureExtractor(window_size)
        self.z_threshold = initial_z_threshold
        self.scores_window = deque(maxlen=window_size) #deque(double end queue) is a list-like container with fast appends and pops on either end

    def detect(self, data_point: float) -> Tuple[bool, float]:
        """
        Detect if a data point is an anomaly.

        Args:
            data_point (float): The current data point to analyze.

        Returns:
            Tuple[bool, float]: A tuple containing a boolean indicating if the point is an anomaly,and the maximum Z-score of the extracted features.
                                Z= (X - μ) / σ
            where X is the data point, μ is the mean of the window, and σ is the standard deviation of the window.
        """
        features = self.feature_extractor.extract(data_point)

        # Calculate Z-scores for each feature
        z_scores = [
            (f - np.mean(self.feature_extractor.window)) / (np.std(self.feature_extractor.window) or 1)
            for f in features
        ]
        max_z_score = max(abs(z) for z in z_scores)

        # Update the scores window and adjust the threshold
        self.scores_window.append(max_z_score)
        if len(self.scores_window) == self.scores_window.maxlen:
            self.z_threshold = np.mean(self.scores_window) + 2 * np.std(self.scores_window)

        is_anomaly = max_z_score > self.z_threshold
        return is_anomaly, max_z_score