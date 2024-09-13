import numpy as np
from collections import deque
from typing import Tuple
from feature_extraction import FeatureExtractor

class AnomalyDetector:
    """Detects anomalies in a data stream using adaptive thresholding with Z-score and IQR."""

    def __init__(self, window_size: int = 50, initial_z_threshold: float = 3, iqr_factor: float = 1.5):
        """
        Initialize the AnomalyDetector.

        Args:
            window_size (int): The size of the sliding window for feature extraction.
            initial_z_threshold (float): The initial Z-score threshold for detecting anomalies.
            iqr_factor (float): The factor to multiply with IQR to set the threshold for anomaly detection.
        """
        self.feature_extractor = FeatureExtractor(window_size)
        self.z_threshold = initial_z_threshold
        self.iqr_factor = iqr_factor
        self.scores_window = deque(maxlen=window_size)  # Store recent maximum Z-scores

    def detect(self, data_point: float) -> Tuple[bool, float, float]:
        """
        Detect if a data point is an anomaly using both Z-score and IQR-based methods.

        Args:
            data_point (float): The current data point to analyze.

        Returns:
            Tuple[bool, float, float]: A tuple containing:
                - A boolean indicating if the data point is considered an anomaly.
                - The maximum Z-score of the extracted features.
                - The IQR threshold used for anomaly detection.
        """
        features = self.feature_extractor.extract(data_point)

        # Calculate Z-scores for each feature
        z_scores = [
            (f - np.mean(self.feature_extractor.window)) / (np.std(self.feature_extractor.window) or 1)
            for f in features
        ]
        max_z_score = max(abs(z) for z in z_scores)

        # Calculate IQR-based threshold
        iqr = np.percentile(self.feature_extractor.window, 75) - np.percentile(self.feature_extractor.window, 25)
        iqr_threshold = iqr * self.iqr_factor

        # Update the scores window and adjust the Z-score threshold
        self.scores_window.append(max_z_score)
        if len(self.scores_window) == self.scores_window.maxlen:
            self.z_threshold = np.mean(self.scores_window) + 2 * np.std(self.scores_window)

        # Anomaly detection
        is_anomaly_z = max_z_score > self.z_threshold
        is_anomaly_iqr = abs(data_point - np.mean(self.feature_extractor.window)) > iqr_threshold

        is_anomaly = is_anomaly_z or is_anomaly_iqr
        return is_anomaly, max_z_score, iqr_threshold
