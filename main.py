import numpy as np
import random
import math
from typing import Iterator, List, Tuple
from collections import deque
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def data_stream_simulation(size: int = 1000, drift_rate: float = 0.01) -> Iterator[float]:
    base_value = 50
    for i in range(size):
        seasonal_component = 10 * math.sin(i * 0.05)
        drift_component = i * drift_rate
        noise = random.uniform(-3, 3)
        value = base_value + seasonal_component + drift_component + noise
        # Introduce occasional anomalies
        if random.random() < 0.01:  # 1% chance of anomaly
            value += random.choice([-1, 1]) * random.uniform(20, 30)
        yield value

class DynamicFeatureExtractor:
    def __init__(self, window_size: int):
        self.window = deque(maxlen=window_size)

    def extract_features(self, x: float) -> List[float]:
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

class AdaptiveAnomalyDetector:
    def __init__(self, window_size: int = 50, initial_z_threshold: float = 3):
        self.feature_extractor = DynamicFeatureExtractor(window_size)
        self.z_threshold = initial_z_threshold
        self.scores_window = deque(maxlen=window_size)

    def detect_anomaly(self, data_point: float) -> Tuple[bool, float]:
        features = self.feature_extractor.extract_features(data_point)
        
        # Use multiple features for anomaly detection
        z_scores = [(f - np.mean(self.feature_extractor.window)) / (np.std(self.feature_extractor.window) or 1) 
                    for f in features]
        max_z_score = max(abs(z) for z in z_scores)
        
        self.scores_window.append(max_z_score)
        
        # Adaptive thresholding
        if len(self.scores_window) == self.scores_window.maxlen:
            self.z_threshold = np.mean(self.scores_window) + 2 * np.std(self.scores_window)

        is_anomaly = max_z_score > self.z_threshold
        return is_anomaly, max_z_score

def detect_anomalies_and_visualize(window_size: int = 50, initial_z_threshold: float = 3):
    data_stream = data_stream_simulation()
    detector = AdaptiveAnomalyDetector(window_size, initial_z_threshold)

    data_points = []
    anomalies = []
    scores = []
    indices = []

    # Prepare the plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    line1, = ax1.plot([], [], lw=2, color='blue', label='Data Points')
    scatter1 = ax1.scatter([], [], color='red', label='Anomalies', marker='o')
    line2, = ax2.plot([], [], lw=2, color='green', label='Anomaly Scores')
    threshold_line, = ax2.plot([], [], lw=2, color='red', linestyle='--', label='Threshold')

    ax1.set_title('Data Stream and Anomalies')
    ax1.set_xlabel('Index')
    ax1.set_ylabel('Data Value')
    ax1.legend()

    ax2.set_title('Anomaly Scores')
    ax2.set_xlabel('Index')
    ax2.set_ylabel('Score')
    ax2.legend()

    def init():
        return line1, scatter1, line2, threshold_line

    def update(frame):
        data_point = next(data_stream)
        is_anomaly, score = detector.detect_anomaly(data_point)

        data_points.append(data_point)
        scores.append(score)
        indices.append(frame)
        anomalies.append(data_point if is_anomaly else None)

        line1.set_data(indices, data_points)
        scatter1.set_offsets(np.c_[indices, anomalies])
        line2.set_data(indices, scores)
        threshold_line.set_data(indices, [detector.z_threshold] * len(indices))

        ax1.relim()
        ax1.autoscale_view()
        ax2.relim()
        ax2.autoscale_view()

        return line1, scatter1, line2, threshold_line

    ani = FuncAnimation(fig, update, frames=range(1000), init_func=init, blit=True, interval=50)
    plt.tight_layout()
    plt.show()

# Run the anomaly detection and visualization
detect_anomalies_and_visualize(window_size=50, initial_z_threshold=3)