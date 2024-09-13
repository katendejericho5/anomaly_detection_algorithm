import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from data_stream import DataStreamSimulator
from anomaly_detection import AnomalyDetector

class AnomalyVisualizer:
    """Visualizes the data stream and detected anomalies in real-time."""

    def __init__(self, window_size: int = 50, initial_z_threshold: float = 3):
        """
        Initialize the AnomalyVisualizer.

        Args:
            window_size (int): The size of the sliding window for the AnomalyDetector.
            initial_z_threshold (float): The initial Z-score threshold for anomaly detection.
        """
        self.data_stream = DataStreamSimulator(size=1000).generate()
        self.detector = AnomalyDetector(window_size, initial_z_threshold)
        self.data_points = []
        self.anomalies = []
        self.scores = []
        self.indices = []

    def update(self, frame: int):
        """
        Update the visualization with the next data point.

        Args:
            frame (int): The current frame number (unused, required by FuncAnimation).

        Returns:
            tuple: Updated plot elements.
        """
        data_point = next(self.data_stream)
        is_anomaly, score = self.detector.detect(data_point)

        self.data_points.append(data_point)
        self.scores.append(score)
        self.indices.append(frame)
        self.anomalies.append(data_point if is_anomaly else None)

        self.line1.set_data(self.indices, self.data_points)
        self.scatter1.set_offsets(np.c_[self.indices, self.anomalies])
        self.line2.set_data(self.indices, self.scores)
        self.threshold_line.set_data(self.indices, [self.detector.z_threshold] * len(self.indices))

        self.ax1.relim()
        self.ax1.autoscale_view()
        self.ax2.relim()
        self.ax2.autoscale_view()

        return self.line1, self.scatter1, self.line2, self.threshold_line

    def visualize(self):
        """Set up and start the real-time visualization."""
        fig, (self.ax1, self.ax2) = plt.subplots(2, 1, figsize=(12, 10))
        self.line1, = self.ax1.plot([], [], lw=2, color='blue', label='Data Points')
        self.scatter1 = self.ax1.scatter([], [], color='red', label='Anomalies', marker='o')
        self.line2, = self.ax2.plot([], [], lw=2, color='green', label='Anomaly Scores')
        self.threshold_line, = self.ax2.plot([], [], lw=2, color='red', linestyle='--', label='Threshold')

        self.ax1.set_title('Data Stream and Anomalies')
        self.ax1.set_xlabel('Index')
        self.ax1.set_ylabel('Data Value')
        self.ax1.legend()

        self.ax2.set_title('Anomaly Scores')
        self.ax2.set_xlabel('Index')
        self.ax2.set_ylabel('Score')
        self.ax2.legend()

        ani = FuncAnimation(fig, self.update, frames=range(1000), init_func=self.init, blit=True, interval=50)
        plt.tight_layout()
        plt.show()

    def init(self):
        """Initialize the plot elements."""
        return self.line1, self.scatter1, self.line2, self.threshold_line
