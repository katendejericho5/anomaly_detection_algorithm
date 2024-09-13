import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from data_stream import DataStreamSimulator
from anomaly_detection import AnomalyDetector

class AnomalyVisualizer:
    """Visualizes the data stream and detected anomalies in real-time."""

    def __init__(self, window_size: int = 50, initial_z_threshold: float = 3, iqr_factor: float = 1.5):
        """
        Initialize the AnomalyVisualizer.

        Args:
            window_size (int): The size of the sliding window for the AnomalyDetector.
            initial_z_threshold (float): The initial Z-score threshold for anomaly detection.
            iqr_factor (float): The factor to multiply with IQR to set the threshold for anomaly detection.
        """
        # Initialize a data stream simulator to generate data points
        self.data_stream = DataStreamSimulator(size=1000).generate()
        
        # Initialize the anomaly detector with specified parameters
        self.detector = AnomalyDetector(window_size, initial_z_threshold, iqr_factor)
        
        # Lists to store data points, anomalies, anomaly scores, and indices for plotting
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
        # Get the next data point from the data stream
        data_point = next(self.data_stream)
        
        # Detect if the new data point is an anomaly and get the anomaly score
        is_anomaly, score, iqr_threshold = self.detector.detect(data_point)

        # Append the new data point, score, index, and anomaly status to the lists
        self.data_points.append(data_point)
        self.scores.append(score)
        self.indices.append(frame)
        self.anomalies.append(data_point if is_anomaly else None)

        # Update the data for the line plot of data points
        self.line1.set_data(self.indices, self.data_points)
        
        # Update the scatter plot for anomalies
        self.scatter1.set_offsets(np.c_[self.indices, self.anomalies])
        
        # Update the data for the line plot of anomaly scores
        self.line2.set_data(self.indices, self.scores)
        
        # Update the line plot for the Z-score threshold
        self.threshold_line.set_data(self.indices, [self.detector.z_threshold] * len(self.indices))

        # Update the line plot for the IQR threshold if indices are available
        if len(self.indices) > 0:
            self.iqr_threshold_line.set_data(self.indices, [iqr_threshold] * len(self.indices))

        # Rescale the axes to fit the updated data
        self.ax1.relim()
        self.ax1.autoscale_view()
        self.ax2.relim()
        self.ax2.autoscale_view()

        # Return the updated plot elements
        return self.line1, self.scatter1, self.line2, self.threshold_line, self.iqr_threshold_line

    def visualize(self):
        """Set up and start the real-time visualization."""
        fig, (self.ax1, self.ax2) = plt.subplots(2, 1, figsize=(14, 8))
        
        # Customize the appearance of the plots
        self.ax1.grid(True, linestyle='--', alpha=0.7)
        self.ax2.grid(True, linestyle='--', alpha=0.7)
        
        # Data points and anomalies plot
        self.line1, = self.ax1.plot([], [], lw=2, color='royalblue', label='Data Points', zorder=2)
        self.scatter1 = self.ax1.scatter([], [], color='tomato', label='Anomalies', marker='o', edgecolor='black', zorder=3)
        
        # Anomaly scores and thresholds plot
        self.line2, = self.ax2.plot([], [], lw=2, color='mediumseagreen', label='Anomaly Scores', zorder=2)
        self.threshold_line, = self.ax2.plot([], [], lw=2, color='crimson', linestyle='--', label='Z-score Threshold', zorder=1)
        self.iqr_threshold_line, = self.ax2.plot([], [], lw=2, color='darkorange', linestyle='--', label='IQR Threshold', zorder=1)

        # Plot settings
        self.ax1.set_title('Real-Time Data Stream and Anomalies', fontsize=14)
        self.ax1.set_xlabel('Index', fontsize=12)
        self.ax1.set_ylabel('Data Value', fontsize=12)
        self.ax1.legend()
        
        self.ax2.set_title('Anomaly Scores and Detection Thresholds', fontsize=14)
        self.ax2.set_xlabel('Index', fontsize=12)
        self.ax2.set_ylabel('Score', fontsize=12)
        self.ax2.legend()

        # Create an animation object that calls the `update` method at each frame
        ani = FuncAnimation(fig, self.update, frames=range(1000), init_func=self.init, blit=True, interval=50)
        
        # Adjust layout to fit the plots and display
        plt.tight_layout()
        plt.show()

    def init(self):
        """Initialize the plot elements."""
        # Return the initial plot elements to be used by FuncAnimation
        return self.line1, self.scatter1, self.line2, self.threshold_line, self.iqr_threshold_line
