import random
import math
from typing import Iterator

class DataStreamSimulator:
    """Simulates a data stream with seasonal patterns, drift, and occasional anomalies."""

    def __init__(self, size: int = 1000, drift_rate: float = 0.01):
        """
        Initialize the DataStreamSimulator.

        Args:
            size (int): The number of data points to generate.
            drift_rate (float): The rate at which the base value drifts over time.
        """
        self.size = size
        self.drift_rate = drift_rate

    def generate(self) -> Iterator[float]:
        """
        Generate a stream of data points.

        Yields:
            float: The next data point in the stream.
        """
        base_value = 50
        for i in range(self.size):
            seasonal_component = 10 * math.sin(i * 0.05) # Adds a sinusoidal component to simulate seasonal variations.
            drift_component = i * self.drift_rate # Adds a linear drift over time.
            noise = random.uniform(-3, 3) # Adds random noise between -3 and 3 to the data.
            value = base_value + seasonal_component + drift_component + noise

            # Introduce occasional anomalies
            if random.random() < 0.01:  #With a 1% probability, an anomaly is added to the value.
                value += random.choice([-1, 1]) * random.uniform(20, 30)

            yield value