# Anomaly Detection and Visualization

## Overview

The Anomaly Detection and Visualization project is a Python-based tool that detects and visualizes anomalies in a data stream. The project combines data simulation, feature extraction, anomaly detection, and real-time visualization to provide insights into data behavior and anomalies. It utilizes statistical methods such as Z-score and Interquartile Range (IQR) for anomaly detection and employs real-time plotting for visualization.

## Components

1. **DataStreamSimulator**: Simulates a data stream with seasonal patterns, drift, and occasional anomalies.
2. **FeatureExtractor**: Extracts statistical features from the data stream using a sliding window approach.
3. **AnomalyDetector**: Detects anomalies in the data stream using adaptive thresholds based on Z-score and IQR.
4. **AnomalyVisualizer**: Provides real-time visualization of the data stream and detected anomalies using `matplotlib`.

## Techniques Used

- **Data Simulation**: Generates synthetic data with seasonal variations, drift, and anomalies.
- **Feature Extraction**: Uses statistical methods (mean, standard deviation, Z-score, IQR) to extract features from the data stream.
- **Anomaly Detection**: Applies Z-score and IQR-based methods to identify anomalies.
- **Real-Time Visualization**: Uses `matplotlib` to create real-time plots of data points, anomaly scores, and detection thresholds.

## Requirements

The project requires the following Python packages:

- `numpy`
- `matplotlib`


Install the dependencies using:

```bash
pip install -r requirements.txt
```

## Code Overview

### 1. `data_stream.py`

Simulates a data stream with:
- **Seasonal Patterns**: Sinusoidal variations to mimic seasonal changes.
- **Drift**: Gradual increase or decrease in the base value.
- **Anomalies**: Randomly introduced anomalies with a 1% probability.

```python
import random
import math
from typing import Iterator

class DataStreamSimulator:
    """Simulates a data stream with seasonal patterns, drift, and occasional anomalies."""
    # Initialization and data generation methods...
```

### 2. `feature_extraction.py`

Extracts features from the data using:
- **Moving Average**: Mean of the recent data points.
- **Standard Deviation**: Measure of data spread.
- **Z-Score**: Standardized measure of deviation from the mean.
- **Interquartile Range (IQR)**: Range between 75th and 25th percentiles.
- **Trend (Slope)**: Linear trend using regression.

```python
import numpy as np
from collections import deque
from typing import List

class FeatureExtractor:
    """Extracts statistical features from a sliding window of data."""
    # Initialization and feature extraction methods...
```

### 3. `anomaly_detection.py`

Detects anomalies using:
- **Z-Score**: Measures how many standard deviations a data point is from the mean.
- **IQR**: Compares data point deviation to the IQR-based threshold.

```python
import numpy as np
from collections import deque
from typing import Tuple
from feature_extraction import FeatureExtractor

class AnomalyDetector:
    """Detects anomalies using Z-score and IQR-based methods."""
    # Initialization and anomaly detection methods...
```

### 4. `visualisation.py`

Visualizes the data stream and detected anomalies using `matplotlib`:
- **Data Points Plot**: Line plot of data values.
- **Anomalies Plot**: Scatter plot for detected anomalies.
- **Scores and Thresholds**: Line plots for anomaly scores and detection thresholds.

```python
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from data_stream import DataStreamSimulator
from anomaly_detection import AnomalyDetector

class AnomalyVisualizer:
    """Visualizes real-time data and anomalies."""
    # Initialization and visualization methods...
```

### 5. `main.py`

The entry point of the application. Initializes and starts the anomaly visualization process.

```python
from visualisation import AnomalyVisualizer

def main():
    """Starts the visualization of the anomaly detection process."""
    window_size = 50
    initial_z_threshold = 3
    visualizer = AnomalyVisualizer(window_size=window_size, initial_z_threshold=initial_z_threshold)
    visualizer.visualize()

if __name__ == "__main__":
    main()
```

## How to Run

1. **Clone the Repository**:

   ```bash
   git clone https://github.com/katendejericho5/anomaly_detection_algorithm
   cd anomaly_detection_algorithm
   ```

2. **Install Dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Visualization**:

   ```bash
   python main.py
   ```

   This command starts the visualization process, displaying real-time data points, detected anomalies, anomaly scores, and detection thresholds.