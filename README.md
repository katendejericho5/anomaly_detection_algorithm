# Anomaly Detection in Data Streams

This project implements a basic anomaly detection system for streaming data. It generates a stream of data with seasonal patterns, gradual changes, and occasional anomalies, and it detects outliers or unusual data points in real-time.

## Table of Contents

1. [Introduction](#introduction)
2. [How It Works](#how-it-works)
3. [Components](#components)
   - [1. FeatureExtractor](#1-featureextractor)
   - [2. AnomalyDetector](#2-anomalydetector)
   - [3. DataStreamSimulator](#3-datastreamsimulator)
   - [4. AnomalyVisualizer](#4-anomalyvisualizer)
4. [Running the Program](#running-the-program)
5. [Visualizing the Data](#visualizing-the-data)

---

## Introduction

This Python project detects anomalies in a streaming data set using adaptive thresholding. It simulates a data stream and continuously checks if incoming data points are normal or abnormal. The project also includes a real-time visualization to see both the data points and detected anomalies.

## How It Works

- A **data stream** is generated with normal patterns, like waves and some random noise.
- Occasionally, **anomalies** (unusual spikes or drops) are introduced into the stream.
- The code extracts **features** from the incoming data points, such as the average and trend over time.
- Using a **Z-score**, the system calculates how far a data point is from the expected range.
- If the Z-score is higher than a certain threshold, the point is flagged as an **anomaly**.
- A real-time graph is drawn to show both the data and the detected anomalies.

## Components

### 1. FeatureExtractor

This component processes the incoming data and extracts useful features to help identify anomalies.

- **Sliding Window:** A small window of the most recent data points is maintained.
- **Features:** From this window, features like the moving average, standard deviation, and trend are calculated.
  
  **Key Functionality:**
  - **Raw Value**: The latest data point.
  - **Moving Average**: Average of the last `N` data points.
  - **Z-Score**: How far the data point is from the average (in terms of standard deviations).
  - **Trend (Slope)**: The rate at which the data is increasing or decreasing.

### 2. AnomalyDetector

This component decides if a data point is an anomaly or not. 

- **Z-score Threshold**: A data point is flagged as an anomaly if its Z-score is higher than a certain threshold (e.g., 3).
- **Adaptive Threshold**: As new data comes in, the threshold is adjusted based on recent Z-scores to ensure it's dynamic.

  **Key Functionality:**
  - **Z-Score Calculation**: Determines how extreme a data point is compared to recent values.
  - **Anomaly Detection**: Compares the maximum Z-score of the extracted features to the threshold to decide if it is an anomaly.

### 3. DataStreamSimulator

This component simulates a data stream with patterns and random anomalies.

- **Seasonal Component**: A wave-like pattern is added to mimic real-world behavior like temperature changes or stock prices.
- **Drift**: Gradual changes over time to make the data realistic.
- **Noise**: Random noise is added to each data point to make it less predictable.
- **Anomalies**: Occasionally, a random large jump or drop is introduced to simulate an anomaly.

### 4. AnomalyVisualizer

This component handles real-time visualization of the data stream and detected anomalies using `matplotlib`.

- **Real-time Plot**: It shows the incoming data points and highlights anomalies in red.
- **Scores Plot**: It also displays a plot of the Z-scores and the threshold used to detect anomalies.

  **Key Functionality:**
  - **Data Stream Visualization**: Plots the real-time data stream with anomalies highlighted.
  - **Anomaly Scores**: Plots the Z-scores to see how far each point deviates from the normal range.
  
## Running the Program

To run this program, make sure you have Python installed along with the required libraries. You can install the required libraries by running:

```bash
pip install numpy matplotlib
```

Then, simply run the Python script:

```bash
python anomaly_detection.py
```

## Visualizing the Data

Once the script is running, a real-time graph will appear. You will see:

- **Data Stream (Blue Line):** The simulated data being generated.
- **Anomalies (Red Dots):** Points where the system has detected an anomaly.
- **Anomaly Scores (Green Line):** The Z-scores of each data point.
- **Threshold (Red Dashed Line):** The current threshold for detecting anomalies.

---

### Example:

![Example Graph](#)

In the above visualization:
- **Blue Line**: Represents the data stream.
- **Red Dots**: Represent the detected anomalies.
- **Green Line**: Represents the anomaly scores.
- **Red Dashed Line**: Represents the adaptive Z-score threshold.

---

## Conclusion

This project demonstrates a basic approach to detecting anomalies in a real-time data stream using simple feature extraction and Z-scores. It also provides a way to visualize the data and detected anomalies for easy interpretation.