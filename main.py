from visualisation import AnomalyVisualizer

def main():
    # Initialize the visualizer with specified parameters
    window_size = 50
    initial_z_threshold = 3

    # Create an instance of the AnomalyVisualizer
    visualizer = AnomalyVisualizer(window_size=window_size, initial_z_threshold=initial_z_threshold)

    # Start the visualization process
    visualizer.visualize()

if __name__ == "__main__":
    main()
