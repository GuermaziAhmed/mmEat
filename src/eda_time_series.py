import numpy as np
import matplotlib.pyplot as plt
import cv2
import pandas as pd
from pathlib import Path


def extract_velocity_map(img):
    """
    Extract velocity from a color heatmap image (red = positive, blue = negative).

    Args:
        img (np.array): BGR image.

    Returns:
        velocity (np.array): Velocity map in range [-1, 1].
    """
    # Split into B, G, R channels
    blue, green, red = cv2.split(img)

    # Normalize channels to [0, 1]
    red = red.astype(float) / 255.0
    blue = blue.astype(float) / 255.0

    # Compute velocity: red (positive) - blue (negative)
    velocity = red - blue  # Range [-1, 1]

    return velocity


def extract_time_series(img):
    """
    Compute the time series by averaging the velocity map over the distance axis.

    Args:
        img (np.array): BGR image.

    Returns:
        ts (np.array): Time series of average velocity.
    """
    velocity = extract_velocity_map(img)
    return np.mean(velocity, axis=0)  # Average over distance (y-axis)


def plot_time_series(df, activity, num_samples=5):
    """
    Plot time series of average velocity for a given activity.

    Args:
        df (pd.DataFrame): DataFrame with file_path and activity columns.
        activity (str): Activity to plot.
        num_samples (int): Number of samples to plot.
    """
    samples = df[df["activity"] == activity].sample(num_samples)
    plt.figure(figsize=(12, 6))
    for _, row in samples.iterrows():
        # Load image in color (BGR)
        img = cv2.imread(row["file_path"])
        if img is None:
            continue
        ts = extract_time_series(img)
        plt.plot(ts, label=f"Sample {row.get('sample_idx', 'N/A')}")
    plt.title(f"Time Series of Average Velocity for {activity.capitalize()}")
    plt.xlabel("Time")
    plt.ylabel("Average Velocity")
    plt.legend()
    plt.grid(True)
    plt.show()


def main():
    # Load DataFrame from CSV
    data_dir = Path(r"C:\Users\Ahmed\OneDrive\Bureau\mmEat\outputs\csv\all_activities.csv")
    df = pd.read_csv(data_dir)
    print("Dataset loaded successfully:")
    print(df["activity"].value_counts())

    # Plot time series for each activity
    for activity in ["chopsticks", "fork", "bare_hand", "fork_knife", "spoon"]:
        plot_time_series(df, activity, num_samples=1)


if __name__ == "__main__":
    main()