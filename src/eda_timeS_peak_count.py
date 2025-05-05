import numpy as np
import matplotlib.pyplot as plt
import cv2
import pandas as pd
from pathlib import Path
from scipy.signal import find_peaks


def extract_velocity_map(img):
    """
    Extract velocity from a color heatmap image (red = positive, blue = negative).

    Args:
        img (np.array): BGR image with shape (distance, time, 3).

    Returns:
        velocity (np.array): Velocity map in range [-1, 1], shape (distance, time).
    """
    blue, green, red = cv2.split(img)
    red = red.astype(float) / 255.0
    blue = blue.astype(float) / 255.0
    velocity = red - blue
    return velocity


def extract_time_series_velocity(img):
    """
    Compute the time series of average velocity over distance for each time point.

    Args:
        img (np.array): BGR image with shape (distance, time, 3).

    Returns:
        ts (np.array): Time series of average velocity, shape (time,).
    """
    velocity = extract_velocity_map(img)
    return np.mean(velocity, axis=0)


def detect_peaks(ts, height_threshold=0.02):
    """
    Detect peaks and troughs in the time series and classify their direction.

    Args:
        ts (np.array): Time series of average velocity.
        height_threshold (float): Minimum absolute height for a peak/trough.

    Returns:
        dict: Counts of positive and negative peaks.
    """
    # Detect positive peaks (local maxima)
    peaks, _ = find_peaks(ts, height=height_threshold)
    # Detect troughs (local minima) by negating the time series
    troughs, _ = find_peaks(-ts, height=height_threshold)

    # Classify peaks by direction
    positive_peaks = [p for p in peaks if ts[p] > 0]
    negative_peaks = [p for p in troughs if ts[p] < 0]

    return {
        "positive_peaks": len(positive_peaks),
        "negative_peaks": len(negative_peaks),
        "positive_peak_indices": positive_peaks,
        "negative_peak_indices": negative_peaks
    }


def plot_time_series_with_peaks(df, activity, num_samples=5, output_dir=None):
    """
    Plot time series with peaks marked and print peak counts.

    Args:
        df (pd.DataFrame): DataFrame with file_path and activity columns.
        activity (str): Activity to plot.
        num_samples (int): Number of samples to plot.
        output_dir (Path): Directory to save plots.
    """
    samples = df[df["activity"] == activity].sample(num_samples)
    peak_counts = []

    for idx, (_, row) in enumerate(samples.iterrows()):
        img = cv2.imread(row["file_path"])
        if img is None:
            continue
        ts = extract_time_series_velocity(img)

        # Detect peaks
        peak_info = detect_peaks(ts)
        peak_counts.append({
            "activity": activity,
            "sample_idx": row.get("sample_idx", "N/A"),
            "positive_peaks": peak_info["positive_peaks"],
            "negative_peaks": peak_info["negative_peaks"]
        })

        # Plot time series with peaks
        plt.figure(figsize=(12, 6))
        plt.plot(ts, label="Average Velocity", color="blue")

        # Mark positive peaks
        if peak_info["positive_peak_indices"]:
            plt.plot(peak_info["positive_peak_indices"], ts[peak_info["positive_peak_indices"]], "r^",
                     label="Positive Peaks")
        # Mark negative peaks
        if peak_info["negative_peak_indices"]:
            plt.plot(peak_info["negative_peak_indices"], ts[peak_info["negative_peak_indices"]], "bv",
                     label="Negative Peaks")

        plt.title(
            f"Time Series of Average Velocity for {activity.capitalize()} (Sample {row.get('sample_idx', 'N/A')})")
        plt.xlabel("Time")
        plt.ylabel("Average Velocity")
        plt.legend()
        plt.grid(True)

        if output_dir:
            plt.savefig(output_dir / f"{activity}_sample_{row.get('sample_idx', idx)}_peaks.png")
        plt.show()

    # Summarize peak counts
    peak_df = pd.DataFrame(peak_counts)
    print(f"\nPeak counts for {activity}:")
    print(peak_df[["sample_idx", "positive_peaks", "negative_peaks"]])

    return peak_df


def main():
    # Paths
    data_dir = Path(r"C:\Users\Ahmed\OneDrive\Bureau\mmEat\outputs\csv\all_activities.csv")
    output_dir = Path(r"C:\Users\Ahmed\OneDrive\Bureau\mmEat\outputs\figures")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load DataFrame
    df = pd.read_csv(data_dir)
    print("Dataset loaded successfully:")
    print(df["activity"].value_counts())

    # Analyze peaks for each activity
    all_peak_counts = []
    for activity in ["chopsticks", "fork", "bare_hand", "fork_knife", "spoon"]:
        peak_df = plot_time_series_with_peaks(df, activity, num_samples=3, output_dir=output_dir)
        all_peak_counts.append(peak_df)

    # Aggregate peak counts across activities
    all_peak_counts = pd.concat(all_peak_counts, ignore_index=True)
    print("\nSummary of peak counts across activities:")
    print(all_peak_counts.groupby("activity")[["positive_peaks", "negative_peaks"]].mean())

    # Save summary to CSV
    all_peak_counts.to_csv(output_dir.parent / "csv" / "peak_counts.csv", index=False)
    print(f"Peak counts saved to {output_dir.parent / 'csv' / 'peak_counts.csv'}")


if __name__ == "__main__":
    main()