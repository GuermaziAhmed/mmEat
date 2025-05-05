import numpy as np
import pandas as pd
import cv2
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

# Define activity mapping for readable names
activities = {
    "EA1": "chopsticks",
    "EA2": "fork",
    "EA3": "bare_hand",
    "EA4": "fork_knife",
    "EA5": "spoon"
}

# Reverse mapping for display
activity_display_names = {
    "chopsticks": "Chopsticks",
    "fork": "Fork",
    "bare_hand": "Bare Hand",
    "fork_knife": "Fork & Knife",
    "spoon": "Spoon"
}


def extract_velocity_map(img_rgb):
    """
    Extract velocity from a color heatmap image (white ~ 0, blue = -1, red = 1).

    Args:
        img_rgb (np.array): RGB image with shape (height, width, 3).

    Returns:
        velocity (np.array): Velocity map in range [-1, 1], shape (height, width).
    """
    r, g, b = img_rgb[:, :, 0], img_rgb[:, :, 1], img_rgb[:, :, 2]

    # Normalize channels to [0, 1]
    r = r.astype(float) / 255.0
    g = g.astype(float) / 255.0
    b = b.astype(float) / 255.0

    # Approximate velocity: R - B
    velocity = r - b

    # Adjust for white areas (where R, G, B are similar)
    color_diff = np.abs(r - g) + np.abs(g - b) + np.abs(b - r)
    white_mask = color_diff < 0.1
    velocity[white_mask] = 0

    return velocity


def compute_heatmap_stats(df):
    """
    Compute statistics on the velocity map extracted from heatmap images.

    Args:
        df (pd.DataFrame): DataFrame with columns 'file_path' and 'activity'.

    Returns:
        pd.DataFrame: DataFrame with velocity-based statistics.
    """
    stats = []
    for _, row in df.iterrows():
        img = cv2.imread(row["file_path"])  # Load in color (BGR)
        if img is None:
            continue

        # Convert to RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Extract velocity
        velocity = extract_velocity_map(img_rgb)

        # Scale velocity to [-100, 100] for interpretability
        velocity_scaled = velocity * 100

        # Compute statistics on scaled velocity
        stats.append({
            "activity": row["activity"],
            "mean_velocity": np.mean(velocity_scaled),
            "std_velocity": np.std(velocity_scaled),
            "min_velocity": np.min(velocity_scaled),
            "max_velocity": np.max(velocity_scaled),
            "median_velocity": np.median(velocity_scaled)
        })

    stats_df = pd.DataFrame(stats)
    # Map activity names to display names
    stats_df["activity"] = stats_df["activity"].map(activity_display_names)
    return stats_df


def plot_stats_boxplot(stats_df, stat_name):
    """
    Plot a boxplot of the specified statistic by activity.

    Args:
        stats_df (pd.DataFrame): DataFrame with statistics.
        stat_name (str): Name of the statistic to plot (e.g., 'mean_velocity').
    """
    plt.figure(figsize=(10, 6))
    sns.boxplot(x="activity", y=stat_name, data=stats_df)
    plt.title(f"{stat_name.replace('_', ' ').title()} of Heatmaps by Activity")
    plt.xlabel("Activity")
    plt.ylabel(stat_name.replace("_", " ").title())
    plt.xticks(rotation=45)
    plt.show()


def main():
    data_dir = Path(r"C:\Users\Ahmed\OneDrive\Bureau\mmEat\outputs\csv\all_activities.csv")
    df = pd.read_csv(data_dir)

    # Compute statistics
    stats_df = compute_heatmap_stats(df)

    # Print grouped statistics
    print("\nHeatmap Statistics (Velocity-based, Scaled):")
    print(stats_df.groupby("activity").mean(numeric_only=True))

    # Plot boxplots
    plot_stats_boxplot(stats_df, "mean_velocity")
    plot_stats_boxplot(stats_df, "std_velocity")


if __name__ == "__main__":
    main()