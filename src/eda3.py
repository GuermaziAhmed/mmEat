import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path


def plot_activity_samples(df, samples_per_activity=5):
    """
    Plot 5 sample heatmaps for each activity in a single figure with a specified colormap.

    Args:
        df (pd.DataFrame): DataFrame with file_path and activity columns.
        output_path (Path): Path to save the output figure.
        samples_per_activity (int): Number of samples to plot per activity.
        cmap (str): Matplotlib colormap name (default: 'viridis').
    """
    # Define activities
    activities = ["chopsticks", "fork", "bare_hand", "fork_knife", "spoon"]

    # Set up the plot: 5 rows (activities), 5 columns (samples)
    fig, axes = plt.subplots(nrows=5, ncols=5, figsize=(15, 10), constrained_layout=True)

    # Optional: Set a light background for better contrast
    fig.patch.set_facecolor('black')

    for i, activity in enumerate(activities):
        # Select random samples for the activity
        activity_df = df[df["activity"] == activity]
        if len(activity_df) < samples_per_activity:
            print(f"Warning: Only {len(activity_df)} samples for {activity}, using all.")
            samples = activity_df
        else:
            samples = activity_df.sample(n=samples_per_activity, random_state=42)

        # Plot each sample
        for j, (_, row) in enumerate(samples.iterrows()):
            # Load heatmap
            img = cv2.imread(row["file_path"])
            if img is None:
                print(f"Failed to load {row['file_path']}")
                continue

            # Normalize image to [0, 1] to balance color distribution
            img = img.astype(float)
            img = (img - img.min()) / (img.max() - img.min() + 1e-6)  # Avoid division by zero

            # Plot on the corresponding subplot
            ax = axes[i, j]
            im = ax.imshow(img, aspect="auto")
            ax.set_title(f"Sample {j + 1}", fontsize=10,color="white")
            ax.axis("off")  # Hide axes for clarity

        # Set row label
        axes[i, 0].set_ylabel(activity.capitalize(), fontsize=12, rotation=0, labelpad=40,color="white")

    # Add main title
    fig.suptitle("Sample Heatmaps for Each Eating Activity", fontsize=16,color="white")

    # Add a single colorbar for the entire figure
    fig.colorbar(im, ax=axes.ravel().tolist(), label="Intensity", shrink=0.8)

    # Save and show
    plt.show()


def main():
    # Define paths
    data_dir = Path(r"C:\Users\Ahmed\OneDrive\Bureau\mmEat\outputs\csv\all_activities.csv")
    output_dir = Path(r"C:\Users\Ahmed\OneDrive\Bureau\mmEat\outputs\figures")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "activity_samples_new_cmap.png"

    # Load dataset
    try:
        df = pd.read_csv(data_dir)
        print("Dataset loaded successfully:")
        print(df["activity"].value_counts())
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    # Plot samples with a new colormap
    plot_activity_samples(df)  # Try "gray", "plasma", "magma", etc.


if __name__ == "__main__":
    main()
