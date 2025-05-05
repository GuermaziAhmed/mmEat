import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
from tensorflow.keras.preprocessing.image import img_to_array
#from data_loader import load_dataset


def plot_activity_samples(df, output_path, samples_per_activity=5):
    """
    Plot 5 sample heatmaps for each activity in a single figure.

    Args:
        df (pd.DataFrame): DataFrame with file_path and activity columns.
        output_path (Path): Path to save the output figure.
        samples_per_activity (int): Number of samples to plot per activity.
    """
    # Define activities
    activities = ["chopsticks", "fork", "bare_hand", "fork_knife", "spoon"]
    users=["user-1","user-2"]
    # Set up the plot: 5 rows (activities), 5 columns (samples)
    fig, axes = plt.subplots(nrows=2, ncols=5, figsize=(15, 10), constrained_layout=True)
    activity="bare_hand"
    for i, user_id in enumerate(users):
        # Select random samples for the activity
        #activity_df = df[df["activity"] == activity]
        user_df=df[(df["user_id"]==i+1) & (df["activity"] == activity) ]
        if len(user_df) < samples_per_activity:
            print(len(user_df))
            print(f"Warning: Only {len(user_df)} samples for {user_id}, using all.")
            samples = user_df
        else:
            samples = user_df.sample(n=samples_per_activity, random_state=42)

        # Plot each sample
        for j, (_, row) in enumerate(samples.iterrows()):
            # Load heatmap
            img = cv2.imread(row["file_path"])
            if img is None:
                print(f"Failed to load {row['file_path']}")
                continue

            # Plot on the corresponding subplot
            ax = axes[i, j]
            ax.imshow(img, aspect="auto")
            ax.set_title(f"sample{j + 1} of {user_id} eating with {activity}  ", fontsize=10)
            ax.axis("off")  # Hide axes for clarity

        # Set row label
        axes[i, 0].set_ylabel(user_id.capitalize(), fontsize=12, rotation=0, labelpad=40)

    # Add main title
    fig.suptitle("Sample Heatmaps for Each Eating Activity", fontsize=16)

    # Save and show
    #plt.savefig(output_path, dpi=300, bbox_inches="tight")
    #print(f"Figure saved to {output_path}")
    plt.show()


def main():
    # Define paths
    output_dir = Path(r"C:\Users\Ahmed\OneDrive\Bureau\mmEat\outputs\figures")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "activity_samples.png"

    # Load dataset
    try:
        df=pd.read_csv("C:\\Users\\Ahmed\\OneDrive\\Bureau\\mmEat\\outputs\\csv\\all_activities.csv")

        print("Dataset loaded successfully:")
        print(df["activity"].value_counts())
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    # Plot samples
    plot_activity_samples(df, output_path)


if __name__ == "__main__":
    main()