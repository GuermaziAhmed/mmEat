import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

activities = {
    "EA1": "chopsticks",
    "EA2": "fork",
    "EA3": "bare_hand",
    "EA4": "fork_knife",
    "EA5": "spoon"
}

df = pd.read_csv(r"C:\Users\Ahmed\OneDrive\Bureau\mmEat\outputs\csv\all_activities.csv")


def extract_velocity_map(img_rgb):
    """
    Extract velocity from a color heatmap image (white ~ 0, blue = -1, red = 1).

    Args:
        img_rgb (np.array): RGB image with shape (height, width, 3).

    Returns:
        velocity (np.array): Velocity map in range [-1, 1], shape (height, width).
    """
    # Split into R, G, B channels
    r, g, b = img_rgb[:, :, 0], img_rgb[:, :, 1], img_rgb[:, :, 2]

    # Normalize channels to [0, 1]
    r = r.astype(float) / 255.0
    g = g.astype(float) / 255.0
    b = b.astype(float) / 255.0

    # Approximate velocity based on color
    velocity = r - b

    # Adjust for white areas (where R, G, B are similar)
    color_diff = np.abs(r - g) + np.abs(g - b) + np.abs(b - r)
    white_mask = color_diff < 0.1
    velocity[white_mask] = 0

    return velocity


def plot_heatmap(file_path, title):
    """
    Load and display a heatmap in color with appropriate labels and colorbar.

    Args:
        file_path (str): Path to the heatmap image.
        title (str): Title of the plot.

    Returns:
        img_rgb (np.array): RGB image for statistics.
    """
    img = cv2.imread(file_path)  # Load in color (BGR)
    if img is None:
        print(f"Failed to load image: {file_path}")
        return None

    # Convert BGR to RGB for matplotlib
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    plt.figure(figsize=(8, 4))
    plt.imshow(img_rgb)
    plt.title(title)
    plt.xlabel("Time Duration")
    plt.ylabel("Range of Movement")
    plt.colorbar(label="Velocity (Normalized)", ticks=[0, 255], format=lambda x, _: -1 if x == 0 else 1, cmap="RdBu")
    plt.clim(0, 255)
    plt.show()

    return img_rgb


# Plot one sample per activity
for activity in activities.values():
    sample = df[df["activity"] == activity]["file_path"].iloc[0]
    plot_heatmap(sample, f"Heatmap: {activity.capitalize()}")

# Summarize heatmap statistics using velocity
stats = []
for _, row in df.iterrows():
    img = cv2.imread(row["file_path"])
    if img is None:
        continue

    # Convert to RGB
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Extract velocity
    velocity = extract_velocity_map(img_rgb)

    # Compute statistics on velocity and scale
    mean_velocity = np.mean(velocity) * 1000  # Scale to [-1000, 1000]
    std_velocity = np.std(velocity) * 1000  # Scale to [0, 1000]
    # Convert nonzero velocity pixels to percentage
    nonzero_velocity_pixels = (np.count_nonzero(np.abs(velocity) > 0.05) / velocity.size) * 100

    stats.append({
        "activity": row["activity"],
        "mean_velocity": mean_velocity,
        "std_velocity": std_velocity,
        "nonzero_velocity_pixels (%)": nonzero_velocity_pixels
    })

stats_df = pd.DataFrame(stats)
print("\nHeatmap Statistics (Velocity-based, Scaled):")
print(stats_df.groupby("activity").mean())