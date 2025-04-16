import cv2
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import pandas as pd

activities = {
    "EA1": "chopsticks",
    "EA2": "fork",
    "EA3": "bare_hand",
    "EA4": "fork_knife",
    "EA5": "spoon"
}

df=pd.read_csv("C:\\Users\\Ahmed\\OneDrive\\Bureau\\mmEat\\outputs\\csv\\all_activities.csv")
# Function to load and display a heatmap
def plot_heatmap(file_path, title):
    img = cv2.imread(file_path)  # Load as grayscale
    plt.figure(figsize=(8, 4))
    plt.imshow(img)
    plt.title(title)
    plt.xlabel("Time Duration")
    plt.ylabel("Range of Movement")
    plt.colorbar(label="Velocity (Normalized)")
    plt.show()
    return img

# Plot one sample per activity
for activity in activities.values():
    sample = df[df["activity"] == activity]["file_path"].iloc[0]
    plot_heatmap(sample, f"Heatmap: {activity}")

# Summarize heatmap statistics
stats = []
for _, row in df.iterrows():
    img = cv2.imread(row["file_path"], cv2.IMREAD_GRAYSCALE)
    stats.append({
        "activity": row["activity"],
        "mean_intensity": np.mean(img),
        "std_intensity": np.std(img),
        "nonzero_pixels": np.count_nonzero(img) / img.size  # Activity density
    })

stats_df = pd.DataFrame(stats)
print(stats_df.groupby("activity").mean())