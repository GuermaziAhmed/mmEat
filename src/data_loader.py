import os
from glob import glob
import pandas as pd
from pathlib import Path

# Define paths to unzipped folders
data_dir = "C:\\Users\\Ahmed\\OneDrive\\Bureau\\mmEat\\data\\raw"
activities = {
    "EA1": "chopsticks",
    "EA2": "fork",
    "EA3": "bare_hand",
    "EA4": "fork_knife",
    "EA5": "spoon"
}

# Collect all .png files
data = []
for ea_folder, activity in activities.items():
    png_files = glob(os.path.join(data_dir, ea_folder, ea_folder,"*.png"))
    for file in png_files:
        # Extract metadata from filename (e.g., EA_20230401_lab1_2_1_001.png)
        filename = os.path.basename(file)
        parts = filename.split("_")
        date, env, activity_idx, user_id, sample_idx = parts[1:6]
        data.append({
            "file_path": file,
            "activity": activity,
            "user_id": user_id,
            "sample_idx": sample_idx.replace(".png", "")
        })
# Define paths
    output_dir = Path(r"C:\Users\Ahmed\OneDrive\Bureau\mmEat\outputs\csv")
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "all_activities.csv"
# Create a DataFrame
df = pd.DataFrame(data)
#df.to_csv(csv_path, index=False)
# Function to load and display a heatmap


print(df.head())
print(df.shape)  # Check number of samples
print(df["activity"].value_counts())  # Check distribution
