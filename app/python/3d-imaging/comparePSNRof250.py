import os
import pandas as pd

# Base directory containing the psnr.txt files
base_dir = r""

# Dictionary to store data from psnr.txt files
data = []

# Iterate over each folder (1 to 250)
for folder in sorted(os.listdir(base_dir)):
    folder_path = os.path.join(base_dir, folder)
    if os.path.isdir(folder_path):
        psnr_file = os.path.join(folder_path, "psnr.txt")
        if os.path.exists(psnr_file):
            with open(psnr_file, "r") as f:
                content = f.readlines()
                raw_mse = float(content[0].split(":")[1].strip())
                pred_mse = float(content[1].split(":")[1].strip())
                raw_psnr = float(content[2].split(":")[1].strip())
                pred_psnr = float(content[3].split(":")[1].strip())
                data.append({
                    "Folder": int(folder),
                    "Raw MSE": raw_mse,
                    "Prediction MSE": pred_mse,
                    "Raw PSNR": raw_psnr,
                    "Prediction PSNR": pred_psnr,
                    "Delta PSNR": pred_psnr - raw_psnr
                })

# Convert the data into a DataFrame for analysis
df = pd.DataFrame(data)

# Find the required statistics
highest_psnr = df.loc[df["Prediction PSNR"].idxmax()]
lowest_psnr = df.loc[df["Prediction PSNR"].idxmin()]
max_delta_psnr = df.loc[df["Delta PSNR"].idxmax()]
min_delta_psnr = df.loc[df["Delta PSNR"].idxmin()]

highest_psnr, lowest_psnr, max_delta_psnr, min_delta_psnr
