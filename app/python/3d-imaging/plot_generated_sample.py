import h5py
import numpy as np
import matplotlib.pyplot as plt
import os

# ターゲットファイル
# 実際に見つかったファイルパスを指定
target_file = r"D:\nosaka\data\3d-holography_output\Train\random_32x32x128_d=4e-06_pixels=1_1plots_8images_dense1-10\1plots_8images_dense1-10_randomTrue_NumberFrom2433.h5"

output_dir = r"C:\Users\Owner\.gemini\antigravity\brain\cce2baaa-af92-4967-88be-e125312c929a"

print(f"Loading {target_file}...")

with h5py.File(target_file, 'r') as f:
    raw_data = f['raw'][:]
    label_data = f['label'][:]

print(f"Raw shape: {raw_data.shape}, range: {np.min(raw_data)} - {np.max(raw_data)}")
print(f"Label shape: {label_data.shape}, range: {np.min(label_data)} - {np.max(label_data)}")

def create_3d_scatter(data, title, save_path, threshold=None):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # 閾値設定 (Labelは0より大きい場所、Rawは適当な閾値)
    if threshold is None:
        threshold = np.max(data) * 0.1
    
    # インデックス取得
    z, x, y = np.where(data > threshold)
    values = data[data > threshold]
    
    print(f"{title}: Found {len(values)} points above threshold {threshold}")
    
    # プロット
    img = ax.scatter(x, y, z, c=values, cmap='viridis', s=1, alpha=0.5)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title(title)
    
    fig.colorbar(img, ax=ax, shrink=0.5, aspect=5)
    
    plt.savefig(save_path)
    print(f"Saved {save_path}")
    plt.close()

# Label Data (正解) - 0より大きい場所
create_3d_scatter(label_data, "Label Data (Ground Truth)", os.path.join(output_dir, "real_generated_label.png"), threshold=0.01)

# Raw Data (シミュレーション) - ノイズフロアを超える場所
create_3d_scatter(raw_data, "Raw Data (Simulation)", os.path.join(output_dir, "real_generated_raw.png"), threshold=np.max(raw_data)*0.2)

print("Done.")
