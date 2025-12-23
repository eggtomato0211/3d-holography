import h5py
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 最新のHDF5ファイルを指定（現在生成中のもの）
# コマンドステータスから最新のファイルパスを使用
hdf_file = r"D:\nosaka\data\3d-holography_output\Train\random_32x32x128_d=4e-06_pixels=1_1plots_32images_dense1-10\1plots_32images_dense1-10_randomTrue_NumberFrom4353.h5"

output_dir = r"C:\Users\Owner\.gemini\antigravity\brain\cce2baaa-af92-4967-88be-e125312c929a"

print(f"読み込み中: {hdf_file}\n")

with h5py.File(hdf_file, 'r') as f:
    raw_data = f['raw'][:]
    label_data = f['label'][:]

print(f"データ形状: {label_data.shape}")
print(f"Label範囲: {np.min(label_data):.4f} ～ {np.max(label_data):.4f}")
print(f"Raw範囲: {np.min(raw_data):.4f} ～ {np.max(raw_data):.4f}\n")

# ビーズの配置を詳細に分析
non_zero_slices = []
for z in range(label_data.shape[0]):
    if np.max(label_data[z]) > 0:
        non_zero_slices.append(z)
        num_beads = np.sum(label_data[z] > 0.01)
        max_val = np.max(label_data[z])
        print(f"層 {z:3d}: ビーズ数 {num_beads:2d}個, 最大値 {max_val:.3f}")

print(f"\n合計: {len(non_zero_slices)}層にビーズが配置されています")

# 3Dプロット作成（Label Data - 詳細版）
fig = plt.figure(figsize=(15, 12))

# 1. Label Data - 全体像
ax1 = fig.add_subplot(221, projection='3d')
threshold = 0.01
z, y, x = np.where(label_data > threshold)
values = label_data[label_data > threshold]

scatter1 = ax1.scatter(x, y, z, c=values, cmap='hot', s=5, alpha=0.6)
ax1.set_xlabel('X')
ax1.set_ylabel('Y')
ax1.set_zlabel('Z (depth)')
ax1.set_title(f'Label Data (全体)\nビーズ配置層: {len(non_zero_slices)}層')
fig.colorbar(scatter1, ax=ax1, shrink=0.5)

# 2. Label Data - 上から見た図（XY平面）
ax2 = fig.add_subplot(222)
projection_xy = np.max(label_data, axis=0)
im2 = ax2.imshow(projection_xy, cmap='hot', aspect='auto')
ax2.set_xlabel('X')
ax2.set_ylabel('Y')
ax2.set_title('Label Data (XY投影)')
fig.colorbar(im2, ax=ax2)

# 3. Label Data - 横から見た図（XZ平面）
ax3 = fig.add_subplot(223)
projection_xz = np.max(label_data, axis=1)
im3 = ax3.imshow(projection_xz, cmap='hot', aspect='auto')
ax3.set_xlabel('X')
ax3.set_ylabel('Z (depth)')
ax3.set_title('Label Data (XZ投影)')
fig.colorbar(im3, ax=ax3)

# 4. 各層のビーズ数のヒストグラム
ax4 = fig.add_subplot(224)
bead_counts = []
for z in range(label_data.shape[0]):
    count = np.sum(label_data[z] > 0.01)
    if count > 0:
        bead_counts.append(count)

if bead_counts:
    ax4.hist(bead_counts, bins=range(1, 12), edgecolor='black', alpha=0.7)
    ax4.set_xlabel('ビーズ数/層')
    ax4.set_ylabel('層の数')
    ax4.set_title(f'各層のビーズ数分布\n平均: {np.mean(bead_counts):.1f}個/層')
    ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f"{output_dir}/detailed_label_analysis.png", dpi=150, bbox_inches='tight')
print(f"\n保存完了: detailed_label_analysis.png")

# Raw Dataの3Dプロット
fig2 = plt.figure(figsize=(12, 10))
ax5 = fig2.add_subplot(111, projection='3d')

# Raw dataは閾値を調整して表示
threshold_raw = np.max(raw_data) * 0.3
z, y, x = np.where(raw_data > threshold_raw)
values_raw = raw_data[raw_data > threshold_raw]

scatter2 = ax5.scatter(x, y, z, c=values_raw, cmap='viridis', s=1, alpha=0.3)
ax5.set_xlabel('X')
ax5.set_ylabel('Y')
ax5.set_zlabel('Z (depth)')
ax5.set_title(f'Raw Data (シミュレーション結果)\n閾値: {threshold_raw:.3f}')
fig2.colorbar(scatter2, ax=ax5, shrink=0.5)

plt.savefig(f"{output_dir}/detailed_raw_visualization.png", dpi=150, bbox_inches='tight')
print(f"保存完了: detailed_raw_visualization.png")

plt.close('all')
print("\n完了！")
