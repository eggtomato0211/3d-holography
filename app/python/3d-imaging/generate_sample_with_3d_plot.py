
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# パスを通す
sys.path.append(r'C:\Users\Owner\mizusaki\3d-holography\app\python\3d-imaging')

from image_loader import ImageLoader
from propagation import Propagation
from phase_generator import PhaseGenerator
from reconstruction_processor import ReconstructionProcessor

# パラメータ (main.py準拠だがサイズ小さめ)
Nx, Ny = 32, 32
depthlevel = 32 # 可視化しやすいように層数を減らす(128だと重なりすぎるため)
# しかしユーザーは「配置」を見たいはずなので、ある程度の層は維持したいが、計算時間とのトレードオフ。
# 32層で十分確認できるはず。
# --> ユーザー要望は「1-10個/slice」なので、32 sliceなら 32-320 beadsになる。

pixels = 1
number_of_plots = 1
channels_per_batch = 1
dx = 3.45 * 10**-6
dy = dx
dz = 4 * 10**-6
wav_len = 532.0 * 10**-9

# 画像ローダー
print("Loading images...")
# image_folder = r'C:\Users\Owner\mizusaki\3d-holography\app\python\3d-imaging\src\32x32x128_1pxx1'
# 実際にはフォルダ名が変わっている可能性があるので、存在するフォルダを使用
# verify_dense_recon.py で成功したパスを使用
image_folder = r'C:\Users\Owner\mizusaki\3d-holography\app\python\3d-imaging\src\32x32x128_1pxx1'

loader = ImageLoader(Nx, Ny, pixels, number_of_plots, channels_per_batch, depthlevel, folder_path=image_folder)
images = loader.load_images()
print(f"Loaded {len(images)} images.")
images = [img / 255.0 for img in images]

# 生成
print("Generating volumetric data...")
prop = Propagation(wav_len, dx, dy)
phase_gen = PhaseGenerator(False)
recon = ReconstructionProcessor(Nx, Ny, dx, dy, dz, depthlevel, channels_per_batch, images, False, phase_gen, prop)

# channel 0生成
# process_channelの戻り値は (depth, Nx, Ny)
# テスト用に 8枚のスライスに配置する設定で実行
target_slices = 8
print(f"Generating data with {target_slices} active slices...")
raw, label = recon.process_channel(0, target_slices)

print(f"Generated data shape: {label.shape}")
print("Creating 3D scatter plot...")

# 3Dプロット (Labelデータを使用: ビーズの位置が明確なため)
# Rawデータは伝搬してボケているため、散布図にすると雲のようになる。配置確認ならLabelが最適。
# ただしユーザーは「生成されたデータ(Raw)」を見たいかもしれないが、配置ロジックの確認ならLabel。
# 両方作ろう。

def save_3d_scatter(data, filename, threshold_ratio=0.5, title="3D Plot"):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # 閾値でフィルタリング
    threshold = np.max(data) * threshold_ratio
    # dataの軸順序: (z, x, y) -> plotでは (x, y, z)
    z, x, y = np.where(data > threshold)
    values = data[z, x, y]
    
    # z軸は層インデックスそのまま使う、あるいは物理距離に変換
    # ここではインデックスで表示
    
    sc = ax.scatter(x, y, z, c=values, cmap='viridis', s=5, alpha=0.8)
    
    ax.set_xlabel('X pixel')
    ax.set_ylabel('Y pixel')
    ax.set_zlabel('Z layer')
    ax.set_title(title)
    
    # 視点調整
    ax.view_init(elev=20, azim=45)
    
    plt.colorbar(sc, shrink=0.5, aspect=5)
    plt.savefig(filename)
    print(f"Saved {filename}")
    plt.close()

# Label Data Plot
save_3d_scatter(label, "3d_scatter_label.png", threshold_ratio=0.1, title="Bead Placement (Label Data)")

# Raw Data Plot (閾値を高めにして強い光の部分だけ表示)
save_3d_scatter(raw, "3d_scatter_raw.png", threshold_ratio=0.3, title="Reconstructed Volume (Raw Data)")

print("Done.")
