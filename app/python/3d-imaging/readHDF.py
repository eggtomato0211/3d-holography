import h5py
import numpy as np
import matplotlib.pyplot as plt

depthlevel = 16
Nx, Ny, dz = 128, 128, 3.45
channels = 16

# HDF5ファイルのパス
hdf_file_path = f".\\app\\python\\3d-imaging\\hdf\\channels={channels}_{Nx}x{Ny}_{depthlevel}_d={dz}.h5"

# 表示したいチャンネルと深さを指定
channel_to_display = 2  # 例: 最初のチャンネル
depth_to_display = 12   # 例: 中間の深さ (128/2)

# HDF5ファイルを開く
with h5py.File(hdf_file_path, 'r') as f:
    # 'raw'データセットを読み込む
    raw_data = f['raw'][:]

    # 特定のチャンネルと深さのデータを取り出す
    image_to_display = raw_data[channel_to_display, depth_to_display, :, :]

    # 画像の振幅を計算（複素数データの場合）
    image_amplitude = np.abs(image_to_display)

    # 画像を表示
    plt.figure(figsize=(10, 10))
    plt.imshow(image_amplitude, cmap='gray')
    plt.title(f"Channel: {channel_to_display}, Depth: {depth_to_display}")
    plt.colorbar()
    plt.show()