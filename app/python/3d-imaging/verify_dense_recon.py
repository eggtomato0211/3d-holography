
import sys
import os
import numpy as np

# パスを通す
sys.path.append(r'C:\Users\Owner\mizusaki\3d-holography\app\python\3d-imaging')

from image_loader import ImageLoader
from propagation import Propagation
from phase_generator import PhaseGenerator
from reconstruction_processor import ReconstructionProcessor

# テストパラメータ
Nx, Ny = 32, 32
depthlevel = 32 # 高速化のため減らす
pixels = 1
number_of_plots = 1
channels_per_batch = 1
dx = 3.45 * 10**-6
dy = dx
dz = 4 * 10**-6
wav_len = 532.0 * 10**-9

# 画像ローダーのテスト
print("Loading images...")
image_folder = f'C:\\Users\\Owner\\mizusaki\\3d-holography\\app\\python\\3d-imaging\\src\\{Nx}x{Ny}x128_1pxx{number_of_plots}' # 元フォルダはdepth128で作られている可能性があるため合わせる
# 正確なパスを探す必要があるので、既存のknown folderを使う
# list_dirの結果では: C:\Users\Owner\mizusaki\3d-holography\app\python\3d-imaging\src\32x32x128_1pxx1 があるはず
image_folder = r'C:\Users\Owner\mizusaki\3d-holography\app\python\3d-imaging\src\32x32x128_1pxx1'

loader = ImageLoader(Nx, Ny, pixels, number_of_plots, channels_per_batch, depthlevel, folder_path=image_folder)
images = loader.load_images()
print(f"Loaded {len(images)} images.")

# 正規化
images = [img / 255.0 for img in images]

# クラス初期化
prop = Propagation(wav_len, dx, dy)
phase_gen = PhaseGenerator(False)
recon = ReconstructionProcessor(Nx, Ny, dx, dy, dz, depthlevel, channels_per_batch, images, False, phase_gen, prop)

# 指定するスライス数
target_slice_count = 8 # テスト用

# 実行
print(f"Running reconstruction with target_slice_count={target_slice_count}...")
raw, label = recon.process_channel(0, target_slice_count) # channel 0

print(f"Done. Raw shape: {raw.shape}, Label shape: {label.shape}")
print(f"Raw range: {np.min(raw)} - {np.max(raw)}")
print(f"Label range: {np.min(label)} - {np.max(label)}")

# 検証: Labelデータにおいて、各層に値が入っているか（=ビーズが配置されているか）
non_zero_layers = 0
for z in range(depthlevel):
    if np.max(label[z]) > 0:
        non_zero_layers += 1

print(f"Non-zero layers in label: {non_zero_layers}/{depthlevel}")
print(f"Target slice count: {target_slice_count}")

if non_zero_layers == target_slice_count:
    print("SUCCESS: Beads count matches target slice count.")
else:
    print(f"WARNING: Layer count mismatch! Expected {target_slice_count}, got {non_zero_layers}.")
