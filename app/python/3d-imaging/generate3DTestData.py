import os
import random
from PIL import Image, ImageDraw
import concurrent.futures
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 画像サイズ
width, height, depth = 32, 32, 32
voxel_size = 2
box_number = 4

# 保存用ディレクトリがない場合は作成
save_dir = f"C:\\Users\\Owner\\mizusaki\\3d-holography\\app\python\\3d-imaging\\src\\Original3Dimages_{box_number}_{voxel_size}px_{width}x{height}x{depth}"
print(save_dir)
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# 32x32x32の空の3次元配列を生成
data = np.zeros((width, height, depth), dtype=np.uint8)

# 4個の2x2x2の立方体をランダムに配置し、値を255に設定する
for _ in range(box_number):
    x = np.random.randint(0, width-1)
    y = np.random.randint(0, height-1)
    z = np.random.randint(0, depth-1)
    data[x:x+2, y:y+2, z:z+2] = 255

# ファイルのパスを作成
file_path = os.path.join(save_dir, 'random_data.npy')

# データを保存
np.save(file_path, data)

# 3Dプロット
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# データをスキャンし、値が255の座標を取得
x, y, z = np.where(data == 255)

# 値が255の座標に点をプロット
ax.scatter(x, y, z, c='blue', marker='o')

ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')

plt.show()