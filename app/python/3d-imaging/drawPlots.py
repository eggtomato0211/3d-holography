import os
import random
from PIL import Image, ImageDraw
import concurrent.futures
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 画像サイズ
width, height, depth = 32, 32, 128
voxel_size = 1

# 保存用ディレクトリがない場合は作成
save_dir = f"C:\\Users\\Owner\\mizusaki\\3d-holography\\app\python\\3d-imaging\\src\\{width}x{height}x{depth}_{voxel_size}px"
print(save_dir)
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

def generate_image(i):
    img = Image.new("RGB", (width, height), color=(0, 0, 0))
    draw = ImageDraw.Draw(img)
    #box_numberは0~2のランダムな数値を取得
    box_number = np.random.randint(0, 3)
    for _ in range(box_number):
        x = random.randint(0, width - 1)
        y = random.randint(0, height - 1)
        draw.rectangle((x, y, x+voxel_size-1, y+voxel_size-1), fill=(255, 255, 255))
    filename = os.path.join(save_dir, f"image_{i+1:05d}.png")
    img.save(filename)

print("画像の生成を開始します。")

with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:
    futures = [executor.submit(generate_image, i) for i in range(10000)]

    for future in concurrent.futures.as_completed(futures):
        try:
            future.result()
        except Exception as e:
            print(f"Error occurred: {e}")

print("画像の生成が完了しました。")