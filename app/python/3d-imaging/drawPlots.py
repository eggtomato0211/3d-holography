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
number_of_plots = 1
# 保存用ディレクトリがない場合は作成
save_dir = f"C:\\Users\\Owner\\mizusaki\\3d-holography\\app\python\\3d-imaging\\src\\{width}x{height}x{depth}_{voxel_size}pxx{number_of_plots}"
print(save_dir)
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

def generate_image(i):
    # 16bit画像を生成 ("I;16"モード)
    img = Image.new("I;16", (width, height), color=0)  # 背景は0 (黒)
    draw = ImageDraw.Draw(img)
    
    box_number = number_of_plots
    for _ in range(box_number):
        x = random.randint(0, width - 1)
        y = random.randint(0, height - 1)
        # 矩形を白 (最大値 65535) で塗りつぶし
        draw.rectangle((x, y, x+voxel_size-1, y+voxel_size-1), fill=65535)
    
    # ファイル名を指定して保存
    filename = os.path.join(save_dir, f"image_{i+1:05d}.tiff")  # TIFF形式で保存
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