import os
import random
from PIL import Image, ImageDraw
import concurrent.futures

# 画像サイズ
width, height = 1024, 1024

num_images = 10
pixels = 10

# 保存用ディレクトリがない場合は作成
save_dir = f"./src/generated_original_images_{num_images}_{pixels}_{width}x{height}"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
    
def generate_image(i):
    img = Image.new("RGB", (width, height), color=(0, 0, 0))
    draw = ImageDraw.Draw(img)
    for _ in range(10):
        x = random.randint(0, width - 1)
        y = random.randint(0, height - 1)
        draw.rectangle((x, y, x+pixels-1, y+pixels-1), fill=(255, 255, 255))
    filename = os.path.join(save_dir, f"image_{i+1:05d}.png")
    img.save(filename)

with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:
    futures = [executor.submit(generate_image, i) for i in range(num_images)]

    for future in concurrent.futures.as_completed(futures):
        try:
            future.result()
        except Exception as e:
            print(f"Error occurred: {e}")

# # 10000枚の画像を生成
# for i in range(10000):
#     # 新しい画像を作成
#     img = Image.new("RGB", (width, height), color=(0, 0, 0))
#     draw = ImageDraw.Draw(img)

#     # 10個の白い点をランダムな位置に描画
#     for _ in range(10):
#         x = random.randint(0, width - 1)
#         y = random.randint(0, height - 1)
#         draw.rectangle((x, y, x+9, y+9), fill=(255, 255, 255))

#     # 適切な名前で保存
#     filename = os.path.join(save_dir, f"image_{i+1:05d}.png")
#     img.save(filename)

print("画像の生成が完了しました。")