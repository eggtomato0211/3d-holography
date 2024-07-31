import os
import concurrent.futures
from PIL import Image, ImageDraw, ImageFont

def drawNumbers(args):
    i, img_width, img_height, font, save_dir = args
    # 黒い背景の画像を作成
    img = Image.new('RGB', (img_width, img_height), color='black')
    draw = ImageDraw.Draw(img)
    
    # 中央に白い数字を描く
    text = str(i)
    bbox = draw.textbbox((0, 0), text, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    text_x = (img_width - text_width) / 2
    text_y = (img_height - text_height) / 2
    draw.text((text_x, text_y), text, font=font, fill=(255, 255, 255))
    
    # 画像の保存
    img.save(os.path.join(save_dir, f"number_{i:05d}.png"))

# 画像サイズ
voxel_size = 2
box_number = 4

# 画像の幅と高さ
img_width = 1024
img_height = 1024

# 保存用ディレクトリがない場合は作成
save_dir = f"C:\\Users\\Owner\\mizusaki\\3d-holography\\app\python\\3d-imaging\\src\\number_{img_width}x{img_height}"
print(save_dir)
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# フォントの設定
font_size = 400
font = ImageFont.truetype("arial.ttf", font_size)

args_list = [(i, img_width, img_height, font, save_dir) for i in range(1, 10000)]

# 並列処理
with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:
    list(executor.map(drawNumbers, args_list))

print("画像の生成が完了しました。")