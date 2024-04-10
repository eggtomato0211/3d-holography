import numpy as np
import cv2

# 画像の枚数
num_images = 256

# 画像サイズ
image_size = (256, 256)

# 画像生成および保存
for i in range(num_images):
    # 黒い背景の画像生成
    white_image = np.ones(image_size, dtype=np.uint8) * 0

    # ランダムな数字を生成
    number = i+1
    # 数字を描画
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 4
    font_thickness = 5
    font_color = 255  # 白色
    text_size = cv2.getTextSize(str(number), font, font_scale, font_thickness)[0]
    text_position = ((image_size[1] - text_size[0]) // 2, (image_size[0] + text_size[1]) // 2)
    cv2.putText(white_image, str(number), text_position, font, font_scale, font_color, font_thickness, cv2.LINE_AA)

    # 画像の保存
    file_path = f'./src/number_{i + 1:04d}.bmp'
    cv2.imwrite(file_path, white_image)

print(f"{num_images}枚の白い背景に数字が描かれた画像を生成しました。")
