import cv2
import numpy as np
import os

# ソース画像フォルダ
image_folder = r"C:\Users\Owner\mizusaki\3d-holography\app\python\3d-imaging\src\32x32x128_1pxx1"

# フォルダ内の画像を確認
image_files = [f for f in os.listdir(image_folder) if f.endswith('.tiff')][:10]

print(f"Found {len(image_files)} images in folder")
print(f"Checking first 10 images:\n")

for img_file in image_files:
    img_path = os.path.join(image_folder, img_file)
    
    # 16bitで読み込み
    img_16bit = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
    
    # 8bitで読み込み（main.pyと同じ方法）
    img_8bit = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    
    if img_16bit is not None:
        print(f"{img_file}:")
        print(f"  16bit - dtype: {img_16bit.dtype}, shape: {img_16bit.shape}")
        print(f"  16bit - min: {np.min(img_16bit)}, max: {np.max(img_16bit)}, unique values: {len(np.unique(img_16bit))}")
        
        if img_8bit is not None:
            print(f"  8bit  - dtype: {img_8bit.dtype}, shape: {img_8bit.shape}")
            print(f"  8bit  - min: {np.min(img_8bit)}, max: {np.max(img_8bit)}, unique values: {len(np.unique(img_8bit))}")
        
        # ビーズの位置と値を確認
        non_zero = np.where(img_16bit > 0)
        if len(non_zero[0]) > 0:
            print(f"  Non-zero pixels: {len(non_zero[0])}")
            print(f"  Non-zero values (16bit): {np.unique(img_16bit[non_zero])}")
            if img_8bit is not None:
                print(f"  Non-zero values (8bit):  {np.unique(img_8bit[non_zero])}")
        print()
