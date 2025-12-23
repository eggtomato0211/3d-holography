from HDF import HDF
import os
import cv2
import h5py
import numpy as np

tifs_name = "TIE_mouse_brain"
width = 128
height = 128


tifs_folder = fr"C:\\Users\\Owner\\mizusaki\\3d-holography\\app\\python\\3d-imaging\\tiff\\{tifs_name}"
output_file = fr"C:\\Users\\Owner\\mizusaki\\3d-holography\\app\\python\\3d-imaging\\hdf\\{tifs_name}_{width}x{height}.h5"

image_files = sorted(
    [f for f in os.listdir(tifs_folder) if f.endswith(".tif")],
    key=lambda x: int(''.join(filter(str.isdigit, os.path.splitext(x)[0]))),
)

print(image_files)

# 画像サイズ
resize_shape = (width, height)

# HDF5ファイルの作成
with h5py.File(output_file, "w") as h5_file:

    # 画像の3次元配列を作成
    raw_data = np.zeros((len(image_files), *resize_shape), dtype=float)

    # Resize images
    for i, image_file in enumerate(image_files):
        image = cv2.imread(os.path.join(tifs_folder, image_file), cv2.IMREAD_GRAYSCALE)
        # 画像のリサイズの分岐
        if image.shape != resize_shape:
            image = cv2.resize(image, resize_shape, interpolation=cv2.INTER_LINEAR)
        raw_data[i] = image

    h5_file.create_dataset('raw', data=raw_data, compression='gzip')