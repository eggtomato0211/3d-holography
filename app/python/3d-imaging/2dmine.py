import os
from PIL import Image
import numpy as np
from HDF import HDF

# TIFファイルが保存されているディレクトリのパス
directory_path = 'C:\\Users\\Owner\\mizusaki\\3d-holography\\app\\python\\3d-imaging\\tif\\20240724'

# HDF5ファイルのパス
hdf_file_path = 'output_data.h5'

Nx = 1024
Ny = 1024
depthlevel = 1
dz = 0
images = 10000

output_hdfdir = rf'.\app\python\3d-imaging\hdf_mine\{Nx}x{Ny}_images={images}'

label_data = np.zeros((depthlevel, Nx, Ny), dtype=float)
raw_data = np.zeros((depthlevel, Nx, Ny), dtype=float)

for i in range(1, images + 1):
    QD_tif_file_path = os.path.join(directory_path, f"QD\QD_{i}.tif")
    sca_tif_file_path = os.path.join(directory_path, f"scateringQD\sca_{i}.tif")

    QD_image = Image.open(QD_tif_file_path)
    sca_image = Image.open(sca_tif_file_path)

    QD_image_array = np.array(QD_image)
    sca_image_array = np.array(sca_image)

    label_data[0, :, :] = QD_image_array
    raw_data[0, :, :] = sca_image_array

    hdf_maker = HDF(Nx, Ny, depthlevel, dz, output_hdfdir)
    hdf_maker.makeHDF(raw_data, label_data, f"NumberFrom{i}.h5")


print(f"全てのTIFファイルを{hdf_file_path}に保存しました。")
