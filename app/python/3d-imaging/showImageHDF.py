from HDF import HDF
Nx = 128
Ny = 128
depthlevel = 129
dz = 6.9e-06
output_dir = None
hdf = HDF(Nx, Ny, depthlevel, dz, output_dir)
#hdf_dirの名前を実行した日時で作成
import datetime
now = datetime.datetime.now()
hdf_dir = now.strftime('%Y%m%d%H%M%S')
#stride数_教師データ数をhdf_dirに追加
stride = 2
label_num = 1000
f_map = 64
from_num = 9601
# hdf_dir += f'_stride={stride}_labels={label_num}_f_map={f_map}'
hdf_name = "Number1"
hdf_dir += f'_{hdf_name}'

# 例の使用方法
# hdf5_file = fr'C:\Users\Owner\mizusaki\3d-holography\app\python\3d-imaging\hdf\test\NumberFrom{from_num}.h5'  # 読み込むHDF5ファイルのパス
# predictions_hdf5_file = fr'C:\Users\Owner\mizusaki\3d-holography\app\python\3d-imaging\hdf\train\NumberFrom{from_num}_predictions_stride={stride}_f_map={f_map}.h5'  # 読み込むHDF5ファイルのパス
# hdf5_file = fr'C:\Users\Owner\mizusaki\3d-holography\app\python\3d-imaging\hdf\32x32x128_test\test\1plots_8images_FalserandomMode_NumberFrom513.h5'  # 読み込むHDF5ファイルのパス
hdf5_file = fr'C:\Users\Owner\mizusaki\3d-holography\app\python\3d-imaging\hdf\{hdf_name}.h5'  # 読み込むHDF5ファイルのパス
predictions_hdf5_file = fr'C:\Users\Owner\mizusaki\3d-holography\app\python\3d-imaging\hdf\{hdf_name}_nonhalo_predictions.h5'  # 読み込むHDF5ファイルのパス
depth = 1  # 表示したい深さ

predictions = 'predictions'
raw = 'raw'
label = 'label'
image_mode = True
movide_mode = True
value_mode = True
mse_mode = False

import h5py
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr

# HDF5ファイルを開く
with h5py.File(hdf5_file, 'r') as f1, h5py.File(predictions_hdf5_file, 'r') as f2:
    label_data = f1['label'][:]  # 教師データ
    raw_data = f1['raw'][:]  # オリジナルデータ
    pred_data = f2['predictions'][:]  # 予測データ
    print(label_data.shape)
    print(raw_data.shape)
    print(pred_data.shape)

# 1つ次元を削除
pred_data = np.squeeze(pred_data, axis=0)

# ** 正規化の適用（バリデーションと同じスケール）**
def normalize_data(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))

label_data = normalize_data(label_data)
raw_data = normalize_data(raw_data)
pred_data = normalize_data(pred_data)

# PSNRの計算
raw_psnr = psnr(label_data, raw_data, data_range=raw_data.max() - raw_data.min())
pred_psnr = psnr(label_data, pred_data, data_range=pred_data.max() - pred_data.min())

print(f'PSNR (raw): {raw_psnr:.2f}')
print(f'PSNR (predictions): {pred_psnr:.2f}')


# if image_mode:
#     save_dir = fr"C:\Users\Owner\mizusaki\3d-holography\app\python\3d-imaging\images\{hdf_dir}"
#     # hdf5_fileの画像ほぞん
#     hdf.save_images(label, hdf5_file, save_dir)
#     hdf.save_images(raw, hdf5_file, save_dir)
#     hdf.save_images(predictions, predictions_hdf5_file, save_dir)


# if movide_mode:
#     save_dir = fr"C:\Users\Owner\mizusaki\3d-holography\app\python\3d-imaging\movies\{hdf_dir}"
#     # hdf5_fileの動画保存
#     hdf.make_movie(label, hdf5_file, save_dir)
#     hdf.make_movie(raw, hdf5_file, save_dir)
#     hdf.make_movie(predictions, predictions_hdf5_file, save_dir)

# if value_mode:
#     save_dir = fr"C:\Users\Owner\mizusaki\3d-holography\app\python\3d-imaging\values\{hdf_dir}"
#     hdf.savePSNR(hdf5_file, predictions_hdf5_file, save_dir)

# if mse_mode:
#     for z in range(128):
#         hdf.zMSE(hdf5_file, predictions_hdf5_file, z)