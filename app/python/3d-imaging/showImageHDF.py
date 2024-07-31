from HDF import HDF
Nx = 1024
Ny = 1024
depthlevel = 16
dz = 6.9e-06
output_dir = None
hdf = HDF(Nx, Ny, depthlevel, dz, output_dir)
# 例の使用方法
hdf5_file = r'C:\Users\Owner\mizusaki\3d-holography\app\python\3d-imaging\hdf\128x128x128_d=4e-06_pixels=3\NumberFrom1.h5'  # 読み込むHDF5ファイルのパス
# predictions_hdf5_file = r'C:\Users\Owner\mizusaki\3d-holography\app\python\3d-imaging\hdf\128x128x128_d=4e-06_pixels=1\NumberFrom8833_predictions.h5'  # 読み込むHDF5ファイルのパス
depth = 1 # 表示したい深さ

save_dir = '.\\app\\python\\3d-imaging\\movies\\128x128x128_d=4e-06_pixels=3'

predictions = 'predictions'
raw = 'raw'
label = 'label'
background_mode = True

# for n in range(1, 30):
#     hdf.display_image(label, hdf5_file, n)
#     hdf.display_image(raw, hdf5_file, n)
    # hdf.display_predictions_image(predictions, predictions_hdf5_file, n)

# # predictions_hdf5_fileの動画保存
# hdf.make_movie(predictions, predictions_hdf5_file, save_dir)

# hdf5_fileの動画保存
hdf.make_movie2(label, hdf5_file, save_dir)
hdf.make_movie2(raw, hdf5_file, save_dir)
