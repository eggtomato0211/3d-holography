from HDF import HDF
Nx = 1024
Ny = 1024
depthlevel = 16
dz = 6.9e-06
output_dir = None
hdf = HDF(Nx, Ny, depthlevel, dz, output_dir)
# 例の使用方法
hdf5_file = r'C:\Users\Owner\mizusaki\3d-holography\app\python\3d-imaging\hdf\Images=2_128x128x128_d=6.900000000000001e-05_RandomPhase=False\NumberFrom8193.h5'  # 読み込むHDF5ファイルのパス
predictions_hdf5_file = r'C:\Users\Owner\mizusaki\3d-holography\app\python\3d-imaging\hdf\Images=2_128x128x128_d=6.900000000000001e-05_RandomPhase=False\NumberFrom8193_predictions.h5'  # 読み込むHDF5ファイルのパス
depth = 1 # 表示したい深さ

predictions = 'predictions'
raw = 'raw'
label = 'label'

# for n in range(16, 128):
#     hdf.display_image(label, hdf5_file, n)
#     hdf.display_image(raw, hdf5_file, n)
#     hdf.display_predictions_image(predictions, predictions_hdf5_file, n)

# predictions_hdf5_fileの動画保存
# hdf.make_movie(predictions, predictions_hdf5_file)

hdf.make_movie2(raw, hdf5_file)