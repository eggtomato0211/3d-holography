from HDF import HDF
Nx = 1024
Ny = 1024
depthlevel = 16
dz = 6.9e-06
output_dir = None
hdf = HDF(Nx, Ny, depthlevel, dz, output_dir)
# 例の使用方法
hdf5_file = r'C:\Users\Owner\mizusaki\3d-holography\app\python\3d-imaging\hdf\128x128x128_d=6.9e-06\NumberFrom8193.h5'  # 読み込むHDF5ファイルのパス
predictions_hdf5_file = r'C:\Users\Owner\mizusaki\3d-holography\app\python\3d-imaging\hdf\128x128x128_d=6.9e-06\NumberFrom8193_predictions.h5'  # 読み込むHDF5ファイルのパス
depth = 1 # 表示したい深さ

predictions = 'predictions'
raw = 'raw'
label = 'label'

for n in range(32, 48):
    hdf.display_image(label, hdf5_file, n)
    hdf.display_image(raw, hdf5_file, n)
    hdf.display_predictions_image(predictions, predictions_hdf5_file, n)
