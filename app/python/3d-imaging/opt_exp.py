from HDF import HDF
# tifデータを読み込んでＨＤＦ５ファイルに保存する
tifs_folder = r"C:\Users\Owner\mizusaki\20240724\2024-07-24_2\2024-07-24_2"
Nx = 128

hdf = HDF(Nx, Nx, 1, 1, "C:\\Users\\Owner\\mizusaki\\3d-holography\\app\\python\\3d-imaging\\hdf\\2024-07-24_2")
hdf.translateTif2HDF(tifs_folder, "optExp01.h5", Nx)