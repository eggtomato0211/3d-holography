import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import time
import concurrent.futures
import multiprocessing
from threading import Thread
import os
import matplotlib.image as mpimg
import h5py

def nearpropCONV(Comp1, sizex, sizey, dx, dy, shiftx, shifty, wa, d):
    if d == 0:
        Recon = Comp1
    else:
        x1, x2 = -sizex//2, sizex//2-1
        y1, y2 = -sizey//2, sizey//2-1
        Fx, Fy = np.meshgrid(np.arange(x1, x2+1), np.arange(y1, y2+1))

        Fcomp1 = np.fft.fftshift(np.fft.fft2(Comp1)) / np.sqrt(sizex * sizey)

        FresR = np.exp(-1j * np.pi * wa * d * ((Fx**2) / ((dx * sizex)**2) + (Fy**2) / ((dy * sizey)**2)))

        Fcomp2 = Fcomp1 * FresR
        Recon = np.fft.ifft2(np.fft.ifftshift(Fcomp2)) * np.sqrt(sizex * sizey)

    return Recon

# ランダム位相分布
def generate_random_phase(shape):
    return (np.random.rand(*shape) - 0.5) * 2.0 * 2.0 * np.pi

def process_image(n, images, sizex, sizey, dx, dy, wav_len, output_images):
    initial_phase = generate_random_phase(images[n].shape)
    input_image = images[n] * np.exp(1j * initial_phase)
    d = (n+1) * dz * pixels + initial_place
    output_images[n] = nearpropCONV(input_image, sizex, sizey, dx, dy, 0, 0, wav_len, d)

# 波長や画像サイズなどのパラメータ
i = 1j
wav_len = 532.0 * 10**-9
Nx, Ny = 32, 32
dx = 3.45 * 10**-6
dy = dx
dz = dx
wav_num = 2 * np.pi / wav_len
times = -4 
initial_place = (10**times)*1000
pixels = 2
box_number=4

# フィギュアを作成
fig = plt.figure()

# 画像の読み込み時間計測開始
start_time = time.time()

# 画像の枚数
num_images = 32

# フォルダを作成
folder_name = f'.\\app\python\\3d-imaging\\output\\RawOutputData_{num_images}_{Nx}x{Ny}_10{times}_pixels={pixels}_d={dz *10**times}_initialPlace{initial_place}'
os.makedirs(folder_name, exist_ok=True)

max = 32

# 出力画像の初期化
output_images = [np.zeros((Nx, Ny), dtype=np.complex128) for _ in range(num_images)]

# 画像の読み込みとリサイズ
images = [cv2.resize(cv2.imread(f'.\\app\python\\3d-imaging\\src\\Original2Dimages_{box_number}_{pixels}px_{Nx}x{Ny}x{num_images}\\image_{(i):05d}.png', cv2.IMREAD_GRAYSCALE).astype(float), (Nx, Ny)) for i in range(1, num_images + 1)]

# 並列処理
with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:
    # 各画像に対してprocess_image関数を並列に実行
    futures = [executor.submit(process_image, n, images, Nx, Ny, dx, dy, wav_len, output_images) for n in range(num_images)]

    # すべてのタスクの完了を待つ
    concurrent.futures.wait(futures)

# 合計する
total_output_images = np.sum(output_images, axis=0)
print(total_output_images)

# 振幅と位相分布の計算
amplitude_output = np.abs(total_output_images)
phase_output = np.angle(total_output_images)

# 画像の処理時間計測終了
processing_time = time.time() - start_time

# 時間表示
print(f"画像の処理時間: {processing_time} 秒")

# フィギュアを作成
fig = plt.figure()

# 再生計算
SLM_data = np.exp(i * phase_output)

# Initialize a 3D array to store the results
reconst_3d = np.zeros((Nx, Ny, num_images))

def cal_save_image(i, SLM_data, Nx, Ny, dx, dy, wav_len, initial_place, times):
    reconst_2d = nearpropCONV(SLM_data, Nx, Ny, dx, dy, 0, 0, wav_len, -1.0 * ((i) * dz * 10**times + initial_place))
    print("process：", i+1)
    return reconst_2d

# 並列処理
with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:
    start = time.time()
    # 各画像に対してcal_save_image関数を並列に実行
    futures = [executor.submit(cal_save_image, i, SLM_data, Nx, Ny, dx, dy, wav_len, initial_place, times) for i in range(1, num_images+1)]
    for i, future in enumerate(concurrent.futures.as_completed(futures)):
        reconst_3d[:, :, i] = future.result()
    end = time.time()
    print('マルチスレッド: TIME {:.4f}\n'.format(end - start))

# HDFファイルを作成
# 3次元配列に変換
label_3d = np.stack(images, axis=-1)
raw_3d = reconst_3d

# HDFファイルの作成
# 保存先ディレクトリ
output_dir = r'.\\app\\python\\3d-imaging\\hdf'
output_file = os.path.join(output_dir, 'data.h5')

with h5py.File(output_file, 'w') as f:
    # データセットの作成
    f.create_dataset('raw', data=raw_3d, compression='gzip')
    f.create_dataset('label', data=label_3d, compression='gzip')

print(f"HDF5ファイル '{output_file}' が 'raw' と 'label' のデータセットで作成されました")

# # ファイルのパスを作成
# file_path = os.path.join(folder_name, 'random_data.npy')

# # データを保存
# np.save(file_path, reconst_3d)
# print('3D array saved to reconst_3d.npy')

# def compare_slices(reconst_3d, index1, index2):
#     """
#     Compare two slices of the 3D reconstructed data to check if they are identical.
    
#     Parameters:
#     reconst_3d (numpy.ndarray): The 3D array containing the reconstructed data.
#     index1 (int): The index of the first slice to compare.
#     index2 (int): The index of the second slice to compare.
    
#     Returns:
#     bool: True if the slices are identical, False otherwise.
#     """
#     slice1 = reconst_3d[:, :, index1]
#     slice2 = reconst_3d[:, :, index2]
    
#     return np.array_equal(slice1, slice2)

# # Example usage
# index1 = 9
# index2 = 1
# are_identical = compare_slices(reconst_3d, index1, index2)
# print(f"Are the slices at index {index1} and {index2} identical? {are_identical}")