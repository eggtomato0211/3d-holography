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

def nearpropCONV(Comp1, sizex, sizey, sizez, dx, dy, shiftx, shifty, shiftz, wa, d):
    if d == 0:
        Recon = Comp1
    else:
        x1, x2 = -sizex//2, sizex//2-1
        y1, y2 = -sizey//2, sizey//2-1
        z1, z2 = -sizez//2, sizez//2-1
        Fx, Fy, Fz = np.meshgrid(np.arange(x1, x2+1), np.arange(y1, y2+1), np.arange(z1, z2+1))

        Fcomp1 = np.fft.fftshift(np.fft.fftn(Comp1)) / np.sqrt(sizex * sizey * sizez)

        FresR = np.exp(-1j * np.pi * wa * d * (
            (Fx**2) / ((dx * sizex)**2) + 
            (Fy**2) / ((dy * sizey)**2) + 
            (Fz**2) / ((dz * sizez)**2)
        ))

        Fcomp2 = Fcomp1 * FresR
        Recon = np.fft.ifftn(np.fft.ifftshift(Fcomp2)) * np.sqrt(sizex * sizey * sizez)
    
    return Recon

# ランダム位相分布
def generate_random_phase(shape):
    return (np.random.rand(*shape) - 0.5) * 2.0 * 2.0 * np.pi

def process_image(label_data_npy, sizex, sizey, sizez, dx, dy, wav_len):
    initial_phase = generate_random_phase(label_data_npy.shape)
    input_data = label_data_npy * np.exp(1j * initial_phase)
    d = initial_place
    output_data_npy = nearpropCONV(input_data, sizex, sizey, sizez, dx, dy, 0, 0, 0, wav_len, d)
    return output_data_npy

# 波長や画像サイズなどのパラメータ
i = 1j
wav_len = 532.0 * 10**-9
Nx, Ny, Nz = 32, 32, 32
dx = 3.45 * 10**-6
dy = dx
dz = 1
wav_num = 2 * np.pi / wav_len
times = -4 
initial_place = (10**times)*1000
pixels = 2
box_number = 4

# フィギュアを作成
fig = plt.figure()

# 画像の読み込み時間計測開始
start_time = time.time()

name = 'number'

# フォルダを作成
folder_name = f"C:\\Users\\Owner\\mizusaki\\3d-holography\\app\python\\3d-imaging\\output\\Raw3Dimages_{Nx}x{Ny}x{Nz}_10{times}_pixels={pixels}_initialPlace{initial_place}"
os.makedirs(folder_name, exist_ok=True)

# 出力画像の初期化: 空の3次元配列を生成
raw_data = np.zeros((Nx, Ny, Nz), dtype=np.uint8)

# 画像の読み込みとリサイズ
label_data_npy = np.load(f"C:\\Users\\Owner\\mizusaki\\3d-holography\\app\python\\3d-imaging\\src\\Original3Dimages_{box_number}_{pixels}px_{Nx}x{Ny}x{Nz}\\random_data.npy")

label_data_npy = process_image(label_data_npy, Nx, Ny, Nz, dx, dy, wav_len)

# 振幅と位相分布の計算
amplitude_output = np.abs(label_data_npy)
phase_output = np.angle(label_data_npy)

# 画像の処理時間計測終了
processing_time = time.time() - start_time

# 時間表示
print(f"画像の処理時間: {processing_time} 秒")

# フィギュアを作成
fig = plt.figure()

# 再生計算
SLM_data = np.exp(i * phase_output)
reconst = nearpropCONV(SLM_data, Nx, Ny, dx, dy, 0, 0, wav_len, -1.0 * ((i) * dz * 10**times + initial_place))

mpimg.imsave(f'./{folder_name}/image_{(n*10 + i):05d}.png', np.abs(reconst), cmap='gray')

