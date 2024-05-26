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
    d = (n+1) * dz *10**times + initial_place
    output_images[n] = nearpropCONV(input_image, sizex, sizey, dx, dy, 0, 0, wav_len, d)

# 波長や画像サイズなどのパラメータ
i = 1j
wav_len = 532.0 * 10**-9
Nx, Ny = 1024, 1024
dx = 3.45 * 10**-6
dy = dx
dz = 1
wav_num = 2 * np.pi / wav_len
times = -4 
initial_place = (10**times)*1000
pixels = 10

# フィギュアを作成
fig = plt.figure()

# 画像の読み込み時間計測開始
start_time = time.time()

name = 'number'

# 画像の枚数
num_images = 10
total_images = 10000

# フォルダを作成
folder_name = f'output/generateThreadOutputDataSet_{num_images}_{Nx}x{Ny}_10{times}_pixels={pixels}_d={dz *10**times}_initialPlace{initial_place}'
os.makedirs(folder_name, exist_ok=True)

for n in range(0, 1):
    # 出力画像の初期化
    output_images = [np.zeros((Nx, Ny), dtype=np.complex128) for _ in range(num_images)]

    # 画像の読み込みとリサイズ
    images = [cv2.resize(cv2.imread(f'./src/generated_original_images_{total_images}_{pixels}_{Nx}x{Ny}/image_{(n*10 + i):05d}.png', cv2.IMREAD_GRAYSCALE).astype(float), (Nx, Ny)) for i in range(1, num_images + 1)]

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

    def cal_save_image(i, num_images, SLM_data, Nx, Ny, dx, dy, wav_len, initial_place, times, folder_name):
        reconst = nearpropCONV(SLM_data, Nx, Ny, dx, dy, 0, 0, wav_len, -1.0 * ((i) * dz * 10**times + initial_place))
        print("process：", i+1)
        mpimg.imsave(f'./{folder_name}/image_{(n*10 + i):05d}.png', np.abs(reconst), cmap='gray')

    # 並列処理
    with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:
        start = time.time()
        # 各画像に対してcal_save_image関数を並列に実行
        futures = [executor.submit(cal_save_image, i, num_images, SLM_data, Nx, Ny, dx, dy, wav_len, initial_place, times, folder_name) for i in range(1, num_images+1)]
        # すべてのタスクの完了を待つ
        concurrent.futures.wait(futures)
        end = time.time()
        print('マルチスレッド: TIME {:.4f}\n'.format(end - start))