import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import time
import concurrent.futures
import multiprocessing as mp
from threading import Thread
import os

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

def process_image(n, images, sizex, sizey, dx, dy, wav_len, output_images, num_images):
    initial_phase = generate_random_phase(images[n].shape)
    input_image = images[n] * np.exp(1j * initial_phase)
    d = (n+1) * (256/num_images)*10**times + initial_place
    output_images[n] = nearpropCONV(input_image, sizex, sizey, dx, dy, 0, 0, wav_len, d)

# 波長や画像サイズなどのパラメータ
i = 1j
wav_len = 532.0 * 10**-9
Nx, Ny = 1024, 1024
dx = 3.45 * 10**-6
dy = dx
wav_num = 2 * np.pi / wav_len
times = -3
initial_place = (10**times)*1000

# フィギュアを作成
fig = plt.figure()

# 画像の読み込み時間計測開始
start_time = time.time()

name = 'number'

# 画像の枚数
num_images = 128

# 出力画像の初期化
output_images = [np.zeros((Nx, Ny), dtype=np.complex128) for _ in range(num_images)]

# 画像の読み込みとリサイズ
images = [cv2.resize(cv2.imread(f'./src/{name}_{i:04d}.bmp', cv2.IMREAD_GRAYSCALE).astype(float), (Nx, Ny)) for i in range(1, num_images + 1)]

# 並列処理
with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:
    # 各画像に対してprocess_image関数を並列に実行
    futures = [executor.submit(process_image, n, images, Nx, Ny, dx, dy, wav_len, output_images, num_images) for n in range(num_images)]

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

# フォルダを作成
folder_name = f'animation_concurrent_{num_images}_10{times}_initialPlace{initial_place}'
os.makedirs(folder_name, exist_ok=True)

def update(i): 
    start_time = time.time() 
    reconst = nearpropCONV(SLM_data, Nx, Ny, dx, dy, 0, 0, wav_len, -1.0 * ((i+1) * (256*10**times/num_images) + initial_place)) 
    plt.imshow(np.abs(reconst), cmap='gray') 
    # 画像の処理時間計測終了 
    processing_time = time.time() - start_time
      # フレームをファイルとして保存
    plt.savefig(f'./{folder_name}/animation_concurrent_{num_images}_10{times}_initialPlace{initial_place}_{i+1}.png') 
    # 時間表示 
    print(f"画像の処理時間: {processing_time} 秒")

#アニメーションオブジェクトの作成
movie = animation.FuncAnimation(fig, update, frames=num_images, interval=1000)
movie.save(f'./{folder_name}/animation_concurrent_{num_images}_10{times}_initialPlace{initial_place}.gif', writer='pillow')
# def parallel_update(i, SLM_data, Nx, Ny, dx, dy, wav_len, num_images):
#     start_time = time.time()
#     reconst = nearpropCONV(SLM_data, Nx, Ny, dx, dy, 0, 0, wav_len, -1.0 * (i+1) * (256/num_images))
#     processing_time = time.time() - start_time
#     print(f"Image {i} processing time: {processing_time} seconds")
#     return reconst

# def display_images(results):
#     reconst = [r.get() for r in results]
#     plt.imshow(np.abs(np.array(reconst)), cmap='gray')
#     plt.pause(0.01)
    
# pool = mp.Pool(processes=4)
# results = [pool.apply_async(parallel_update, args=(i, SLM_data, Nx, Ny, dx, dy, wav_len, num_images)) for i in range(num_images)]

# display_thread = Thread(target=lambda: display_images(results))
# display_thread.start()

# pool.close()
# pool.join()
# display_thread.join()