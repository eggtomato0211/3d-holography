import numpy as np
import cv2
import matplotlib.pyplot as plt
import time

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

# 波長や画像サイズなどのパラメータ
i = 1j
wav_len = 532.0 * 10**-9
Nx, Ny = 1024, 1024
dx = 3.45 * 10**-6
dy = dx
wav_num = 2 * np.pi / wav_len

# 画像の読み込み時間計測開始
start_time = time.time()

name = 'number'

# 画像の枚数
num_images = 4

# 1つの画像の伝搬計算した結果を入れる配列
output_images = np.zeros((num_images, Nx, Ny), dtype=complex)  # 初期化修正

# 画像の読み込みとリサイズ
images = [cv2.resize(cv2.imread(f'./src/{name}_{i:04d}.bmp', cv2.IMREAD_GRAYSCALE).astype(float), (Nx, Ny)) for i in range(1, num_images + 1)]

for n in range(num_images):
    # ランダム位相分布
    initial_phase = generate_random_phase(images[n].shape)

    # 入力画像
    input_image = images[n] * np.exp(i * initial_phase)

    d = (n+1) * (256/num_images)  # 画像の枚数によって距離を変更

    # nearpropCONVは光波の空間伝搬を計算するフレネル伝搬計算
    output_images[n] = nearpropCONV(input_image, Nx, Ny, dx, dy, 0, 0, wav_len, d)

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

# 位相分布の表示
plt.figure(4)
plt.imshow(phase_output, cmap='gray')
plt.title('Phase Distribution')
plt.show()

# 再生計算
SLM_data = np.exp(i * phase_output)

#それぞれの距離での再生像を表示したい場合
for i in range(num_images):
    reconst = nearpropCONV(SLM_data, Nx, Ny, dx, dy, 0, 0, wav_len, -1.0 * (i+1) * (256/num_images))
    plt.figure(5 + i)
    plt.imshow(np.abs(reconst), cmap='gray')
    plt.title(f'Reconstruction at Distance -d{i+1}')
    plt.show()