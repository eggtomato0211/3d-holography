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

# 画像の読み込み
data01 = cv2.imread(f'./src/{name}_0001.bmp', cv2.IMREAD_GRAYSCALE).astype(float)
data01 = cv2.resize(data01, (data01.shape[1]*4, data01.shape[0]*4))

data02 = cv2.imread(f'./src/{name}_0002.bmp', cv2.IMREAD_GRAYSCALE).astype(float)
data02 = cv2.resize(data02, (data02.shape[1]*4, data02.shape[0]*4))

# ランダム位相分布
initial_phase1 = (np.random.rand(*data01.shape) - 0.5) * 2.0 * 2.0 * np.pi
initial_phase2 = (np.random.rand(*data02.shape) - 0.5) * 2.0 * 2.0 * np.pi

# 入力画像
input1 = data01 * np.exp(i * initial_phase1)
input2 = data02 * np.exp(i * initial_phase2)

d1, d2 = 50.0, 100.0

# 画像の表示
# plt.figure(1)
# plt.imshow(data01, cmap='gray')
# plt.title('Image 1')
# plt.show()

# plt.figure(2)
# plt.imshow(data02, cmap='gray')
# plt.title('Image 2')
# plt.show()

# nearpropCONVは光波の空間伝搬を計算するフレネル伝搬計算
output1 = nearpropCONV(input1, Nx, Ny, dx, dy, 0, 0, wav_len, d1)
output2 = nearpropCONV(input2, Nx, Ny, dx, dy, 0, 0, wav_len, d2)

# 2つの光波の加算
output = output1 + output2

print(output1)
print(output2)
print(output)

# 振幅と位相分布の計算
amplitude_output = np.abs(output)
phase_output = np.angle(output)

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

# 距離-d1での再生像
reconst1 = nearpropCONV(SLM_data, Nx, Ny, dx, dy, 0, 0, wav_len, -1.0 * d1)
plt.figure(5)
plt.imshow(np.abs(reconst1), cmap='gray')
plt.title('Reconstruction at Distance -d1')
plt.show()

# 距離-d2での再生像
reconst2 = nearpropCONV(SLM_data, Nx, Ny, dx, dy, 0, 0, wav_len, -1.0 * d2)
plt.figure(6)
plt.imshow(np.abs(reconst2), cmap='gray')
plt.title('Reconstruction at Distance -d2')
plt.show()
