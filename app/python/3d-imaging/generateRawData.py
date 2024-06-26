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
Nx, Ny = 128, 128
dx = 3.45 * 10**-6
dy = dx
dz = 3.45
wav_num = 2 * np.pi / wav_len
initial_place = 10

# 深さのレベル
depthLevel = 16

# チャンネル数
channels = 1

# 画像の読み込み時間計測開始
start_time = time.time()

total_number = depthLevel * channels
box_number = 4
voxel_size = 2

# 3次元の伝搬計算した結果を入れる配列
output_images = np.zeros((depthLevel, Nx, Ny), dtype=complex)  # 初期化修正

# 画像の読み込み
images = [cv2.imread(f'C:\\Users\\Owner\\mizusaki\\3d-holography\\app\python\\3d-imaging\\src\\number_{Nx}x{Ny}\\number_{(i):05d}.png', cv2.IMREAD_GRAYSCALE).astype(float) for i in range(1, total_number + 1)]

for channel in range(channels):
    for n in range(depthLevel):
        # ランダム位相分布
        initial_phase = generate_random_phase(images[n + channel*depthLevel].shape)

        # 入力画像
        input_image = images[n + channel*depthLevel] * np.exp(1j * initial_phase)

        d = initial_place + n * dz

        # nearpropCONVは光波の空間伝搬を計算するフレネル伝搬計算
        output_images[n] = nearpropCONV(input_image, Nx, Ny, dx, dy, 0, 0, wav_len, d)

        print(f"画像{n + channel*depthLevel + 1}の処理が完了しました。")

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

    # 再生計算
    SLM_data = np.exp(i * phase_output)

    # フィギュアを作成
    fig = plt.figure()

    #それぞれの距離での再生像を表示したい場合
    for n in range(depthLevel):
        d = initial_place + n * dz
        reconst = nearpropCONV(SLM_data, Nx, Ny, dx, dy, 0, 0, wav_len, -1.0 * d)
        plt.imshow(np.abs(reconst), cmap='gray') # 画像を表示
        plt.show()