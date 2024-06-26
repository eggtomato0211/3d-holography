import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import time
import concurrent.futures
import os
import matplotlib.image as mpimg

def nearpropCONV(Comp1, sizex, sizey, dx, dy, wa, d):
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

def process_image(args):
    n, images, sizex, sizey, dx, dy, wav_len, initial_place, dz = args
    initial_phase = generate_random_phase(images[n].shape)
    input_image = images[n] * np.exp(1j * initial_phase)
    d = (n % depthlevel + 1) * dz + initial_place
    output_image = nearpropCONV(input_image, sizex, sizey, dx, dy, wav_len, d)
    return output_image

def cal_save_image(args):
    l, SLM_data, Nx, Ny, dx, dy, wav_len, initial_place = args
    reconst_2d = nearpropCONV(SLM_data, Nx, Ny, dx, dy, wav_len, -1.0 * ((l % depthlevel +1) * dz  + initial_place))
    print("process：", l)
    #  # 画像を保存
    # output_image_path = os.path.join(folder_name, f"reconstructed_{l+1:05d}.png")
    # plt.imsave(output_image_path, np.abs(reconst_2d), cmap='gray')

    return reconst_2d

# 波長や画像サイズなどのパラメータ
i = 1j
wav_len = 532.0 * 10**-9
Nx, Ny = 128, 128
dx = 3.45 * 10**-6
dy = dx
dz = 3.45
wav_num = 2 * np.pi / wav_len
times = -4 
initial_place = (10**times)*1000

# フィギュアを作成
fig = plt.figure()

# 画像の読み込み時間計測開始
start_time = time.time()

# z軸の枚数
depthlevel = 128

#channelの数
channels = 16
init_channel = 16

#(channels, depthlevel, Nx, Ny)の配列を作成
raw_data = np.zeros((channels, depthlevel, Nx, Ny), dtype=complex)
label_data = np.zeros((channels, depthlevel, Nx, Ny), dtype=complex)

# フォルダを作成
folder_name = f'.\\app\\python\\3d-imaging\\output\\RawOutputData_{depthlevel}_{Nx}x{Ny}_d={dz}'
os.makedirs(folder_name, exist_ok=True)


# 画像の読み込み
images = [cv2.imread(f'C:\\Users\\Owner\\mizusaki\\3d-holography\\app\\python\\3d-imaging\\src\\number_{Nx}x{Ny}\\number_{(i):05d}.png', cv2.IMREAD_GRAYSCALE).astype(float) for i in range(1, 10000)]

#Label_dataの作成
for channel in range(init_channel, init_channel + channels):
    for depth in range(depthlevel):
        label_data[channel-init_channel, depth, :, :] = images[channel * depthlevel + depth]

print("計算を開始します。")

for channel in range(init_channel, init_channel + channels):
    # 並列処理の引数を準備
    args_list = [(n, images, Nx, Ny, dx, dy, wav_len, initial_place, dz) for n in range(channel*depthlevel, (channel+1)*depthlevel)]

    # 並列処理
    with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:
        output_images = list(executor.map(process_image, args_list))

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

    # 並列処理の引数を準備
    cal_args_list = [(l, SLM_data, Nx, Ny, dx, dy, wav_len, initial_place) for l in range(channel*depthlevel, (channel+1)*depthlevel)]

    # 並列処理
    with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:
        start = time.time()
        results = list(executor.map(cal_save_image, cal_args_list))
        end = time.time()
        print('マルチスレッド: TIME {:.4f}\n'.format(end - start))
    
    # 結果を配列に格納
    for depth in range(depthlevel):
        raw_data[channel - init_channel, depth, :, :] = np.abs(results[depth])
        output_image_path = os.path.join(folder_name, f"reconstructed_{depthlevel*channel+depth+1:05d}.png")
        plt.imsave(output_image_path, np.abs(results[depth]), cmap='gray')

# Label_dataの1つ表示
plt.imshow(np.abs(label_data[1, 13, :, :]), cmap='gray')
plt.show()

#Raw_dataの1つ表示
plt.imshow(np.abs(raw_data[1, 13, :, :]), cmap='gray')
plt.show()

#HDF5ファイルに保存
from makeHDF import makeHDF
# インスタンスを作成
hdf_maker = makeHDF()
hdf_maker.makeHDF(raw_data, label_data, f"val_channels={channels}_{Nx}x{Ny}_{depthlevel}_d={dz}.h5")
