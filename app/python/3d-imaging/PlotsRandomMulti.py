import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import time
import concurrent.futures
import os
import matplotlib.image as mpimg
from HDF import HDF

def load_image(Nx, Ny):
    #iを1から10000まででランダムな数値を取得
    i = np.random.randint(1, 10000)
    path = f'C:\\Users\\Owner\\mizusaki\\3d-holography\\app\\python\\3d-imaging\\src\\number_{Nx}x{Ny}\\number_{i:05d}.png'
    return cv2.imread(path, cv2.IMREAD_GRAYSCALE).astype(float)

def load_images(Nx, Ny, channels, depthlevel):
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [executor.submit(load_image, Nx, Ny) for i in range(1, channels * depthlevel + 1)]
        images = [future.result() for future in concurrent.futures.as_completed(futures)]
    
    # 元の順序に戻す
    images.sort(key=lambda x: futures.index(next(future for future in futures if future.result() is x)))
    return images

def load_one_image(Nx, Ny, channels, depthlevel, image_number, pixel):
    images = np.zeros((depthlevel*channels + 1, Nx, Ny), dtype=float)
    for n in range(0, channels):
        for k in range(0, image_number):
            print(f"channel: {n}, image: {k}")
            #iを1から10000まででランダムな数値を取得
            i = np.random.randint(1, 10000)
            path = f'C:\\Users\\Owner\\mizusaki\\3d-holography\\app\\python\\3d-imaging\\src\\{Nx}x{Ny}x{depthlevel}_{pixel}px\\image_{i:05d}.png'
            image = cv2.imread(path, cv2.IMREAD_GRAYSCALE).astype(float)
            #depthに代入
            for l in range(0, depthlevel):
                depth = np.random.randint(0, depthlevel)
                if images[n*depthlevel + depth].sum() == 0:
                    images[n*depthlevel + depth] = image
                    break 
    return images

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
def generate_random_phase(shape, random_mode):
    #もしmodeがTrueの場合
    if random_mode == True:
        return (np.random.rand(*shape) - 0.5) * 2.0 * 2.0 * np.pi
    
    #modeがFalseの場合
    return 1.0

def process_image(args):
    n, images, sizex, sizey, dx, dy, wav_len, initial_place, dz, random_mode = args
    initial_phase = generate_random_phase(images[n].shape, random_mode)
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
dz = 4 * 10**-6
wav_num = 2 * np.pi / wav_len
times = -4 
initial_place = 10**times
pixels = 3

# フィギュアを作成
fig = plt.figure()

# 画像の読み込み時間計測開始
start_time = time.time()

# z軸の枚数
depthlevel = 128

#channelの数
channels = 1
init_channel = 0

#(channels, depthlevel, Nx, Ny)の配列を作成
raw_data = np.zeros((depthlevel, Nx, Ny), dtype=float)
label_data = np.zeros((depthlevel, Nx, Ny), dtype=float)

one_image = True

#random位相を使用するか
random_mode = False

# 画像を何枚配置するかどうか
image_number = 3

if one_image:
    images = load_one_image(Nx, Ny, channels, depthlevel, image_number, pixels)
    print("One image loaded.")
    # フォルダを作成
    folder_name = f'.\\app\\python\\3d-imaging\\output\\PlotImages={image_number}_{Nx}x{Ny}x{depthlevel}_pixecls={pixels}_d={dz}_RandomPhase={random_mode}'
    os.makedirs(folder_name, exist_ok=True)
    output_folder_name = rf'.\app\python\3d-imaging\hdf\PlotImages={image_number}_{Nx}x{Ny}x{depthlevel}_pixecls={pixels}_d={dz}_RandomPhase={random_mode}'
else:
    images = load_images(Nx, Ny, channels, depthlevel)
    print("Multi Images loaded.")
    # フォルダを作成
    folder_name = f'.\\app\\python\\3d-imaging\\output\\All_{Nx}x{Ny}x{depthlevel}_pixecls={pixels}_d={dz}_RandomPhase={random_mode}'
    os.makedirs(folder_name, exist_ok=True)
    output_folder_name = rf'.\app\python\3d-imaging\hdf\All_{Nx}x{Ny}x{depthlevel}_pixecls={pixels}_d={dz}_RandomPhase={random_mode}'

print("計算を開始します。")

for channel in range(init_channel, init_channel + channels):
    
    for depth in range(depthlevel):
        label_data[depth, :, :] = images[channel * depthlevel + depth]

    # 並列処理の引数を準備
    args_list = [(n, images, Nx, Ny, dx, dy, wav_len, initial_place, dz, random_mode) for n in range(channel*depthlevel, (channel+1)*depthlevel)]

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
        raw_data[depth, :, :] = np.abs(results[depth])
        output_image_path = os.path.join(folder_name, f"reconstructed_{depthlevel*channel+depth+1:05d}.png")
        plt.imsave(output_image_path, np.abs(results[depth]), cmap='gray')

    # #Label_dataの1つ表示
    # plt.imshow(np.abs(label_data[1, :, :]), cmap='gray')
    # plt.show()

    # #Raw_dataの1つ表示
    # plt.imshow(np.abs(raw_data[1, :, :]), cmap='gray')
    # plt.show()

    # HDF5ファイルに保存
    # インスタンスを作成
    hdf_maker = HDF(Nx, Ny, depthlevel, dz, output_folder_name)
    hdf_maker.makeHDF(raw_data, label_data, f"NumberFrom{channel*depthlevel+1}.h5")