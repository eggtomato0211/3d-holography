import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import time
import concurrent.futures
import os
import matplotlib.image as mpimg
from HDF import HDF

def load_image(i, Nx, Ny, depthlevel, pixels, number_of_plots):
    path = f'C:\\Users\\Owner\\mizusaki\\3d-holography\\app\\python\\3d-imaging\\src\\{Nx}x{Ny}x{depthlevel}_{pixels}pxx{number_of_plots}\\image_{i:05d}.png'
    return cv2.imread(path, cv2.IMREAD_GRAYSCALE).astype(float)

def load_images(Nx, Ny, pixels, number_of_plots, channels, depthlevel):
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [executor.submit(load_image, i, Nx, Ny, depthlevel, pixels, number_of_plots) for i in range(1, channels * depthlevel + 1)]
        images = [future.result() for future in concurrent.futures.as_completed(futures)]
    
    # 元の順序に戻す
    images.sort(key=lambda x: futures.index(next(future for future in futures if future.result() is x)))
    return images

def nearpropCONV(Comp1, sizex, sizey, dx, dy, wa, d):
    if d == 0:
        print("d == 0")
        # print(np.max(Comp1))
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
def generate_random_phase(shape, mode=None):
    #もしmodeがTrueの場合
    if random_mode == True:
        return (np.random.rand(*shape) - 0.5) * 2.0 * 2.0 * np.pi
    
    #modeがFalseの場合
    np.random.seed(42)
    # print("Random phase is not used.")
    return (np.random.rand(*shape) - 0.5) * 2.0 * 2.0 * np.pi

def process_image(args):
    n, images, sizex, sizey, dx, dy, wav_len, dz, random_mode, channel = args
    # initial_phase = generate_random_phase(images[n].shape, random_mode)
    input_image = images[n + channel * depthlevel] * np.exp(1j)
    output_3dimage = np.zeros((depthlevel, Nx, Ny), dtype=float)
    for z in range(depthlevel):
        output_3dimage[z] = np.abs(nearpropCONV(input_image, sizex, sizey, dx, dy, wav_len, (n - z) * dz))**2
    # print(np.max(output_3dimage))
    return output_3dimage


# 波長や画像サイズなどのパラメータ
i = 1j
wav_len = 532.0 * 10**-9
Nx, Ny = 32, 32
dx = 3.45 * 10**-6
dy = dx
dz = 4 * 10**-6
wav_num = 2 * np.pi / wav_len
pixels = 1
number_of_plots = 5

# フィギュアを作成
fig = plt.figure()

# 画像の読み込み時間計測開始
start_time = time.time()

# z軸の枚数
depthlevel = 128

#channelの数
channels = 78
init_channel = 0

image_number = 64

#random位相を使用するか
random_mode = False

# フォルダを作成
folder_name = f'.\\app\\python\\3d-imaging\\output\\Random_{Nx}x{Ny}x{depthlevel}_d={dz}_pixels={pixels}_{number_of_plots}plots_{image_number}images'
os.makedirs(folder_name, exist_ok=True)

output_hdfdir = rf'.\app\\python\3d-imaging\hdf\Random_{Nx}x{Ny}x{depthlevel}_d={dz}_pixels={pixels}_0-{number_of_plots}plots_{image_number}images'

# 画像の読み込み
images = load_images(Nx, Ny, pixels, number_of_plots, channels, depthlevel)

print("計算を開始します。")

for channel in range(init_channel, init_channel + channels):
    #(channels, depthlevel, Nx, Ny)の配列を作成
    raw_data = np.zeros((depthlevel, Nx, Ny), dtype=float)
    label_data = np.zeros((depthlevel, Nx, Ny), dtype=float)

    # image_numberの数だけ0~depthlevelの間のrandomなdepthを選択
    random_depths = np.random.randint(0, depthlevel, image_number)

    print(f"Process: {channel+1}")
    for depth in random_depths:
        label_data[depth, :, :] = images[channel * depthlevel + depth]

    # 並列処理の引数を準備
    args_list = [(n, images, Nx, Ny, dx, dy, wav_len, dz, random_mode, channel) for n in random_depths]

    # 並列処理
    with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:
        output_3dimages = list(executor.map(process_image, args_list))
    
    # output_3dimagesをNumPy配列に変換
    output_3dimages = np.array(output_3dimages)
    
    # 合計を計算してraw_dataに追加
    for depth in range(depthlevel):
        raw_data[depth, :, :] = np.sum(output_3dimages[:, depth, :, :], axis=0)
        print(np.max(raw_data[depth]))


    # for k in range(depthlevel):
    #     if k == 0:
    #         max_value = np.max(np.abs(raw_data[k]))
    #         min_value = np.min(np.abs(raw_data[k]))
    #     # すべてのフレームの中で一番大きい値、小さい値を取得
    #     tmp_max_value = np.max(np.abs(raw_data[k]))
    #     tmp_min_value = np.min(np.abs(raw_data[k]))
    #     # 前のmax_value, min_valueと比較して大きい値、小さい値を取得
    #     if tmp_max_value > max_value:
    #         max_value = tmp_max_value
    #     if tmp_min_value < min_value:
    #         min_value = tmp_min_value
    
    # print(f"max_value: {max_value}, min_value: {min_value}")
    
    # 結果を配列に格納
    for depth in range(depthlevel):
        # raw_data[depth, :, :] = ((np.abs(raw_data[depth]) - min_value) / (max_value - min_value) * 255).astype('uint8')
        output_image_path = os.path.join(folder_name, f"reconstructed_{depthlevel*channel+depth+1:05d}.png")
        plt.imsave(output_image_path, np.abs(raw_data[depth]), cmap='gray')
    
    min_val = np.min(raw_data)
    max_val = np.max(raw_data)

    # 正規化を行う
    raw_data = (raw_data - min_val) / (max_val - min_val)

    min_val = np.min(label_data)
    max_val = np.max(label_data)

    label_data = (label_data - min_val) / (max_val - min_val)

    # # Label_dataの1つ表示
    # plt.imshow(np.abs(label_data[1, :, :]), cmap='gray')
    # plt.show()

    # # Raw_dataの1つ表示
    # plt.imshow(np.abs(raw_data[1, :, :]), cmap='gray')
    # plt.show()

    # 128*128ではないとき、raw_dataとlabel_dataを128*128にリサイズする
    if Nx != 128:
        resized_raw_data = np.zeros((depthlevel, 128, 128), dtype=float)
        resized_label_data = np.zeros((depthlevel, 128, 128), dtype=float)
        
        for i in range(depthlevel):
            # bilinear補間を使用してリサイズ
            resized_raw_data[i] = cv2.resize(raw_data[i], (128, 128), interpolation=cv2.INTER_LINEAR)
            resized_label_data[i] = cv2.resize(label_data[i], (128, 128), interpolation=cv2.INTER_LINEAR)
        
        raw_data = resized_raw_data
        label_data = resized_label_data
        
        print(f"データを 32x32x{depthlevel} から 128x128x{depthlevel} にリサイズしました。")

    # HDF5ファイルに保存
    # インスタンスを作成
    hdf_maker = HDF(128, 128, depthlevel, dz, output_hdfdir)
    hdf_maker.makeHDF(raw_data, label_data, f"NumberFrom{channel*depthlevel+1}.h5")