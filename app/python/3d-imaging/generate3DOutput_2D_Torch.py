import numpy as np
import cv2
import matplotlib.pyplot as plt
import time
import concurrent.futures
import os
import torch
import torch.fft
from torch.multiprocessing import Pool, Process, set_start_method
import init
import sys

def isAllZeros(image):
    nonzero_indices = torch.nonzero(torch.abs(image) > 0)

    if len(nonzero_indices) > 0:
        print("非ゼロ要素が見つかりました。")
        for index in nonzero_indices:
            value = image[index[0], index[1]]
            print(f"非ゼロ要素のインデックス: {index}, 値: {value}")
    else:
        print("全ての要素が0です。")

def nearpropCONV_torch(input_image, initializations, d):
    if d == 0:
        Recon = input_image
    else:
        # x1, x2 = -sizex//2, sizex//2-1
        # y1, y2 = -sizey//2, sizey//2-1
        # Fx, Fy = torch.meshgrid(torch.arange(x1, x2+1, device=Comp1.device), torch.arange(y1, y2+1, device=Comp1.device))

        # Fcomp1 = torch.fft.fftshift(torch.fft.fft2(Comp1)) / torch.sqrt(torch.tensor(sizex * sizey, dtype=torch.float32, device=Comp1.device))

        # FresR = torch.exp(-1j * torch.pi * wa * d * ((Fx**2) / ((dx * sizex)**2) + (Fy**2) / ((dy * sizey)**2)))

        # Fcomp2 = Fcomp1 * FresR
        # Recon = torch.fft.ifft2(torch.fft.ifftshift(Fcomp2)) * torch.sqrt(torch.tensor(sizex * sizey, dtype=torch.float32, device=Comp1.device))
        Recon=input_image
        isAllZeros(Recon)

    return Recon

def generate_random_phase(shape):
    return (np.random.rand(*shape) - 0.5) * 2.0 * 2.0 * np.pi

def process_image_torch(gpu_id, images, initializations):
    device = torch.device(f'cuda:{gpu_id}')
    initial_phase = [torch.tensor(generate_random_phase(images[n].shape), dtype=torch.complex64, device=device) for n in range(len(images))]
    output_images = [torch.zeros((initializations.Nx, initializations.Ny), dtype=torch.complex64, device=device) for _ in range(len(images))]
    initializations.device = device
    for n in range(len(images)):
        input_image = images[n].to(device) * torch.exp(1j * initial_phase[n])
        d = (n+1) * initializations.dz * initializations.pixels + initializations.initial_place
        output_images[n] = nearpropCONV_torch(input_image, initializations, d)

    return output_images

# def process_on_gpu(gpu_id, images, output_images):
#     device = torch.device(f'cuda:{gpu_id}')
#     local_output_images = [img.to(device) for img in output_images]
#     local_images = [img.to(device) for img in images]
    
#     with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
#         futures = [executor.submit(process_image_torch, n, local_images, Nx, Ny, dx, dy, wav_len, local_output_images, device) for n in range(num_images) if n % num_gpus == gpu_id]
#         concurrent.futures.wait(futures)
    
#     return local_output_images

def cal_save_image_torch(i, SLM_data, Nx, Ny, dx, dy, wav_len, initial_place, times):
    reconst_2d = nearpropCONV_torch(SLM_data, Nx, Ny, dx, dy, 0, 0, wav_len, -1.0 * ((i) * dz * 10**times + initial_place))
    print("process：", i+1)
    return reconst_2d

def reconstruct_on_gpu(gpu_id):
    device = torch.device(f'cuda:{gpu_id}')
    SLM_data_gpu = SLM_data.to(device)
    local_reconst_3d = torch.zeros((Nx, Ny, num_images // num_gpus), dtype=torch.float32, device=device)
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(cal_save_image_torch, i, SLM_data_gpu, Nx, Ny, dx, dy, wav_len, initial_place, times) for i in range(gpu_id, num_images, num_gpus)]
        for i, future in enumerate(concurrent.futures.as_completed(futures)):
            print(i)
            local_reconst_3d[:, :, i] = future.result()
    
    return local_reconst_3d.cpu()

if __name__ == "__main__":
    try:
        set_start_method('spawn')
    except RuntimeError:
        pass

    # 必要なパラメータの設定
    initializations = init.Initialization()
    i, wav_len, Nx, Ny, dx, dy, dz, wav_num, times, initial_place, pixels, box_number, num_images, num_gpus = initializations.i, initializations.wav_len, initializations.Nx, initializations.Ny, initializations.dx, initializations.dy, initializations.dz, initializations.wav_num, initializations.times, initializations.initial_place, initializations.pixels, initializations.box_number, initializations.num_images, initializations.num_gpus

    # フィギュアの作成
    fig = plt.figure()

    # 画像読み込み時間の計測開始
    start_time = time.time()

    # フォルダの作成
    folder_name = f'C:\\Users\\Owner\\mizusaki\\3d-holography\\app\\python\\3d-imaging\\output\\RawOutputData_{num_images}_{Nx}x{Ny}_10{times}_pixels={pixels}_d={dz *10**times}_initialPlace{initial_place}'
    os.makedirs(folder_name, exist_ok=True)

    # 画像の読み込みとリサイズ、PyTorchテンソルへの変換
    images = [cv2.resize(cv2.imread(f'C:\\Users\\Owner\\mizusaki\\3d-holography\\app\\python\\3d-imaging\\src\\Original2Dimages_{box_number}_{pixels}px_{Nx}x{Ny}x{num_images}\\image_{(i):05d}.png', cv2.IMREAD_GRAYSCALE).astype(float), (Nx, Ny)) for i in range(1, num_images + 1)]
    part_size = num_images // num_gpus
    images_parts = [
    [torch.tensor(img, dtype=torch.float32) for img in images[i * part_size : (i + 1) * part_size]]
    for i in range(num_gpus)
    ]
    args = [(i, images_parts[i], initializations) for i in range(num_gpus)]

    # 出力画像の初期化
    output_images = [torch.zeros((Nx, Ny), dtype=torch.complex64) for _ in range(num_images)]

    # マルチプロセシングプールを利用して各GPUで並列処理を実行
    with Pool(processes=num_gpus) as pool:
        all_results = pool.starmap(process_image_torch, args)

    # GPUからCPUに結果を戻し、合計する
    output_images = [img.cpu() for sublist in all_results for img in sublist]
    total_output_images = torch.sum(torch.stack(output_images), axis=0)
    print(total_output_images)

    sys.exit()

    # 振幅と位相分布の計算
    amplitude_output = torch.abs(total_output_images)
    phase_output = torch.angle(total_output_images)

    # 画像の処理時間計測終了
    processing_time = time.time() - start_time
    print(f"画像の処理時間: {processing_time} 秒")

    # フィギュアを作成
    fig = plt.figure()

    # 再生計算
    SLM_data = torch.exp(i * phase_output).to(torch.device('cuda'))

    # 3D配列を初期化
    reconst_3d = torch.zeros((Nx, Ny, num_images), dtype=torch.float32, device=torch.device('cuda'))

    # マルチプロセシングプールを利用して各GPUで並列処理を実行
    with Pool(processes=num_gpus) as pool:
        all_results = pool.map(reconstruct_on_gpu, range(num_gpus))

    # GPUからCPUに結果を戻し、3D配列に統合
    reconst_3d = torch.cat(all_results, dim=2)
    file_path = os.path.join(folder_name, 'random_data.npy')
    np.save(file_path, reconst_3d.numpy())
    print('3D array saved to reconst_3d.npy')
