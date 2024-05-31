import numpy as np
import cv2
import matplotlib.pyplot as plt
import time
import concurrent.futures
import os
import torch
import torch.fft
from torch.multiprocessing import Pool, Process, set_start_method
import sys

# 使用可能なGPUの数を確認
num_gpus = torch.cuda.device_count()
print(f"使用可能なGPUの数: {num_gpus}")

# GPUが少なくとも1つ利用可能か確認
if num_gpus < 1:
    raise ValueError("GPUが検出されませんでした。少なくとも1つのGPUが必要です。")

def nearpropCONV_torch(Comp1, sizex, sizey, dx, dy, shiftx, shifty, wa, d):
    if d == 0:
        Recon = Comp1
    else:
        x1, x2 = -sizex // 2, sizex // 2 - 1
        y1, y2 = -sizey // 2, sizey // 2 - 1
        Fx, Fy = torch.meshgrid(torch.arange(x1, x2 + 1, device=Comp1.device), torch.arange(y1, y2 + 1, device=Comp1.device), indexing='xy')

        Fcomp1 = torch.fft.fftshift(torch.fft.fft2(Comp1)) / torch.sqrt(torch.tensor(sizex * sizey, dtype=torch.complex64, device=Comp1.device))

        FresR = torch.exp(-1j * torch.pi * wa * d * ((Fx ** 2) / ((dx * sizex) ** 2) + (Fy ** 2) / ((dy * sizey) ** 2)))

        Fcomp2 = Fcomp1 * FresR
        Recon = torch.fft.ifft2(torch.fft.ifftshift(Fcomp2)) * torch.sqrt(torch.tensor(sizex * sizey, dtype=torch.complex64, device=Comp1.device))

    return Recon

def generate_random_phase(shape, device):
    return (torch.rand(*shape, device=device) - 0.5) * 2.0 * 2.0 * torch.pi

def process_image(n, images, sizex, sizey, dx, dy, dz, wav_len, device, gpu_id, pixels, initial_place):
    initial_phase = generate_random_phase(images[n].shape, device)
    input_image = images[n] * torch.exp(1j * initial_phase)
    d = (gpu_id*num_gpus+n+1) * dz * pixels + initial_place
    output_image = nearpropCONV_torch(input_image, sizex, sizey, dx, dy, 0, 0, wav_len, d)
    return output_image

def process_images(gpu_id, images, Nx, Ny, dx, dy, dz, wav_len, pixels, initial_place, num_images):
    device = f'cuda:{gpu_id}'
    local_output_images = [torch.zeros((Nx, Ny), dtype=torch.complex64, device=device) for _ in range(num_images // num_gpus)]
    for n in range(num_images):
        if n % num_gpus == gpu_id:
            local_output_images[n // num_gpus] = process_image(n, images, Nx, Ny, dx, dy, dz, wav_len, device, gpu_id, pixels, initial_place)
    return local_output_images

def cal_save_image_torch(i, SLM_data, Nx, Ny, dx, dy, wav_len, initial_place, pixels, dz, device):
    d = (i + 1) * dz * pixels + initial_place
    reconst_2d = nearpropCONV_torch(SLM_data, Nx, Ny, dx, dy, 0, 0, wav_len, (-1) * d)
    return reconst_2d

def reconstruct_on_gpu(gpu_id, SLM_data, Nx, Ny, num_images, dx, dy, dz, wav_len, initial_place, pixels):
    device = torch.device(f'cuda:{gpu_id}')
    SLM_data_gpu = SLM_data.to(device)
    local_reconst_3d = torch.zeros((Nx, Ny, num_images // num_gpus), dtype=torch.complex64, device=device)
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(cal_save_image_torch, i, SLM_data_gpu, Nx, Ny, dx, dy, wav_len, initial_place, pixels, dz, device) for i in range(gpu_id, num_images, num_gpus)]
        for i, future in enumerate(concurrent.futures.as_completed(futures)):
            local_reconst_3d[:, :, i] = future.result()
    
    return local_reconst_3d.cpu()

if __name__ == "__main__":
    try:
        set_start_method('spawn')
    except RuntimeError:
        pass

    # 必要なパラメータの設定
    i = 1j
    wav_len = 532.0 * 10**-9
    Nx, Ny = 32, 32
    dx = 3.45 * 10**-6
    dy = dx
    dz = dx
    wav_num = 2 * np.pi / wav_len
    times = -4 
    initial_place = (10**times)*1000
    pixels = 2
    box_number = 4
    # 画像の枚数
    num_images = 32

    # フィギュアの作成
    fig = plt.figure()

    # 画像読み込み時間の計測開始
    start_time = time.time()

    # フォルダの作成
    folder_name = f'C:\\Users\\Owner\\mizusaki\\3d-holography\\app\\python\\3d-imaging\\output\\RawOutputData_{num_images}_{Nx}x{Ny}_10{times}_pixels={pixels}_d={dz *10**times}_initialPlace{initial_place}'
    os.makedirs(folder_name, exist_ok=True)

    # 画像の読み込みとリサイズ、PyTorchテンソルへの変換
    images = [cv2.resize(cv2.imread(f'C:\\Users\\Owner\\mizusaki\\3d-holography\\app\\python\\3d-imaging\\src\\Original2Dimages_{box_number}_{pixels}px_{Nx}x{Ny}x{num_images}\\image_{(i):05d}.png', cv2.IMREAD_GRAYSCALE).astype(float), (Nx, Ny)) for i in range(1, num_images + 1)]
    images = [torch.tensor(img, dtype=torch.float32) for img in images]

    # GPUごとに画像を分割して並列処理
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_gpus) as executor:
        futures = [executor.submit(process_images, gpu_id, images, Nx, Ny, dx, dy, dz, wav_len, pixels, initial_place, num_images) for gpu_id in range(num_gpus)]
        results = [future.result() for future in concurrent.futures.as_completed(futures)]

    # 出力画像を合計する
    output_images = [torch.zeros((Nx, Ny), dtype=torch.complex64) for _ in range(num_images)]
    for gpu_id in range(num_gpus):
        for n in range(num_images // num_gpus):
            output_images[gpu_id + n * num_gpus] = results[gpu_id][n]

    total_output_images = torch.sum(torch.stack(output_images), dim=0)
    print(total_output_images)

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
        all_results = pool.starmap(reconstruct_on_gpu, [(gpu_id, SLM_data, Nx, Ny, num_images, dx, dy, dz, wav_len, initial_place, pixels) for gpu_id in range(num_gpus)])

    # GPUからCPUに結果を戻し、3D配列に統合
    reconst_3d = torch.cat(all_results, dim=2)
    file_path = os.path.join(folder_name, 'random_data_torch.npy')
    np.save(file_path, reconst_3d.cpu().numpy())
    print('3D array saved to random_data_torch.npy')
