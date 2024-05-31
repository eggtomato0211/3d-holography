import numpy as np
import torch

class Initialization:
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
    
    # 使用可能なGPUの数を確認
    num_gpus = torch.cuda.device_count()
    print(f"使用可能なGPUの数: {num_gpus}")

    # GPUが少なくとも1つ利用可能か確認
    if num_gpus < 1:
        raise ValueError("GPUが検出されませんでした。少なくとも1つのGPUが必要です。")
