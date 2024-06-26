import numpy as np
import torch

class Initialization:
    def __init__(self):
        # 必要なパラメータの設定
        self.i = 1j
        self.wav_len = 532.0 * 10**-9
        self.Nx, self.Ny = 32, 32
        self.dx = 3.45 * 10**-6
        self.dy = 3.45 * 10**-6
        self.dz = 3.45
        self.wav_num = 2 * np.pi / self.wav_len
        self.times = -4 
        self.initial_place = (10**self.times)*1000