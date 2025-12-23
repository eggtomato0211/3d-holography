import numpy as np
from concurrent.futures import ThreadPoolExecutor
from propagation import Propagation
import cv2

class ReconstructionProcessor:
    def __init__(self, Nx, Ny, dx, dy, dz, depthlevel, channels, images, random_mode, phase_gen, propagation):
        self.Nx = Nx
        self.Ny = Ny
        self.dx = dx
        self.dy = dy
        self.dz = dz
        self.depthlevel = depthlevel
        self.channels = channels
        self.images = images
        self.random_mode = random_mode
        self.phase_gen = phase_gen
        self.propagation = propagation

    def process_image(self, z, input_image):
        # input_image: ランダムに選ばれたビーズ画像 (2D array)
        # z: ビーズが配置される深さ
        
        # 位相項の付与 (元のロジックに準拠: input * exp(1j))
        input_field = input_image * np.exp(1j)
        
        output_3d_component = np.zeros((self.depthlevel, self.Nx, self.Ny), dtype=float)
        
        # 深さ z にある物体を、全層 target_z に伝搬
        for target_z in range(self.depthlevel):
            dist = (z - target_z) * self.dz
            # 強度のみ計算
            output_3d_component[target_z] = np.abs(self.propagation.nearprop_conv(input_field, self.Nx, self.Ny, dist))**2
            
        return output_3d_component

    def process_channel(self, channel, image_number, number_of_plots):
         # `random_mode`がTrueならランダム、Falseなら等間隔のdepthsを生成
        if self.random_mode:
            # 重複なしで深さを選ぶ (配置するスライスを決める)
            depths = np.random.choice(self.depthlevel, image_number, replace=False)
            depths.sort()
        else:
            # 等間隔でdepthsを生成
            step = self.depthlevel // image_number
            depths = np.arange(0, self.depthlevel, step)[:image_number]
        
        raw_data = np.zeros((self.depthlevel, self.Nx, self.Ny), dtype=float)
        label_data = np.zeros((self.depthlevel, self.Nx, self.Ny), dtype=float)

        num_total_images = len(self.images)
        
        # 処理するタスクのリストを作成
        # タスク: (配置深さz, 使用する画像image)
        tasks = []

        # 選択された depths (スライス) に対してのみビーズを配置
        # アプローチA: 各層で1画像を配置（画像にはnumber_of_plots個のビーズが含まれる）
        for z in depths:
            # 1画像をランダムに選ぶ（この画像にnumber_of_plots個のビーズが含まれる）
            idx = np.random.choice(num_total_images, 1)[0]
            img = self.images[idx]
            
            # Labelデータ作成
            label_data[z] += img
            
            # 計算タスクに追加
            tasks.append((z, img))

        if self.propagation.use_gpu:
            import cupy as cp
            print("Using GPU for accumulation...")
            raw_data_gpu = cp.zeros((self.depthlevel, self.Nx, self.Ny), dtype=float)
            
            # GPUの場合はスレッドプールを使わず、シーケンシャルに処理してGPUで加算する
            # (スレッドからのGPU呼び出しはオーバーヘッドやコンテキストスイッチの問題があるため)
            for z, img in tasks:
                 # 位相項の付与
                input_field = img * np.exp(1j)
                
                # 各層への伝搬と加算
                for target_z in range(self.depthlevel):
                    dist = (z - target_z) * self.dz
                    # GPU配列として受け取る (return_gpu=True)
                    propagated_field = self.propagation.nearprop_conv(input_field, self.Nx, self.Ny, dist, return_gpu=True)
                    # 強度計算と加算 (GPU上で実行)
                    raw_data_gpu[target_z] += cp.abs(propagated_field)**2
            
            # 最後にCPUへ転送
            raw_data = cp.asnumpy(raw_data_gpu)

        else:
            print("Using CPU with ThreadPoolExecutor...")
            # 並列処理で伝搬計算 (CPU)
            with ThreadPoolExecutor(max_workers=16) as executor:
                # tasksの各要素を引数として process_image を呼ぶ
                output_volumes = list(executor.map(lambda args: self.process_image(*args), tasks))
            
            # 全ての計算結果を加算
            for volume in output_volumes:
                raw_data += volume

        if self.Nx != 128:
            resized_raw_data = np.zeros((self.depthlevel, 128, 128), dtype=float)
            resized_label_data = np.zeros((self.depthlevel, 128, 128), dtype=float)
            
            for i in range(self.depthlevel):
                # bilinear補間を使用してリサイズ
                resized_raw_data[i] = cv2.resize(raw_data[i], (128, 128), interpolation=cv2.INTER_LINEAR)
                resized_label_data[i] = cv2.resize(label_data[i], (128, 128), interpolation=cv2.INTER_LINEAR)
            
            raw_data = resized_raw_data
            label_data = resized_label_data
            
            print(f"データを{self.Nx}*{self.Ny}から128*128 にリサイズしました。")
        
        return raw_data, label_data
