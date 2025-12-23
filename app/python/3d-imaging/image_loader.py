import os
import cv2
import random
import concurrent.futures

class ImageLoader:
    def __init__(self, Nx, Ny, pixels, number_of_plots, channels, depthlevel, folder_path):
        self.Nx = Nx
        self.Ny = Ny
        self.pixels = pixels
        self.number_of_plots = number_of_plots
        self.channels = channels
        self.depthlevel = depthlevel
        self.folder_path = folder_path

    def load_images(self):
        # フォルダ内のすべての画像パスを取得
        image_paths = [os.path.join(self.folder_path, f) for f in os.listdir(self.folder_path) if f.endswith('.tiff')]
        
        # 画像パスの順序をランダムにシャッフル
        random.shuffle(image_paths)
        
        # 指定された数だけの画像を読み込む (各層に平均5.5個配置するため、余裕を持って10倍読み込む)
        # ただし、フォルダ内の画像総数が上限
        total_needed = self.channels * self.depthlevel * 10
        if total_needed > len(image_paths):
            total_needed = len(image_paths)
            
        selected_paths = image_paths[:total_needed]
        
        # 並列で画像を読み込む
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [executor.submit(cv2.imread, path, cv2.IMREAD_GRAYSCALE) for path in selected_paths]
            images = [future.result().astype(float) for future in concurrent.futures.as_completed(futures)]
        
        return images