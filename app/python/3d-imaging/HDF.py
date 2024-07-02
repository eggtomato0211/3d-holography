import os
import numpy as np
import h5py
import matplotlib.pyplot as plt

class HDF:
    def __init__(self, Nx, Ny, depthlevel, dz, output_dir=None):
        if output_dir is None:
            self.output_dir = rf'.\app\python\3d-imaging\hdf\{Nx}x{Ny}x{depthlevel}_d={dz}'
        else:
            self.output_dir = output_dir

        # ディレクトリが存在しない場合は作成する
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        
    def makeHDF(self, raw_3d, label_3d, output_fileName):
        # 出力ファイル名
        self.output_file = os.path.join(self.output_dir, output_fileName)
        with h5py.File(self.output_file, 'w') as f:
            # データセットの作成
            f.create_dataset('raw', data=raw_3d, compression='gzip')
            f.create_dataset('label', data=label_3d, compression='gzip')

        print(f"HDF5ファイル '{self.output_file}' が 'raw' と 'label' のデータセットで作成されました")
        
        return
    
    def display_image(self, type, hdf5_file, depth):
        with h5py.File(hdf5_file, 'r') as f:
            data = f[type][:]
            print(f"{type}データの形状: {data.shape}")

            # 指定したdepthのスライスを表示
            plt.imshow(np.abs(data[depth, :, :]), cmap='gray')
            plt.title(f'Depth: {depth}')
            plt.show()
        return

    
    # HDF5ファイルを読み込んで画像を表示する関数
    def display_predictions_image(self, type ,hdf5_file, depth):
        with h5py.File(hdf5_file, 'r') as f:
            data = f[type][:]
            print(f"predictionsデータの形状: {data.shape}")

            # 指定したdepthのスライスを表示
            plt.imshow(np.abs(data[0, depth, :, :]), cmap='gray')
            plt.title(f'Depth: {depth + 961}')
            plt.show()
        return