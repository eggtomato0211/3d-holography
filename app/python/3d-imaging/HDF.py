import os
import numpy as np
import h5py
import matplotlib.pyplot as plt
import cv2
import os
from skimage.metrics import peak_signal_noise_ratio

class HDF:
    def __init__(self, Nx, Ny, depthlevel, dz, output_dir=None):
        if output_dir is None:
            self.output_dir = rf'.\app\python\3d-imaging\hdf\{Nx}x{Ny}x{depthlevel}_d={dz}'
        else:
            self.output_dir = output_dir

        # ディレクトリが存在しない場合は作成する
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        
    def makeHDF(self, raw_data, label_data, output_fileName):
        # 出力ファイル名
        self.output_file = os.path.join(self.output_dir, output_fileName)
        # HDF5ファイルの作成、ただしすでに存在する場合はエラーを吐き出す
        # if os.path.exists(self.output_file):
        #     raise FileExistsError(f"{self.output_file} はすでに存在します")

        # それぞれのサイズを出力
        print("Size of raw_data:", raw_data.shape)
        print("Size of label_data:", label_data.shape)
        
        with h5py.File(self.output_file, 'w') as f:
            # データセットの作成
            f.create_dataset('raw', data=raw_data, compression='gzip')
            f.create_dataset('label', data=label_data, compression='gzip')

        print(f"HDF5ファイル '{self.output_file}' が 'raw' と 'label' のデータセットで作成されました")
        
        return
    
    def display_image(self, type, hdf5_file, depth):
        with h5py.File(hdf5_file, 'r') as f:
            data = f[type][:]
            print(f"{type}データの形状: {data.shape}")

            # 指定したdepthのスライスを表示
            plt.imshow(data[depth, :, :], cmap='gray')
            plt.title(f'Depth: {depth}')
            plt.show()
        return
    
    def save_images(self, type, hdf5_file, save_dir):
        #save_dirがあるとき、self.output_dirに代入
        if save_dir is not None:
            self.output_dir = save_dir
        else:
            self.output_dir =  '.\\app\\python\\3d-imaging\\images'
        
        self.output_dir = os.path.join(self.output_dir, type)
        # 出力ディレクトリが存在しない場合は作成
        os.makedirs(self.output_dir, exist_ok=True)

        with h5py.File(hdf5_file, 'r') as f:
            if type == 'predictions':
                data = f[type][:]
                data = data[0, :, :, :]
                print(f"{type}データの形状: {data.shape}")
                
            else:
                data = f[type][:]
                print(f"{type}データの形状: {data.shape}")
            
            num_frames = data.shape[0]
            max_value = -1
            min_value = 1

            for k in range(num_frames):
                # すべてのフレームの中で一番大きい値、小さい値を取得
                tmp_max_value = np.max(data[k, :, :])
                tmp_min_value = np.min(data[k, :, :])
                # 前のmax_value, min_valueと比較して大きい値、小さい値を取得
                if tmp_max_value > max_value:
                    max_value = tmp_max_value
                if tmp_min_value < min_value:
                    min_value = tmp_min_value
            
            print(f"max_value: {max_value}")
            print(f"min_value: {min_value}")
            
            for i in range(num_frames):
                # フレームを正しい形式に変換
                frame = data[i, :, :]
                # フレームの正規化
                # フレームを0〜255の範囲にスケーリング
                frame = ((frame - min_value) / (max_value - min_value) * 255).astype('uint8')
                # 画像を保存
                image_file = os.path.join(self.output_dir, f'{type}_{i:05d}.png')
                plt.imsave(image_file, frame, cmap='gray', vmin=0, vmax=255)
        return

    # HDF5ファイルを読み込んで画像を表示する関数
    def display_predictions_image(self, type ,hdf5_file, depth):
        with h5py.File(hdf5_file, 'r') as f:
            data = f[type][:]
            print(f"predictionsデータの形状: {data.shape}")

            # 指定したdepthのスライスを表示
            plt.imshow(np.abs(data[0, depth, :, :]), cmap='gray', vmin=-1, vmax=1)
            plt.title(f'Depth: {depth}')
            plt.show()
        return
    
    #　HDFファイルから動画を生成して動画ファイルとして保存する関数
    def make_movie(self, type, hdf5_file, save_dir):
        #save_dirがあるとき、self.output_dirに代入
        if save_dir is not None:
            self.output_dir = save_dir
        else:
            self.output_dir =  '.\\app\\python\\3d-imaging\\movies'

        # 出力ディレクトリが存在しない場合は作成
        os.makedirs(self.output_dir, exist_ok=True)

        with h5py.File(hdf5_file, 'r') as f:
            if type == 'predictions':
                prediction_data = f[type][:]
                # 次元を削減 (最初の次元を削除)
                data = np.squeeze(prediction_data, axis=0)
            else:
                data = f[type][:]
            
            print(f"{type}データの形状: {data.shape}")

            # フレーム数
            num_frames = data.shape[0]
            # フレームのサイズ (width, height)
            image_size = (data.shape[1], data.shape[2])

            # 動画のフレームレート
            fps = 1
            # 動画の保存先
            movie_file = os.path.join(self.output_dir, f'{type}_movie.mp4')
            print(f"動画ファイル '{movie_file}' を作成します")

             # 動画の作成
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video = cv2.VideoWriter(movie_file, fourcc, fps, image_size, isColor=False)
            if not video.isOpened():
                print(f"動画ファイル '{movie_file}' の作成に失敗しました")
                return
            
            max_value = -1
            min_value = 1

            for k in range(num_frames):
                # すべてのフレームの中で一番大きい値、小さい値を取得
                tmp_max_value = np.max(data[k, :, :])
                tmp_min_value = np.min(data[k, :, :])
                # 前のmax_value, min_valueと比較して大きい値、小さい値を取得
                if tmp_max_value > max_value:
                    max_value = tmp_max_value
                if tmp_min_value < min_value:
                    min_value = tmp_min_value
        
            
            print(f"max_value: {max_value}")
            print(f"min_value: {min_value}")

            for i in range(num_frames):
                # フレームを正しい形式に変換
                frame = data[i, :, :]
                # フレームの正規化
                # フレームを0〜255の範囲にスケーリング
                frame = ((frame - min_value) / (max_value - min_value)).astype('float32')
                video.write(frame)
            
            video.release()
            print(f"動画ファイル '{movie_file}' が作成されました")
        return
    
    def show_histogram(self, type, hdf5_file, depth, z_range=0):
        with h5py.File(hdf5_file, 'r') as f:
            data = f[type][:]
            print(f"{type}データの形状: {data.shape}")
             # "predictions" データセットから指定されたzスライスのデータを取得
            
            # 指定された範囲のzスライスを取得
            z_slices = range(max(0, depth - z_range), min(data.shape[1], depth + z_range + 1))

            # ヒストグラムをプロット
            plt.figure(figsize=(10, 8))
            for z in z_slices:
                slice_data = data[0, z, :, :]
                # 各スライスの最大値と最小値を計算
                max_value = np.max(slice_data)
                min_value = np.min(slice_data)
                
                # 最大値と最小値を表示
                print(f'z = {z}: Max = {max_value}, Min = {min_value}')
                plt.hist(slice_data.flatten(), bins=100, alpha=0.5, label=f'z = {z}')
        
        # プロットの装飾
        plt.title(f'Histogram of z-slices from {depth - z_range} to {depth + z_range}')
        plt.xlabel('Pixel Value')
        plt.ylabel('Frequency')
        plt.legend()
        plt.show()
    
    def translateTif2HDF(self, tifs_folder, output_fileName, width, height):
        # 出力ファイル名
        self.output_file = os.path.join(self.output_dir, output_fileName)

        # 画像サイズ
        resize_shape = (width, height)

        # HDF5ファイルの作成
        with h5py.File(self.output_file, "w") as h5_file:
            # フォルダ内のファイルを読み込み、連番順にソート
            image_files = sorted(
                [f for f in os.listdir(tifs_folder) if f.endswith(".tif")],
                key=lambda x: int(x.split("_")[1].split(".")[0]) if "_" in x else 0,
            )

            # 画像の3次元配列を作成
            raw_data = np.zeros((len(image_files), *resize_shape), dtype=float)

            # Resize images
            for i, image_file in enumerate(image_files):
                image = cv2.imread(os.path.join(tifs_folder, image_file), cv2.IMREAD_GRAYSCALE)
                image = cv2.resize(image, resize_shape, interpolation=cv2.INTER_LINEAR)
                raw_data[i] = image

            h5_file.create_dataset('raw', data=raw_data, compression='gzip')


        print(f"HDF5ファイルに保存しました: {self.output_file}")
    
    def compareHDF(self, originalhdf, prediction_hdf, threshold):
        with h5py.File(originalhdf, 'r') as f:
            label_data = f['label'][:]
        
        with h5py.File(prediction_hdf, 'r') as f:
            prediction_data = np.squeeze(f['prediction'][:], axis=0)
        
        # 各 (x, y) において、閾値を超えた z のインデックスを収集
        label_thresholded = np.apply_along_axis(
            lambda z_values: np.where(z_values > threshold)[0] + 1, 
            axis=0, 
            arr=label_data
        )  # shape (y, x), 各要素はリスト
        prediction_thresholded = np.apply_along_axis(
            lambda z_values: np.where(z_values > threshold)[0] + 1, 
            axis=0, 
            arr=prediction_data
        )  # shape (y, x), 各要素はリスト

         # 比較のために、リストの差を計算
        difference = np.empty(label_thresholded.shape, dtype=object)
        for y in range(label_thresholded.shape[0]):
            for x in range(label_thresholded.shape[1]):
                # リストの差異を計算
                difference[y, x] = list(set(label_thresholded[y, x]) ^ set(prediction_thresholded[y, x]))

        # 結果を返す
        return label_thresholded, prediction_thresholded, difference
    
    def savePSNR(self, original_hdf, prediction_hdf, save_dir):
        with h5py.File(original_hdf, 'r') as f:
            label_data = f['label'][:]
            raw_data = f['raw'][:]

        with h5py.File(prediction_hdf, 'r') as f:
            prediction_data = np.squeeze(f['predictions'][:], axis=0)

        # label_dataとprediction_dataの形状が異なる場合はエラーを出力
        if label_data.shape != prediction_data.shape:
            raise ValueError(f'Label data shape {label_data.shape} does not match predictions data shape {prediction_data.shape}')

        print(f"Label data range: min = {np.min(label_data)}, max = {np.max(label_data)}")
        print(f"Raw data range: min = {np.min(raw_data)}, max = {np.max(raw_data)}")
        print(f"Prediction data range: min = {np.min(prediction_data)}, max = {np.max(prediction_data)}")

        #正規化を行う
        # label_data, raw_data, prediction_dataが -1 から 1 の範囲になるようにスケーリング
        label_data = ((label_data - np.min(label_data)) / (np.max(label_data) - np.min(label_data)) * 2 - 1).astype('float32')
        raw_data = ((raw_data - np.min(raw_data)) / (np.max(raw_data) - np.min(raw_data)) * 2 - 1).astype('float32')

        # `skimage.metrics.peak_signal_noise_ratio()` に合わせるため、data_range を取得
        data_range = np.max(label_data) - np.min(label_data)
        
        # PSNR を skimage の関数で計算
        raw_psnr = peak_signal_noise_ratio(label_data, raw_data, data_range=data_range)
        prediction_psnr = peak_signal_noise_ratio(label_data, prediction_data, data_range=data_range)

        # 出力ディレクトリが存在しない場合は作成
        os.makedirs(save_dir, exist_ok=True)

        # PSNR を保存
        with open(os.path.join(save_dir, 'psnr.txt'), 'w') as f:
            f.write(f'Raw PSNR: {raw_psnr}\n')
            f.write(f'Prediction PSNR: {prediction_psnr}\n')

        return raw_psnr, prediction_psnr
    
    def zMSE(self, original_hdf, prediction_hdf, z):
        # HDF5ファイルを読み込む
        with h5py.File(original_hdf, 'r') as f:
            label_data = f['label'][:]
            raw_data = f['raw'][:]
        
        with h5py.File(prediction_hdf, 'r') as f:
            prediction_data = np.squeeze(f['predictions'][:], axis=0)
        
         #label_dataとprediction_dataの形状が異なる場合はエラーを出力
        if label_data.shape != prediction_data.shape:
            raise ValueError(f'Label data shape {label_data.shape} does not match predictions data shape {prediction_data.shape}')

        print(f"Label data range: min = {np.min(label_data)}, max = {np.max(label_data)}")
        print(f"Raw data range: min = {np.min(raw_data)}, max = {np.max(raw_data)}")
        print(f"Prediction data range: min = {np.min(prediction_data)}, max = {np.max(prediction_data)}")
        
        # # label_data, raw_dataが 0 から 255 の範囲になるようにスケーリング
        # label_data = ((label_data - np.min(label_data)) / (np.max(label_data) - np.min(label_data)) * 255).astype('uint8')
        # raw_data = ((raw_data - np.min(raw_data)) / (np.max(raw_data) - np.min(raw_data)) * 255).astype('uint8')
        # prediction_data = ((prediction_data - np.min(prediction_data)) / (np.max(prediction_data) - np.min(prediction_data)) * 255).astype('uint8')
        

        # prediction_data[prediction_data <= 0] = 0
        
        #あるzスライスにおけるMSEを計算
        raw_mse = np.mean((label_data[z, :, :] - raw_data[z, :, :]) ** 2)
        prediction_mse = np.mean((label_data[z, :, :] - prediction_data[z, :, :]) ** 2)

        #z=120~127の範囲での画像を表示
        if z <= 20:
            plt.figure(figsize=(15, 5))
            plt.subplot(1, 3, 1)
            plt.imshow(label_data[z, :, :], cmap='gray')
            plt.title('Label')
            plt.subplot(1, 3, 2)
            plt.imshow(raw_data[z, :, :], cmap='gray')
            plt.title('Raw')
            plt.subplot(1, 3, 3)
            plt.imshow(prediction_data[z, :, :], cmap='gray')
            plt.title('Prediction')
            plt.show()

        #MSEを表示
        print(f'z = {z}')
        print(f'Raw MSE: {raw_mse}')
        print(f'Prediction MSE: {prediction_mse}')