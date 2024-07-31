import os
import numpy as np
import h5py
import matplotlib.pyplot as plt
import cv2

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
    
    #　HDFファイルから動画を生成して動画ファイルとして保存する関数
    def make_movie(self, type, hdf5_file, save_dir,background_mode=False):
        #save_dirがあるとき、self.output_dirに代入
        if save_dir is not None:
            self.output_dir = save_dir
        else:
            self.output_dir =  '.\\app\\python\\3d-imaging\\movies'

        # 出力ディレクトリが存在しない場合は作成
        os.makedirs(self.output_dir, exist_ok=True)

        with h5py.File(hdf5_file, 'r') as f:
            data = f[type][:]
            print(f"{type}データの形状: {data.shape}")
            # フレーム数
            num_frames = data.shape[1]
            # フレームのサイズ (width, height)
            image_size = (data.shape[2], data.shape[3])  # (128, 128)
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
            
            max_value = 0
            min_value = 255
            background_value = 0
            
            for k in range(num_frames):
                if k == 0:
                    if background_mode == True:
                        #1フレームあたりのすべてのピクセルの平均値を取得
                        background_value = np.mean(np.abs(data[0, k, :, :]))
                        print(f"background_value: {background_value}")

                # すべてのフレームの中で一番大きい値、小さい値を取得
                tmp_max_value = np.max(np.abs(data[0, k, :, :]))
                tmp_min_value = np.min(np.abs(data[0, k, :, :]))
                # 前のmax_value, min_valueと比較して大きい値、小さい値を取得
                if tmp_max_value > max_value:
                    max_value = tmp_max_value
                if tmp_min_value < min_value:
                    min_value = tmp_min_value

                # 計算を行い、結果が0未満の場合は0にする
                data[0, k, :, :] = np.maximum(np.abs(data[0, k, :, :]) - background_value, 0)
            
            max_value = max_value - background_value
            # min_value = min_value - background_value ただし、0未満の場合は0にする
            if min_value - background_value < 0:
                min_value = 0
            else:
                min_value = min_value - background_value
            
            print(f"max_value: {max_value}")
            print(f"min_value: {min_value}")

            for i in range(num_frames):
                # フレームを正しい形式に変換
                frame = np.abs(data[0, i, :, :])
                # フレームの正規化
                # フレームを0〜255の範囲にスケーリング
                frame = ((frame - min_value) / (max_value - min_value) * 255).astype('uint8')
                video.write(frame)
            
            video.release()
            print(f"動画ファイル '{movie_file}' が作成されました")
        return
    
    def make_movie2(self, type, hdf5_file, save_dir):
        #save_dirがあるとき、self.output_dirに代入
        if save_dir is not None:
            self.output_dir = save_dir
        else:
            self.output_dir =  '.\\app\\python\\3d-imaging\\movies'

        # 出力ディレクトリが存在しない場合は作成
        os.makedirs(self.output_dir, exist_ok=True)

        with h5py.File(hdf5_file, 'r') as f:
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
            
            max_value = 0
            min_value = 255
            
            for k in range(num_frames):

                # すべてのフレームの中で一番大きい値、小さい値を取得
                tmp_max_value = np.max(np.abs(data[k, :, :]))
                tmp_min_value = np.min(np.abs(data[k, :, :]))
                # 前のmax_value, min_valueと比較して大きい値、小さい値を取得
                if tmp_max_value > max_value:
                    max_value = tmp_max_value
                if tmp_min_value < min_value:
                    min_value = tmp_min_value

            print(f"max_value: {max_value}")
            print(f"min_value: {min_value}")

            for i in range(num_frames):
                # フレームを正しい形式に変換
                frame = np.abs(data[i, :, :])
                # フレームの正規化
                # フレームを0〜255の範囲にスケーリング
                frame = ((frame - min_value) / (max_value - min_value) * 255).astype('uint8')
                video.write(frame)
            
            video.release()
            print(f"動画ファイル '{movie_file}' が作成されました")

        return
    
    def make_movie3(self, type, hdf5_file, save_dir):
        #save_dirがあるとき、self.output_dirに代入
        if save_dir is not None:
            self.output_dir = save_dir
        else:
            self.output_dir =  '.\\app\\python\\3d-imaging\\movies'

        # 出力ディレクトリが存在しない場合は作成
        os.makedirs(self.output_dir, exist_ok=True)

        with h5py.File(hdf5_file, 'r') as f:
            data = f[type][:]
            print(f"{type}データの形状: {data.shape}")
            # フレーム数
            num_frames = data.shape[1]
            # フレームのサイズ (width, height)
            image_size = (data.shape[2], data.shape[3])  # (128, 128)
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

            for i in range(num_frames):
                # フレームを正しい形式に変換
                frame = np.abs(data[0, i, :, :])
                frame = ((frame - np.min(frame)) / (np.max(frame) - np.min(frame)) * 255).astype('uint8')
                video.write(frame)
            
            video.release()
            print(f"動画ファイル '{movie_file}' が作成されました")
        return