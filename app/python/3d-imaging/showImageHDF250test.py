from HDF import HDF
import os
from concurrent.futures import ThreadPoolExecutor
import h5py
import numpy as np
import tifffile
import imageio

# 定数設定
Nx = 128
Ny = 128
depthlevel = 128
dz = 6.9e-06
output_dir = None
folder_name = 'test250'

# hdf_dirs のリスト
hdf_dirs = [
    # r'patch=64_stride=8_fm=16_valpatch=64',
    # r'patch=64_stride=8_fm=64_valpatch=128',
    # r'patch=64_stride=16_fm=16_valpatch=64',
    # r'patch=64_stride=16_fm=16_valpatch=128',
    # r'patch=64_stride=32_fm=16_valpatch=64',
    r'patch=96_stride=4_fm=16_valpatch=128',
    r'patch=96_stride=24_fm=64_valpatch=128',
    # r'patch=128_stride=1_fm=64_valpatch=128',
    # r'patch=128_stride=2_fm=64_valpatch=128',
    # r'patch=128_stride=4_fm=64_valpatch=128',
    # r'patch=128_stride=8_fm=16_valpatch=128',
    # r'patch=128_stride=8_fm=64_valpatch=128',
    # r'patch=128_stride=16_fm=16_valpatch=128',
    # r'patch=128_stride=16_fm=64_valpatch=128',
    # r'patch=128_stride=32_fm=16_valpatch=128',
    # r'patch=128_stride=32_fm=64_valpatch=128',
    # r'patch=128_stride=64_fm=64_valpatch=128',
    # r'patch=128_stride=128_fm=64_valpatch=128',
]

# HDF5フォルダのパスがあるかを確認
for hdf_dir in hdf_dirs:
    check_path = fr'C:\Users\Owner\mizusaki\3d-holography\app\python\3d-imaging\hdf\predictions\{hdf_dir}'
    if not os.path.exists(check_path):
        raise Exception(f'Path does not exist: {check_path}')

# 各 hdf_dir ごとに処理する関数
def process_hdf_dir(hdf_dir):
    # ※ 各スレッドで独自の HDF インスタンスを作成
    hdf = HDF(Nx, Ny, depthlevel, dz, output_dir)
    
    raw_psnrs = []
    prediction_psnrs = []

    # from_num を 1～250 まで処理
    for from_num in range(1, 251):
        # HDF5 ファイルのパスを指定
        hdf5_file = fr'C:\Users\Owner\mizusaki\3d-holography\app\python\3d-imaging\hdf\{folder_name}\Number{from_num}.h5'
        predictions_hdf5_file = fr'C:\Users\Owner\mizusaki\3d-holography\app\python\3d-imaging\hdf\predictions\{hdf_dir}\Number{from_num}_predictions.h5'

        image_mode = False
        movide_mode = False
        value_mode = True
        tiff_mode = True

         # HDF5 ファイルを開いて、目的のデータセットを取得
        # ※ ここでは例として、データセット名を 'data' としています。ご自身のファイルに合わせて変更してください。
        with h5py.File(hdf5_file, 'r') as f:
            # データセット名を確認するには、f.keys() などを利用できます
            # print(list(f.keys()))
            label_data = f['label'][:]
            raw_data = f['raw'][:]
        
        with h5py.File(predictions_hdf5_file, 'r') as f:
            predictions_data = f['predictions'][:]
            predictions_data = predictions_data[0, :, :, :]

        # 読み込んだデータの形状を確認
        print(f"label_data.shape: {label_data.shape}")
        print(f"raw_data.shape: {raw_data.shape}")

        # label_data と raw_data の形状が異なる場合は、エラーを出力
        if label_data.shape != raw_data.shape:
            raise Exception("label_data と raw_data の形状が異なります")

        # label_dataとraw_data、predictionを-1～1に正規化
        label_data = (label_data - label_data.min()) / (label_data.max() - label_data.min()) * 2 - 1
        raw_data = (raw_data - raw_data.min()) / (raw_data.max() - raw_data.min()) * 2 - 1

        if tiff_mode:

            # 必要に応じてデータの前処理を実施（例：スケーリングや型変換など）
            # ここではそのまま出力する例です
            save_dir = fr"C:\Users\Owner\mizusaki\3d-holography\app\python\3d-imaging\tiff\{folder_name}\{hdf_dir}\{from_num}"

            # TIFF ファイルとして保存 save_dir にlabel, raw, predictionsを保存
            os.makedirs(save_dir, exist_ok=True)
                
            # TIFF ファイルの保存
            tifffile.imwrite(os.path.join(save_dir, "label_data.tiff"), label_data.astype(np.float32))
            tifffile.imwrite(os.path.join(save_dir, "raw_data.tiff"), raw_data.astype(np.float32))
            tifffile.imwrite(os.path.join(save_dir, "predictions_data.tiff"), predictions_data.astype(np.float32))

            print(f"TIFF files saved in {save_dir}")

        # モードに応じた処理
        if image_mode:
            save_dir = fr"C:\Users\Owner\mizusaki\3d-holography\app\python\3d-imaging\images\{folder_name}\{hdf_dir}\{from_num}"
            # ディレクトリがなければ作成（必要に応じて）
            os.makedirs(save_dir, exist_ok=True)
            predictions_data = (predictions_data - predictions_data.min()) / (predictions_data.max() - predictions_data.min()) * 2 - 1
            
            def normalize_to_uint8(data):
                data = (data + 1) / 2 * 255
                return np.clip(data, 0, 255).astype(np.uint8)

            label_img = normalize_to_uint8(label_data)
            raw_img = normalize_to_uint8(raw_data)
            predictions_img = normalize_to_uint8(predictions_data)

            # ここでは、3次元データ（例：(frames, height, width)）の各スライスを個別の PNG として保存
            num_slices = label_img.shape[0]
            for i in range(num_slices):
                #label, raw, predictionsを別々のフォルダ(label, raw, prediction)に保存
                imageio.imwrite(os.path.join(save_dir, f"label\{i:03d}.png"), label_img[i])
                imageio.imwrite(os.path.join(save_dir, f"raw\{i:03d}.png"), raw_img[i])
                imageio.imwrite(os.path.join(save_dir, f"predictions\{i:03d}.png"), predictions_img[i])
            print(f"Image files saved in {save_dir}")

        if movide_mode:
            save_dir = fr"C:\Users\Owner\mizusaki\3d-holography\app\python\3d-imaging\movies\{folder_name}\{hdf_dir}\{from_num}"
            os.makedirs(save_dir, exist_ok=True)
            predictions_data = (predictions_data - predictions_data.min()) / (predictions_data.max() - predictions_data.min()) * 2 - 1

            # -1～1 の値を [0,255] の uint8 に変換
            def normalize_to_uint8(data):
                data = (data + 1) / 2 * 255
                return np.clip(data, 0, 255).astype(np.uint8)

            label_vid = normalize_to_uint8(label_data)
            raw_vid = normalize_to_uint8(raw_data)
            predictions_vid = normalize_to_uint8(predictions_data)

            # 動画のフレームレートを指定（例：10fps）
            fps = 1

            # 動画ファイルとして保存（ここでは imageio.mimwrite を利用）
            imageio.mimwrite(os.path.join(save_dir, "label_data.mp4"), label_vid, fps=fps, macro_block_size=None)
            imageio.mimwrite(os.path.join(save_dir, "raw_data.mp4"), raw_vid, fps=fps, macro_block_size=None)
            imageio.mimwrite(os.path.join(save_dir, "predictions_data.mp4"), predictions_vid, fps=fps, macro_block_size=None)
            print(f"Movie files saved in {save_dir}")

        if value_mode:
            save_dir = fr"C:\Users\Owner\mizusaki\3d-holography\app\python\3d-imaging\values\{folder_name}\{hdf_dir}\{from_num}"
            os.makedirs(save_dir, exist_ok=True)
            raw_psnr, prediction_psnr = hdf.savePSNR(hdf5_file, predictions_hdf5_file, save_dir)
            print(f'hdf_dir: {hdf_dir}  from_num: {from_num}  raw_psnr: {raw_psnr}  prediction_psnr: {prediction_psnr}')
            raw_psnrs.append(raw_psnr)
            prediction_psnrs.append(prediction_psnr)

    # 平均値の計算
    raw_psnr_average = sum(raw_psnrs) / len(raw_psnrs)
    prediction_psnr_average = sum(prediction_psnrs) / len(prediction_psnrs)

    # prediction が最も良い（大きい）／悪い（小さい）の番号（インデックス）を取得
    best_prediction_index = prediction_psnrs.index(max(prediction_psnrs))
    worst_prediction_index = prediction_psnrs.index(min(prediction_psnrs))

    # raw と prediction の PSNR 差分の最大・最小のインデックスを取得
    psnr_differences = [pred - r for pred, r in zip(prediction_psnrs, raw_psnrs)]
    best_difference_index = psnr_differences.index(max(psnr_differences))
    worst_difference_index = psnr_differences.index(min(psnr_differences))

    # 結果をファイルに保存
    save_path = fr'C:\Users\Owner\mizusaki\3d-holography\app\python\3d-imaging\values\{folder_name}\{hdf_dir}\psnr.txt'
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'w') as f:
        f.write(f'raw_psnrs: {raw_psnrs}\n')
        f.write(f'prediction_psnrs: {prediction_psnrs}\n')
        f.write(f'raw_psnr_average: {raw_psnr_average}\n')
        f.write(f'prediction_psnr_average: {prediction_psnr_average}\n')
        f.write(f'best_prediction_index: {best_prediction_index}\n')
        f.write(f'worst_prediction_index: {worst_prediction_index}\n')
        f.write(f'best_difference_index: {best_difference_index}\n')
        f.write(f'worst_difference_index: {worst_difference_index}\n')

    print(f"Completed processing for hdf_dir: {hdf_dir}")

# hdf_dirs ごとに並列実行
with ThreadPoolExecutor() as executor:
    executor.map(process_hdf_dir, hdf_dirs)
