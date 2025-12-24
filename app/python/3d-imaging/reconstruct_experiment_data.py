import numpy as np
import os
import cv2
from tie_solver import solve_tie
from propagation import Propagation
from hdf_manager import HDFManager

def main():
    # --- 1. パラメータ設定 (修士論文の実験条件に準拠) ---
    wav_len = 532.0 * 10**-9  # 波長 [m]
    dx = 3.45 * 10**-6        # 画素ピッチ [m]
    dy = dx
    dz_tie = 4.0 * 10**-6     # TIE用の撮影間隔 Δz [m] [cite: 207]
    dz_recon = 1.0 * 10**-6   # 再構成のピッチ [m] (マウス脳の場合 1um) [cite: 226]
    
    Nx, Ny = 128, 128         # 処理サイズ (GPUメモリを考慮) [cite: 207]
    depth_layers = 51         # 再構成する層の数 [cite: 226]
    
    # --- 2. 実験データの読み込み ---
    # ここでは例として、特定のフォルダから3枚の画像を読み込む想定
    data_dir = r"C:\Users\Owner\mizusaki\experiment_data" 
    
    # 画像の読み込みと正規化
    def load_and_norm(name):
        img = cv2.imread(os.path.join(data_dir, name), cv2.IMREAD_GRAYSCALE)
        if img is None:
            return np.zeros((Nx, Ny)) # ダミーデータ
        img_resized = cv2.resize(img, (Nx, Ny), interpolation=cv2.INTER_LINEAR)
        return img_resized.astype(float) / 255.0

    I_minus = load_and_norm("z_minus_4um.tif")
    I_focus = load_and_norm("z_0um.tif")
    I_plus  = load_and_norm("z_plus_4um.tif")

    # --- 3. TIEによる位相復元 ---
    print("Solving TIE for phase retrieval...")
    phi = solve_tie(I_focus, I_minus, I_plus, dz_tie, wav_len, dx, dy)
    
    # --- 4. 複素振幅の作成 ---
    # 振幅は強度の平方根、これに復元した位相を結合
    u_focus = np.sqrt(I_focus) * np.exp(1j * phi) [cite: 102]
    
    # --- 5. 逆伝搬による3次元再構成 ---
    print(f"Reconstructing 3D volume ({depth_layers} layers)...")
    propagation = Propagation(wav_len, dx, dy)
    raw_data = np.zeros((depth_layers, Ny, Nx), dtype=float)
    
    # 中心(z=0)を基準に前後をスキャン
    start_z = -(depth_layers // 2) * dz_recon
    
    for i in range(depth_layers):
        dist = start_z + (i * dz_recon)
        # nearprop_conv を使って任意の距離の像を得る [cite: 119]
        recon_field = propagation.nearprop_conv(u_focus, Nx, Ny, dist)
        raw_data[i] = np.abs(recon_field)**2 # 強度画像として保存

    # --- 6. 結果の保存 ---
    output_folder = r"D:\nosaka\data\reconstructed_experiment"
    os.makedirs(output_folder, exist_ok=True)
    
    # ラベルデータはないので、評価用に空のデータを渡すか、rawをそのまま渡す
    hdf_manager = HDFManager(Nx, Ny, depth_layers, dz_recon, output_folder)
    hdf_manager.save_to_hdf(raw_data, raw_data, "reconstructed_volume.h5")
    
    print(f"Success! Reconstruction saved to {output_folder}")

if __name__ == "__main__":
    main()