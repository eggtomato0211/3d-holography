import os
import h5py
import numpy as np
import tifffile
from pathlib import Path
from tqdm import tqdm

# ==========================================
# ⚙️ 設定エリア
# ==========================================
INPUT_DIR = Path(r"D:\nosaka\outputs\mouse_prediction_by_harder_noisy_model")
OUTPUT_DIR = Path(r"D:/nosaka/outputs/mouse_prediction_by_harder_noisy_model_tiff")
DATA_KEY = "predictions" 

# ★追加設定: オートスケールを行うかどうか
# True:  見やすいように 0.0〜1.0 に引き伸ばす (Fijiで即見れる)
# False: 生の値をそのまま保存する (定量解析用)
DO_NORMALIZE = True
# ==========================================

def h5_to_tiff_simple():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    h5_files = sorted(list(INPUT_DIR.glob("*.h5")))
    
    if not h5_files:
        print(f"⚠️ 指定したフォルダに .h5 ファイルが見つかりませんでした: {INPUT_DIR}")
        return

    print(f"🚀 変換開始: {len(h5_files)} 個のファイルを処理します")
    if DO_NORMALIZE:
        print("✨ オートスケール (0.0 - 1.0) : ON")
    else:
        print("💾 生データ保存モード : ON")

    for h5_path in tqdm(h5_files):
        try:
            with h5py.File(h5_path, "r") as f:
                if DATA_KEY not in f:
                    print(f"⚠️ Skip: {h5_path.name}")
                    continue
                
                # データを読み込む
                data = f[DATA_KEY][:]
                data = data.astype(np.float32)

                # ==========================================
                # ★ ここにオートスケール処理を追加
                # ==========================================
                if DO_NORMALIZE:
                    # 異常値(ホットピクセル等)を除外するため、最大・最小ではなく
                    # 0.1%タイルと99.9%タイルを使用（ロバストな正規化）
                    vmin = np.percentile(data, 0.1)
                    vmax = np.percentile(data, 99.9)

                    # ゼロ除算防止
                    if vmax > vmin:
                        # 0.0 〜 1.0 に引き伸ばす
                        data = (data - vmin) / (vmax - vmin)
                        data = np.clip(data, 0.0, 1.0)
                    else:
                        # 完全に真っ黒な画像などの場合
                        data[:] = 0.0

            # 保存ファイル名
            output_path = OUTPUT_DIR / (h5_path.stem + ".tif")
            
            # TIFFとして保存
            tifffile.imwrite(output_path, data)

            # (オプション) 確認用にログを出すならコメントアウトを外す
            # if DO_NORMALIZE:
            #     tqdm.write(f"   [{h5_path.stem}] Scaled: {vmin:.2e} -> {vmax:.2e}")

        except Exception as e:
            print(f"❌ Error converting {h5_path.name}: {e}")

    print("\n✅ すべての変換が完了しました！")

if __name__ == "__main__":
    h5_to_tiff_simple()