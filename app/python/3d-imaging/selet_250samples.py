from pathlib import Path
import shutil
import numpy as np

def pick_random_files(root_dir, pattern="*.h5", k=250, seed=None):
    #fileオブジェクトを作成
    root = Path(root_dir)
    #.h5にマッチするファイルをすべて取得
    files = sorted(root.glob(pattern))
    #例外処理
    if len(files) < k:
        raise ValueError(f"files={len(files)} is less than k={k}")
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(files), size=k, replace=False)
    selected_files = [files[i] for i in idx]
    return selected_files

raw_data_files = r"D:/nosaka/data/noisy_datasets_harder/Val"
picked = pick_random_files(raw_data_files, pattern="*.h5", k=250)

dest_dir = Path(r"D:/nosaka/data/noisy_datasets_harder/Val_250")
dest_dir.mkdir(parents=True, exist_ok=True)
for p in picked:
    shutil.copy2(p, dest_dir / p.name)
