import numpy as np

def compare_npy_files(file1, file2):
    # ファイルを読み込む
    data1 = np.load(file1)
    data2 = np.load(file2)
    
    # 形状が一致しているかチェック
    if data1.shape != data2.shape:
        print(f"形状が異なります: {file1} の形状 {data1.shape}, {file2} の形状 {data2.shape}")
        return
    
    # 要素ごとに比較
    diff_indices = np.where(data1 != data2)
    
    if len(diff_indices[0]) == 0:
        print("2つのNPYファイルの中身は一致しています。")
    else:
        print(f"{len(diff_indices[0])}個の要素が一致していません。")
        
        # 1つだけ表示
        diff_idx = tuple(idx[0] for idx in diff_indices)
        print(f"最初の不一致箇所: {file1}[{diff_idx}] = {data1[diff_idx]}, {file2}[{diff_idx}] = {data2[diff_idx]}")
        print(f"{data1[(1,1,1)]}")

# 使用例
compare_npy_files("random_data_torch.npy", "random_data.npy")