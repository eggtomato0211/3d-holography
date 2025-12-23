import os
import hashlib
import h5py
import numpy as np

def get_file_hash(file_path):
    """ ファイルの SHA256 ハッシュを取得 """
    hash_sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_sha256.update(chunk)
    return hash_sha256.hexdigest()

def compare_hdf5(file1, file2):
    """ HDF5 ファイルの内容を比較 """
    with h5py.File(file1, 'r') as f1, h5py.File(file2, 'r') as f2:
        # データセットのキーを取得
        keys1 = set(f1.keys())
        keys2 = set(f2.keys())

        if keys1 != keys2:
            print(f"❌ データセットのキーが一致しません: {keys1} vs {keys2}")
            return False

        # 各データセットの内容を比較
        for key in keys1:
            data1 = f1[key][:]
            data2 = f2[key][:]

            if not np.array_equal(data1, data2):
                print(f"❌ データセット '{key}' の内容が異なります")
                return False

    print("✅ HDF5 ファイルの内容は完全に一致します")
    return True

def compare_files(file1, file2):
    """ 2つのファイルが完全に一致するかチェック """
    if not os.path.exists(file1) or not os.path.exists(file2):
        print("❌ どちらかのファイルが存在しません")
        return False

    # ファイルサイズの比較
    size1 = os.path.getsize(file1)
    size2 = os.path.getsize(file2)
    if size1 != size2:
        print(f"❌ ファイルサイズが異なります: {size1} vs {size2}")
        return False

    # SHA256 ハッシュの比較
    hash1 = get_file_hash(file1)
    hash2 = get_file_hash(file2)
    if hash1 != hash2:
        print(f"❌ ハッシュ値が異なります: {hash1} vs {hash2}")
        return False

    # HDF5 の内容を比較
    return compare_hdf5(file1, file2)


# 比較するファイルのパス
file1 = r"C:\Users\Owner\mizusaki\3d-holography\app\python\3d-imaging\hdf\32x32x128_0-1\val\10plots_8images_TruerandomMode_NumberFrom1025.h5"
file2 = r"C:\Users\Owner\mizusaki\3d-holography\app\python\3d-imaging\hdf\val250\Number1.h5"

# ファイル比較を実行
compare_files(file1, file2)
