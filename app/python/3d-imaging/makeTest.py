import os
import shutil
import random

# 元のフォルダとコピー先フォルダのパス
source_folder = r"C:\Users\Owner\mizusaki\3d-holography\app\python\3d-imaging\hdf\32x32x128_0-1\val"
destination_folder = r"C:\Users\Owner\mizusaki\3d-holography\app\python\3d-imaging\hdf\val250"
mapping_file_path = os.path.join(source_folder, "file_mapping.txt")

# .h5 ファイルを取得
all_h5_files = [f for f in os.listdir(source_folder) if f.endswith(".h5")]

# ランダムに250個選択
selected_files = random.sample(all_h5_files, 250)

# コピー先フォルダが存在しない場合は作成
os.makedirs(destination_folder, exist_ok=True)

# 対応表を記録するためのリスト
mapping_list = []

# ファイルをコピーして名前を変更
for i, file_name in enumerate(selected_files, start=1):
    src_path = os.path.join(source_folder, file_name)
    new_file_name = f"Number{i}.h5"
    dest_path = os.path.join(destination_folder, new_file_name)

    # ファイルをコピー
    shutil.copy2(src_path, dest_path)

    # 対応表に追加
    mapping_list.append(f"{file_name} -> {new_file_name}")

# 対応表を保存
with open(mapping_file_path, "w", encoding="utf-8") as f:
    f.write("\n".join(mapping_list))

print(f"250個のファイルを {destination_folder} にコピーしました。")
print(f"対応表を {mapping_file_path} に保存しました。")
