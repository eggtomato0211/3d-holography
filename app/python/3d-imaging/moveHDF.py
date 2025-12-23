import os
import shutil
import random

# 元のhdfディレクトリ
hdf_base_path = 'C:\\Users\\Owner\\mizusaki\\3d-holography\\hdf'

# 新しい移動先のディレクトリ
new_base_path = 'C:\\Users\\Owner\\mizusaki\\3d-holography\\app\\python\\3d-imaging\\hdf\\32x32x128_test'

# 新しいディレクトリが存在しない場合は作成
if not os.path.exists(new_base_path):
    os.makedirs(new_base_path)

# # ベースパス内のファイルを確認
# for file_name in os.listdir(hdf_base_path):
#     file_path = os.path.join(hdf_base_path, file_name)

#     # ファイルであることを確認
#     if os.path.isfile(file_path):
#         new_file_path = os.path.join(new_base_path, file_name)
#         shutil.copy(file_path, new_file_path)

        # # 同名のファイルが既に存在しないか確認
        # if not os.path.exists(new_file_path):
        #     # 20%の確率でコピーを作成
        #     if random.random() < 0.2:
        #         shutil.copy(file_path, new_file_path)
        #         print(f'Copied {file_name} to {new_file_path}')
        # else:
        #     print(f'File {file_name} already exists in {new_base_path}, skipping.')

# /hdf内の各サブディレクトリを確認
for folder_name in os.listdir(hdf_base_path):
    folder_path = os.path.join(hdf_base_path, folder_name)

    # サブディレクトリの場合
    if os.path.isdir(folder_path):
        print(f'Checking {folder_name}...')
        # フォルダ内のファイルをランダム移動
        for file_name in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file_name)

            # ファイルが存在する場合
            if os.path.isfile(file_path):
                new_file_path = os.path.join(new_base_path, file_name)

                # 同名のファイルが既に存在しないか確認
                if not os.path.exists(new_file_path):
                    print(f'Moved {file_name} to {new_file_path}')
                    # コピーしたファイルを新しいディレクトリに移動
                    shutil.copy(file_path, new_file_path)
                    # # ファイルを新しいディレクトリに20%の確立でコピーさせたものを移動
                    # if random.random() < 0.2:
                    #     # コピーしたファイルを新しいディレクトリに移動
                    #     shutil.copy(file_path, new_file_path)
                    #     print(f'Moved {file_name} to {new_file_path}')
                else:
                    print(f'File {file_name} already exists in {new_base_path}, skipping.')