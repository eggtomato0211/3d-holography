import os
import re
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.cm as cm
import numpy as np

# ベースパスを複数提示して比較したい場合は、以下のようにリストで指定する
model_names = [
    # r'patch=64_stride=8_fm=16_valpatch=64', #Stride
    # r'patch=64_stride=8_fm=64_valpatch=128', # Match TrainPatch
    # r'patch=64_stride=16_fm=16_valpatch=64', # Match # ValPatch
    # r'patch=64_stride=16_fm=16_valpatch=128', # Match # ValPatch TrainPatch
    # r'patch=64_stride=32_fm=16_valpatch=64', #Stride
    # r'patch=96_stride=4_fm=16_valpatch=128',
    # r'patch=96_stride=24_fm=16_valpatch=128',
    # r'patch=96_stride=24_fm=64_valpatch=128',
    # r'patch=128_stride=1_fm=64_valpatch=128',
    # r'patch=128_stride=2_fm=64_valpatch=128',
    # r'patch=128_stride=4_fm=64_valpatch=128',
    r'patch=128_stride=8_fm=16_valpatch=128', #Stride #Fmap
    r'patch=128_stride=8_fm=64_valpatch=128', # Match TrainPatch #Fmap
    r'patch=128_stride=16_fm=16_valpatch=128', #Stride TrainPatch #Fmap
    r'patch=128_stride=16_fm=64_valpatch=128', #Fmap
    r'patch=128_stride=32_fm=16_valpatch=128', #Stride #Fmap
    r'patch=128_stride=32_fm=64_valpatch=128', #Fmap
    # r'patch=128_stride=64_fm=64_valpatch=128',
    # r'patch=128_stride=128_fm=64_valpatch=128',
]
base_paths = []

# モデルのベースパス最後にlogsを入れたい
for i in range(len(model_names)):
    path = fr'C:\Users\Owner\mizusaki\pytorch-3dunet\checkpoint\32x32x128\{model_names[i]}\logs'
    base_paths.append(path) 

def extract_label_from_path_updated(base_path):
    """
    パスから stride, fm, patch, valpatch の情報を抽出する関数
    """
    # stride のパターンを探す
    match_stride = re.search(r"stride=([\d-]+)", base_path)
    stride = match_stride.group(1) if match_stride else None

    # feature_map のパターンを探す
    match_fm = re.search(r"fm=(\d+)", base_path)
    fm = int(match_fm.group(1)) if match_fm else None

    # patch のパターンを探す
    match_patch = re.search(r"patch=([\d-]+)", base_path)
    patch = match_patch.group(1) if match_patch else None

    # valpatch のパターンを探す
    match_valpatch = re.search(r"valpatch=([\d-]+)", base_path)
    valpatch = match_valpatch.group(1) if match_valpatch else None

    return stride, fm, patch, valpatch

def read_logs(base_path):
    training_file_path = os.path.join(base_path, "training_log.txt")
    validation_file_path = os.path.join(base_path, "validation_log.txt")

    # Training log用データ
    training_iterations = []
    training_losses = []
    training_PSNR = []

    if os.path.exists(training_file_path):
        with open(training_file_path, 'r') as file:
            for line in file:
                match = re.match(r"Iteration: (\d+), Loss: ([\d.]+), Evaluation score: ([\d.]+)", line)
                if match:
                    training_iterations.append(int(match.group(1)))
                    training_losses.append(float(match.group(2)))
                    training_PSNR.append(float(match.group(3)))
                
    # Validation log用データがない場合は、エラーを出力
    else:
        raise Exception(f"Validation log file not found: {validation_file_path}")

    # Validation log用データ
    validation_iterations = []
    validation_losses = []
    validation_PSNR = []

    if os.path.exists(validation_file_path):
        with open(validation_file_path, 'r') as file:
            for line in file:
                match = re.match(
                    r"Iteration: (\d+), Validation finished. Loss: ([\d.]+). Evaluation score: ([\d.]+)", 
                    line
                )
                if match:
                    validation_iterations.append(int(match.group(1)))
                    validation_losses.append(float(match.group(2)))
                    validation_PSNR.append(float(match.group(3)))
                else:
                    # Iteration: 20000, Loss: 0.0008170859481906518, PSNR (PSNR): 36.854768128530935
                    match = re.match(r"Iteration: (\d+), Loss: ([\d.]+), Evaluation score \(PSNR\): ([\d.]+)", line)

                    if match:
                        validation_iterations.append(int(match.group(1)))
                        validation_losses.append(float(match.group(2)))
                        validation_PSNR.append(float(match.group(3)))

    return {
        "training_iterations": training_iterations,
        "training_losses": training_losses,
        "training_PSNR": training_PSNR,
        "validation_iterations": validation_iterations,
        "validation_losses": validation_losses,
        "validation_PSNR": validation_PSNR
    }

def generate_labels(base_paths):
    """ ラベルを生成する関数 """
    labels = []
    for bp in base_paths:
        stride, fm, patch, valpatch = extract_label_from_path_updated(bp)
        label = f"feature_map={fm}, patch_size={patch}, stride_size={stride}, valpatch_size={valpatch}"
        labels.append(label)
    return labels

def plot_combined_results(all_data, base_paths):
    """ ログデータを元にグラフを描画 """
    labels = generate_labels(base_paths)

    # # Training Lossのグラフ
    # plt.figure(figsize=(10, 6))
    # for data, label in zip(all_data, labels):
    #     plt.plot(data["training_iterations"], data["training_losses"], label=f"Training Loss ({label})")
    # plt.title("Training Loss")
    # plt.xlabel("Iteration")
    # plt.ylabel("Loss")
    # plt.legend()
    # plt.grid(True)
    # plt.show()

    # # Training PSNRのグラフ
    # plt.figure(figsize=(10, 6))
    # for data, label in zip(all_data, labels):
    #     plt.plot(data["training_iterations"], data["training_PSNR"], label=f"Training PSNR ({label})")
    # plt.title("Training PSNR")
    # plt.xlabel("Iteration")
    # plt.ylabel("PSNR")
    # plt.legend()
    # plt.grid(True)
    # plt.show()

    # # Validation Lossのグラフ
    # plt.figure(figsize=(10, 6))
    # for data, label in zip(all_data, labels):
    #     plt.plot(data["validation_iterations"], data["validation_losses"], label=f"{label}")
    # plt.title("Validation Loss")
    # plt.xlabel("Iteration")
    # plt.ylabel("Loss")
    # plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1))
    # plt.grid(True)
    # plt.show()

    # # Validation PSNRのグラフ
    # plt.figure(figsize=(10, 6))
    # for data, label in zip(all_data, labels):
    #     plt.plot(data["validation_iterations"], data["validation_PSNR"], label=f"{label}")
    # plt.title("Validation PSNR")
    # plt.xlabel("Iteration")
    # plt.ylabel("PSNR")
    # plt.legend(loc='lower center', bbox_to_anchor=(0.5, 0))
    # plt.grid(True)
    # plt.show()

    # Validation PSNRのグラフ（patchとvaipatchが一致しているかどうかで色分け、patch sizeで線種をかえる）
    # 1) patch == valpatch かどうかで色を分ける設定
    #   True(一致) → 青("blue"), False(不一致) → 赤("red") としてみる
    color_map = {
        True: "purple",
        False: "orange"
    }

    # 2) base_paths からユニークな stride を取り出し、line style を割り当てる
    unique_strides = set()
    for bp in base_paths:
        stride, fm, patch, valpatch = extract_label_from_path_updated(bp)
        unique_strides.add(stride)

    # ソートしてリスト化
    unique_strides = sorted(unique_strides)

    # ラインスタイルの候補
    line_styles = ['-', '--', '-.', ':', (0, (3,1,1,1)), (0, (5,5))]

    # strideごとに異なる線のスタイルを割り当てる
    line_styles = ['-', '--', '-.', ':', (0, (3, 1, 1)), (0, (5, 5))]
    stride_to_style = {}
    for i, s in enumerate(unique_strides):
        stride_to_style[s] = line_styles[i % len(line_styles)]

    # 3) Validation PSNR のグラフを描画
    plt.figure(figsize=(10, 6))

    for bp, data in zip(base_paths, all_data):
        stride, fm, patch, valpatch = extract_label_from_path_updated(bp)

        # patch, valpatch が数値文字列の場合は int に変換して扱う
        patch_int = None
        valpatch_int = None
        try:
            patch_int = int(patch)
            valpatch_int = int(valpatch)
        except:
            pass  # 変換失敗時は None のまま扱う

        # patch == valpatch かどうか
        same_patch_flag = (patch_int == valpatch_int)

        # 色を決定
        color = color_map[same_patch_flag]

        # ラインスタイルを決定
        linestyle = stride_to_style[stride]


        # 表示用ラベル（お好みで変更）
        label = f"fm={fm}, patch={patch}, valpatch={valpatch}, stride={stride}"

        plt.plot(
            data["validation_iterations"],
            data["validation_PSNR"],
            label=label,
            color=color,
            linestyle=linestyle
        )

    plt.title("Validation PSNR (Color by patch==valpatch, Linestyle by patch size)")
    plt.xlabel("Iteration")
    plt.ylabel("PSNR")
    plt.legend(loc='lower center', bbox_to_anchor=(0.5, 0))
    plt.grid(True)
    plt.show()


    # Validation PSNRのグラフ（valpatchごとに色分け, stride sizeごとに線種を変える）
    # まず base_paths からユニークな valpatch と stride を取り出す
    unique_valpatches = set()
    unique_strides = set()
    for bp in base_paths:
        stride, fm, patch, valpatch = extract_label_from_path_updated(bp)
        if valpatch is not None:
            unique_valpatches.add(valpatch)
        if stride is not None:
            unique_strides.add(stride)

    # ソートしてリスト化（strideは数値化してソートしたい場合はint変換も可）
    unique_valpatches = sorted(unique_valpatches)
    # stride を数値順にソートする場合：unique_strides = sorted(unique_strides, key=lambda x: int(x))
    unique_strides = sorted(unique_strides)
    
    # valpatchごとに異なる色を振る（2職しかないのでRed, Greenで指定する）
    valpatch_to_color = {valpatch: c for valpatch, c in zip(unique_valpatches, ['r', 'g'])}

    # strideごとに異なる線のスタイルを割り当てる
    line_styles = ['-', '--', '-.', ':', (0, (3, 1, 1)), (0, (5, 5))]
    stride_to_style = {}
    for i, s in enumerate(unique_strides):
        stride_to_style[s] = line_styles[i % len(line_styles)]

    # Validation PSNRを色(=valpatch), 線種(=stride size)で区別しつつ表示
    plt.figure(figsize=(10, 6))

    for bp, data in zip(base_paths, all_data):
        
        stride, fm, patch, valpatch = extract_label_from_path_updated(bp)
        if (valpatch is None) or (stride is None):
            # 万が一抽出に失敗した場合はデフォルトの描画
            plt.plot(data["validation_iterations"], data["validation_PSNR"], label=bp)
        else:
            color = valpatch_to_color[valpatch]
            linestyle = stride_to_style[stride]
            label = f"fm={fm}, patch={patch}, valpatch={valpatch}, stride={stride}"
            plt.plot(
                data["validation_iterations"],
                data["validation_PSNR"],
                label=label,
                color=color,
                linestyle=linestyle
            )

    plt.title("Validation PSNR (Color=validation patch size)")
    plt.xlabel("Iteration")
    plt.ylabel("PSNR")
    plt.legend(loc='lower center', bbox_to_anchor=(0.5, 0))
    plt.grid(True)
    plt.show()

    # Validation PSNRのグラフ（stride sizeごとに色分け, patch sizeごとに線種をかえる）
    # まず base_paths からユニークな stride と patch を取り出す
    unique_strides = set()
    unique_patches = set()
    for bp in base_paths:
        stride, fm, patch, _ = extract_label_from_path_updated(bp)
        if stride is not None:
            unique_strides.add(stride)
        if patch is not None:
            unique_patches.add(patch)

    # ソートしてリスト化（strideは数値化してソートしたい場合はint変換も可）
    unique_strides = sorted(unique_strides)
    # stride を数値順にソートする場合：unique_strides = sorted(unique_strides, key=lambda x: int(x))
    unique_patches = sorted(unique_patches)

    # strideごとに異なる色を振る
    color_list = cm.viridis(np.linspace(0, 1, len(unique_strides)))
    stride_to_color = {stride: c for stride, c in zip(unique_strides, color_list)}

    # patchごとに異なる線のスタイルを割り当てる
    line_styles = ['-', '--', '-.', ':', (0, (3, 1, 1)), (0, (5, 5))]
    patch_to_style = {}
    for i, p in enumerate(unique_patches):
        patch_to_style[p] = line_styles[i % len(line_styles)]
    
    # Validation PSNRを色(=stride size), 線種(=patch size)で区別しつつ表示
    plt.figure(figsize=(10, 6))

    for bp, data in zip(base_paths, all_data):
        stride, fm, patch, valpatch = extract_label_from_path_updated(bp)
        if (stride is None) or (patch is None):
            # 万が一抽出に失敗した場合はデフォルトの描画
            plt.plot(data["validation_iterations"], data["validation_PSNR"], label=bp)
        else:
            color = stride_to_color[stride]
            linestyle = patch_to_style[patch]
            label = f"fm={fm}, patch={patch}, valpatch={valpatch}, stride={stride}"
            plt.plot(
                data["validation_iterations"],
                data["validation_PSNR"],
                label=label,
                color=color,
                linestyle=linestyle
            )

    plt.title("Validation PSNR (Color=stride size, Linestyle=patch size)")
    plt.xlabel("Iteration")
    plt.ylabel("PSNR")
    plt.legend(loc='lower center', bbox_to_anchor=(0.5, 0))
    plt.grid(True)
    plt.show()

    # Validation PSNRのグラフ（patch sizeごとに色分け、strideごとに線種をかえる）
    # まず base_paths からユニークな patch size と stride を取り出す
    unique_patches = set()
    unique_strides = set()
    for bp in base_paths:
        stride, fm, patch, _ = extract_label_from_path_updated(bp)
        if patch is not None:
            unique_patches.add(patch)
        if stride is not None:
            unique_strides.add(stride)

    # ソートしてリスト化（strideは数値化してソートしたい場合はint変換も可）
    unique_patches = sorted(unique_patches)
    # stride を数値順にソートする場合：unique_strides = sorted(unique_strides, key=lambda x: int(x))
    unique_strides = sorted(unique_strides)
    
    # patch sizeごとに異なる色を振る
    color_list = cm.PiYG(np.linspace(0, 1, len(unique_patches)))
    patch_to_color = {patch: c for patch, c in zip(unique_patches, color_list)}

    # strideごとに異なる線のスタイルを割り当てる
    line_styles = ['-', '--', '-.', ':', (0, (3, 1, 1)), (0, (5, 5))]
    stride_to_style = {}
    for i, s in enumerate(unique_strides):
        stride_to_style[s] = line_styles[i % len(line_styles)]

    # Validation PSNRを色(=patch size), 線種(=stride size)で区別しつつ表示
    plt.figure(figsize=(10, 6))

    for bp, data in zip(base_paths, all_data):
        stride, fm, patch, valpatch = extract_label_from_path_updated(bp)
        if (stride is None) or (patch is None):
            # 万が一抽出に失敗した場合はデフォルトの描画
            plt.plot(data["validation_iterations"], data["validation_PSNR"], label=bp)
        else:
            color = patch_to_color[patch]
            linestyle = stride_to_style[stride]
            label = f"fm={fm}, patch={patch}, valpatch={valpatch}, stride={stride}"
            plt.plot(
                data["validation_iterations"],
                data["validation_PSNR"],
                label=label,
                color=color,
                linestyle=linestyle
            )

    plt.title("Validation PSNR (Color=patch size, Linestyle=stride size)")
    plt.xlabel("Iteration")
    plt.ylabel("PSNR")
    plt.legend(loc='lower center', bbox_to_anchor=(0.5, 0))
    plt.grid(True)
    plt.show()

    # Validation PSNRのグラフ（feature mapごとに色分け、stride sizeごとに線種を変える）
    # まず base_paths からユニークな fm と stride を取り出す
    unique_fms = set()
    unique_strides = set()
    for bp in base_paths:
        stride, fm, _, _ = extract_label_from_path_updated(bp)
        if fm is not None:
            unique_fms.add(fm)
        if stride is not None:
            unique_strides.add(stride)

    # ソートしてリスト化（strideは数値化してソートしたい場合はint変換も可）
    unique_fms = sorted(unique_fms)
    # stride を数値順にソートする場合：unique_strides = sorted(unique_strides, key=lambda x: int(x))
    unique_strides = sorted(unique_strides)

    # fmごとに異なる色を振る
    color_list = cm.bwr(np.linspace(0, 1, len(unique_fms)))
    fm_to_color = {fm: c for fm, c in zip(unique_fms, color_list)}

    # strideごとに異なる線のスタイルを割り当てる
    line_styles = ['-', '--', '-.', ':', (0, (3, 1, 1, 1)), (0, (5, 5))]
    stride_to_style = {}
    for i, s in enumerate(unique_strides):
        stride_to_style[s] = line_styles[i % len(line_styles)]

    # Validation PSNRを色(=fm), 線種(=stride)で区別しつつ表示
    plt.figure(figsize=(10, 6))

    for bp, data in zip(base_paths, all_data):
        stride, fm, patch, valpatch = extract_label_from_path_updated(bp)
        if (stride is None) or (fm is None):
            # 万が一抽出に失敗した場合はデフォルトの描画
            plt.plot(data["validation_iterations"], data["validation_PSNR"], label=bp)
        else:
            color = fm_to_color[fm]
            linestyle = stride_to_style[stride]
            label = f"fm={fm}, patch={patch}, valpatch={valpatch}, stride={stride}"
            plt.plot(
                data["validation_iterations"],
                data["validation_PSNR"],
                label=label,
                color=color,
                linestyle=linestyle
            )

    plt.title("Validation PSNR (Color=feature map, Linestyle=stride size)")
    plt.xlabel("Iteration")
    plt.ylabel("PSNR")
    plt.legend(loc='lower center', bbox_to_anchor=(0.5, 0))
    plt.grid(True)
    plt.show()



def generate_psnr_table(all_data, base_paths):
    """ PSNRの表を生成 """
    psnr_data = []
    for base_path, data in zip(base_paths, all_data):
        max_psnr = max(data["validation_PSNR"]) if data["validation_PSNR"] else None
        stride, fm, patch, valpatch = extract_label_from_path_updated(base_path)

        if fm == 16:
            fm = [16, 32, 64, 128, 256]
        elif fm == 32:
            fm = [32, 64, 128, 256, 512]
        elif fm == 64:
            fm = [64, 128, 256, 512, 1024]

        # max_psnrは小数点以下1桁まで表示
        max_psnr = round(max_psnr, 1) if max_psnr is not None else None
        
        psnr_entry = {
            "fm": fm,
            "patch": patch,
            "stride": stride,
            "valpatch": valpatch,
            "Max PSNR": max_psnr
        }

        psnr_data.append(psnr_entry)

    # DataFrame に変換
    psnr_df = pd.DataFrame(psnr_data)

    # patchで昇順でソート
    # psnr_df = psnr_df.sort_values(by="patch")

    # タブ区切り形式で出力
    psnr_table_tab_separated = psnr_df.to_string(index=False, float_format="{:.1f}".format, justify="center")

    # 表示
    print(psnr_table_tab_separated)

print("Base Paths:", base_paths)

# データ収集
all_data = []
for base_path in base_paths:
    print(f"Processing logs in: {base_path}")
    data = read_logs(base_path)
    all_data.append(data)

# グラフのプロット
plot_combined_results(all_data, base_paths)

# PSNRの表を作成
generate_psnr_table(all_data, base_paths)
