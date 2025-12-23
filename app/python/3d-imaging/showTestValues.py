import os
import re
# model のリスト
model_names = [
    r'patch=64_stride=8_fm=16_valpatch=64',
    r'patch=64_stride=8_fm=64_valpatch=128',
    r'patch=64_stride=16_fm=16_valpatch=64',
    r'patch=64_stride=16_fm=16_valpatch=128',
    r'patch=64_stride=32_fm=16_valpatch=64',
    r'patch=96_stride=24_fm=16_valpatch=128',
    r'patch=128_stride=1_fm=64_valpatch=128',
    r'patch=128_stride=2_fm=64_valpatch=128',
    r'patch=128_stride=4_fm=64_valpatch=128',
    r'patch=128_stride=8_fm=16_valpatch=128',
    r'patch=128_stride=8_fm=64_valpatch=128',
    r'patch=128_stride=16_fm=16_valpatch=128',
    r'patch=128_stride=16_fm=64_valpatch=128',
    r'patch=128_stride=32_fm=16_valpatch=128',
    r'patch=128_stride=32_fm=64_valpatch=128',
    r'patch=128_stride=64_fm=64_valpatch=128',
    r'patch=128_stride=128_fm=64_valpatch=128',
]

# フォルダのパスがあるかを確認
for model_name in model_names:
    check_path = fr'C:\Users\Owner\mizusaki\3d-holography\app\python\3d-imaging\values\test250\{model_name}'
    if not os.path.exists(check_path):
        raise Exception(f'Path does not exist: {check_path}')

# 各 model_name ごとに処理する関数
# 各model_nameにはprediction_psnr_average: 36.62084568849496　という形でpsnr.txtに記録されている
# この値を取得して、model_nameと一緒に出力する
def process_model_name(model_name):
    # txt野中の値を取得
    psnr_file = fr'C:\Users\Owner\mizusaki\3d-holography\app\python\3d-imaging\values\test250\{model_name}\psnr.txt'
    with open(psnr_file, 'r') as file:
        for line in file:
            match = re.match(r"prediction_psnr_average: ([\d.]+)", line)
            if match:
                prediction_psnr_average = float(match.group(1))            
                print(f'model_name: {model_name}, prediction_psnr_average: {prediction_psnr_average}')
                return prediction_psnr_average

# 各model_nameに対して処理を行い表を作成
model_name_psnr = []
for model_name in model_names:
    model_name_psnr.append([model_name, process_model_name(model_name)])

# 表を出力
print(model_name_psnr)