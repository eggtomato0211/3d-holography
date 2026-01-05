import os
import multiprocessing as mp

# パラメータの設宁E
wav_len = 532.0 * 10**-9
Nx, Ny = 32, 32
dx = 3.45 * 10**-6
dy = dx
dz = 4 * 10**-6
pixels = 1
depthlevel = 128
# バッチ設宁E(Val用: 紁E00 samples)
# 10 plots ÁE5 image_numbers ÁE2 random_modes ÁE3 batches ÁE1 channel = 300
num_batches = 10
channels_per_batch = 1

# チE�EタセチE��の種顁E 'Train', 'Val', 'Test' など
# これを変更して実行することで、�E力フォルダを�Eけることができる
dataset_type = 'Val'

# `random_mode`がTrueならランダム、Falseなら等間隔�Edepthsを生戁E
random_mode = True

# plot数がrandomかどぁE��
random_plots_mode = False

# 位相をランダムに生�EするかどぁE��
random_phase_mode = False


def build_tasks():
    tasks = []
    for random_mode in [True, False]:
        for image_number in [8, 16, 32, 64, 128]:
            for number_of_plots in range(1, 11):
                for batch in range(num_batches):
                    tasks.append((random_mode, image_number, number_of_plots, batch))
    return tasks


def split_tasks(tasks, num_workers):
    buckets = [[] for _ in range(num_workers)]
    for idx, task in enumerate(tasks):
        buckets[idx % num_workers].append(task)
    return buckets


def run_tasks(task_list):
    from image_loader import ImageLoader
    from propagation import Propagation
    from phase_generator import PhaseGenerator
    from hdf_manager import HDFManager
    from reconstruction_processor import ReconstructionProcessor

    for random_mode, image_number, number_of_plots, batch in task_list:
        print(f"--- Batch {batch+1}/{num_batches} ---")
        print(f"--- random_mode: {random_mode}, image_number: {image_number}, plots: {number_of_plots} ---")

        mode_prefix = "random_" if random_mode else ""
        plots_mode_prefix = "randomPlots_" if random_plots_mode else ""

        # Trainフォルダ直下に保存（サブフォルダなし！E
        hdf_folder = f'D:/nosaka/data/clean_datasets/{dataset_type}'

        # ファイル名に条件を含める
        feature_desc = f"{image_number}images_{number_of_plots}plots"

        # number_of_plots個�Eビ�Eズが含まれるフォルダから読み込む
        image_folder = (
            f'C:\\Users\\Owner\\mizusaki\\3d-holography\\app\\python\\3d-imaging\\src\\'
            f'{plots_mode_prefix}{Nx}x{Ny}x{depthlevel}_1pxx{number_of_plots}'
        )

        # 画像�E読み込み�E�Eumber_of_plots個�Eビ�Eズが含まれる画像！E
        loader = ImageLoader(Nx, Ny, pixels, number_of_plots, channels_per_batch, depthlevel, folder_path=image_folder)
        images = loader.load_images()

        # 16bitから0~1の篁E��に正規化
        images = [img / 255.0 for img in images]

        # 波面の伝搬クラス
        propagation = Propagation(wav_len, dx, dy)

        # 位相生�Eクラス
        phase_gen = PhaseGenerator(random_phase_mode)

        # 再構�E処琁E��ラス
        reconstructor = ReconstructionProcessor(
            Nx,
            Ny,
            dx,
            dy,
            dz,
            depthlevel,
            channels_per_batch,
            images,
            random_mode,
            phase_gen,
            propagation,
        )

        # 吁E��ャンネルごとに再構�Eを行い、HDFに保孁E
        for channel in range(channels_per_batch):
            # IDオフセチE��計箁E Batch * Channels + Channel
            global_channel_id = batch * channels_per_batch + channel

            # process_channel に number_of_plots を渡す（アプローチA�E�E
            raw_data, label_data = reconstructor.process_channel(channel, image_number, number_of_plots)

            hdf_manager = HDFManager(128, 128, depthlevel, dz, hdf_folder)
            # ファイル番号を一意にする
            hdf_manager.save_to_hdf(
                raw_data,
                label_data,
                f"{feature_desc}_random{random_mode}_NumberFrom{global_channel_id * depthlevel + 1}.h5",
            )


def worker(gpu_id, task_list):
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    print(f"=== Worker start: GPU {gpu_id}, tasks {len(task_list)} ===")
    run_tasks(task_list)


if __name__ == "__main__":
    tasks = build_tasks()

    # GPUが複数ある場合は並列に分割
    gpu_ids = [0, 1, 2, 3]
    task_slices = split_tasks(tasks, len(gpu_ids))

    processes = []
    for gpu_id, task_list in zip(gpu_ids, task_slices):
        # プロセスごとにGPUを固定して起動
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        p = mp.Process(target=worker, args=(gpu_id, task_list))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
