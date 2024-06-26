import os
import numpy as np
import h5py

class makeHDF:
    def __init__(self):
        self.output_dir = r'.\\app\\python\\3d-imaging\\hdf'
        
    def makeHDF(self, raw_4d, label_4d, output_fileName):
        # 出力ファイル名
        self.output_file = os.path.join(self.output_dir, output_fileName)
        with h5py.File(self.output_file, 'w') as f:
            # データセットの作成
            f.create_dataset('raw', data=raw_4d, compression='gzip')
            f.create_dataset('label', data=label_4d, compression='gzip')

        print(f"HDF5ファイル '{self.output_file}' が 'raw' と 'label' のデータセットで作成されました")
        
        return