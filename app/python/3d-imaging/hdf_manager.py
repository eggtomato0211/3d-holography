from HDF import HDF

class HDFManager:
    def __init__(self, Nx, Ny, depthlevel, dz, output_hdfdir):
        self.hdf_maker = HDF(Nx, Ny, depthlevel, dz, output_hdfdir)

    def save_to_hdf(self, raw_data, label_data, filename):
        self.hdf_maker.makeHDF(raw_data, label_data, filename)
