import os
import sys
import glob
# Add path to import HDF
sys.path.append(r'C:\Users\Owner\mizusaki\3d-holography\app\python\3d-imaging')
from HDF import HDF

def main():
    # Input directory (from previous step)
    hdf_input_dir = r"D:\nosaka\simulation_data_test\hdf"
    
    # Output directories for visualization
    # We will save vis outputs next to the hdf folder
    vis_output_base = r"D:\nosaka\simulation_data_test"
    
    # Find all generated HDF5 files
    hdf_files = sorted(glob.glob(os.path.join(hdf_input_dir, "SimData_*.h5")))
    
    if not hdf_files:
        print(f"No HDF files found in {hdf_input_dir}")
        return

    # Initialize HDF helper (dims don't matter much for save/movie functions as they read from file, but good to be consistent)
    # The class usually takes output_dir in init, but methods allow overriding it.
    nx, ny, depth, dz = 128, 128, 128, 4e-6
    hdf_helper = HDF(nx, ny, depth, dz, output_dir=vis_output_base)

    print(f"Found {len(hdf_files)} files to visualize.")

    for hdf_path in hdf_files:
        filename = os.path.basename(hdf_path)
        base_name = os.path.splitext(filename)[0] # e.g. SimData_00001
        
        print(f"Processing {filename}...")
        
        # 1. Generate Movie
        # Output: D:\nosaka\simulation_data_test\movies\SimData_00001\raw_movie.mp4
        movie_out_dir = os.path.join(vis_output_base, "movies", base_name)
        print(f"  Generating movies in {movie_out_dir}")
        
        # 'raw' dataset
        try:
            hdf_helper.make_movie('raw', hdf_path, movie_out_dir)
        except Exception as e:
            print(f"  Error making raw movie: {e}")

        # 'label' dataset
        try:
            hdf_helper.make_movie('label', hdf_path, movie_out_dir)
        except Exception as e:
            print(f"  Error making label movie: {e}")

        # 2. Save Slices (optional, can be many files)
        # Let's verify valid slices. User might want checking.
        # Output: D:\nosaka\simulation_data_test\images\SimData_00001\raw\raw_00000.png ...
        image_out_dir = os.path.join(vis_output_base, "slice_images", base_name)
        print(f"  Saving slice images in {image_out_dir}")
        
        try:
            hdf_helper.save_images('raw', hdf_path, image_out_dir)
        except Exception as e:
            print(f"  Error saving raw images: {e}")
            
        try:
            hdf_helper.save_images('label', hdf_path, image_out_dir)
        except Exception as e:
            print(f"  Error saving label images: {e}")

    print("Visualization completed.")

if __name__ == "__main__":
    main()
