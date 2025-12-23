import numpy as np
import cv2
import matplotlib.pyplot as plt
import time
import concurrent.futures
import os
from HDF import HDF

# 3D imaging path: C:\Users\Owner\mizusaki\3d-holography\app\python\3d-imaging
# Needs to import HDF, so we might need to add this path to sys.path if not running from there.
import sys
sys.path.append(r'C:\Users\Owner\mizusaki\3d-holography\app\python\3d-imaging')

def load_image(i, Nx, Ny, depthlevel, pixels, number_of_plots):
    # Updated path to match the directory structure we saw: 32x32x128_1pxx1
    # Note: The folder name in list_dir was '32x32x128_1pxx1'
    # The original code had flexible naming. I will hardcode the found directory for reliability.
    path = rf'C:\Users\Owner\mizusaki\3d-holography\app\python\3d-imaging\src\32x32x128_{pixels}pxx{number_of_plots}\image_{i:05d}.tiff'
    
    # If .tiff doesn't exist, try .png (original code used .png but I saw .tiff in list_dir)
    if not os.path.exists(path):
        path = path.replace('.tiff', '.png')
    
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Image not found: {path}")
    return img.astype(float)

def load_images(Nx, Ny, pixels, number_of_plots, channels, depthlevel):
    # Sequential loading to avoid issues, or simple loop
    images = []
    total_images = channels * depthlevel # 1 * 128 = 128 images needed per channel block
    # Actually logic in original code: range(1, channels * depthlevel + 1)
    # The folder I saw has 10000 images, so we have plenty.
    
    for i in range(1, total_images + 1):
        images.append(load_image(i, Nx, Ny, depthlevel, pixels, number_of_plots))
    return images

def nearpropCONV(Comp1, sizex, sizey, dx, dy, wa, d):
    if d == 0:
        return Comp1
    
    x1, x2 = -sizex//2, sizex//2-1
    y1, y2 = -sizey//2, sizey//2-1
    Fx, Fy = np.meshgrid(np.arange(x1, x2+1), np.arange(y1, y2+1))

    Fcomp1 = np.fft.fftshift(np.fft.fft2(Comp1)) / np.sqrt(sizex * sizey)
    FresR = np.exp(-1j * np.pi * wa * d * ((Fx**2) / ((dx * sizex)**2) + (Fy**2) / ((dy * sizey)**2)))
    Fcomp2 = Fcomp1 * FresR
    Recon = np.fft.ifft2(np.fft.ifftshift(Fcomp2)) * np.sqrt(sizex * sizey)
    return Recon


def process_image(args):
    # Simplified version for single thread or managed thread
    n, images, sizex, sizey, dx, dy, wav_len, dz, channel, depthlevel = args
    input_image = images[n + channel * depthlevel] * np.exp(1j) # Adding phase 0 for simplicity if random_mode is off
    output_3dimage = np.zeros((depthlevel, sizex, sizey), dtype=float)
    
    for z in range(depthlevel):
        # Propagation distance: (n - z) * dz? 
        # Original code: (n - z) * dz. 
        # Ideally n is the depth where bead is PLACED.
        # If z matches n, d=0. 
        dist = (n - z) * dz
        output_3dimage[z] = np.abs(nearpropCONV(input_image, sizex, sizey, dx, dy, wav_len, dist))**2
    return output_3dimage

def main():
    # Parameters
    wav_len = 532.0 * 10**-9
    Nx, Ny = 32, 32
    target_Nx, target_Ny = 128, 128 # Output size
    dx = 3.45 * 10**-6
    dy = dx
    dz = 4 * 10**-6
    pixels = 1
    number_of_plots = 1 # Directory name suggests 1pxx1
    
    depthlevel = 128
    channels = 10 # Generate a small batch, user asked to "create data", 10 samples?
    # User didn't specify count, but "create data" implies a usable amount. 
    # Let's do 5 samples to be quick but verifying it works.
    
    image_number = 64 # beads per volume
    
    # Output paths
    output_base_dir = r"D:\nosaka\simulation_data_test"
    images_out_dir = os.path.join(output_base_dir, "images")
    hdf_out_dir = os.path.join(output_base_dir, "hdf")
    
    os.makedirs(images_out_dir, exist_ok=True)
    os.makedirs(hdf_out_dir, exist_ok=True)
    
    print("Loading images...")
    # We need enough images for 'channels' * 'depthlevel'. 
    # But wait, logic is: pick 'image_number' random depths. Each bead is a separate image?
    # load_images loads (channels * depthlevel) images. 
    # So if channels=5, depthlevel=128, we need 640 images. Directory has 10000. OK.
    images = load_images(Nx, Ny, pixels, number_of_plots, channels, depthlevel)
    print("Images loaded.")
    
    print("Starting generation...")
    
    for channel in range(channels):
        raw_data = np.zeros((depthlevel, Nx, Ny), dtype=float)
        label_data = np.zeros((depthlevel, Nx, Ny), dtype=float)

        random_depths = np.random.randint(0, depthlevel, image_number)

        print(f"Processing sample {channel+1}/{channels}")
        
        # Prepare patches
        # We need to process each bead (which is at 'depth' in random_depths)
        # But wait, 'images' list is accessed by [channel * depthlevel + depth].
        # If we pick random depths, we might be picking the same image index multiple times if not careful?
        # The original code:
        # label_data[depth] = images[channel * depthlevel + depth]
        # This implies 'images' is a pool of beads. 
        # And we use specific beads assigned to this channel?
        
        # Let's follow original logic exactly for safety.
        # Original: args_list = [(n, images, ... channel) for n in random_depths]
        # process_image uses images[n + channel*depthlevel]
        # So it uses the bead image corresponding to that random depth slot?
        # That means if random_depths has duplicates, we add the same bead twice? Yes.
        
        for depth in random_depths:
            # Accumulate label (beads placement)
            # Note: if multiple beads at same depth, we might overwrite or sum?
            # Original code: label_data[depth, :, :] = images[...]
            # Overwrites. logic seems to assume simple overwrite or distinct depths? 
            # With 64 beads in 128 slots, collision is possible. Original code overwrites.
            label_data[depth, :, :] = images[channel * depthlevel + depth]

        # Calculate propagation for each bead
        # We can run this sequentially or parallel. 
        # Using simple loop to avoid ThreadPoolExecutor issues if any.
        
        # Accumulate intensity
        for n in random_depths:
            # Propagate this single bead
            vol = process_image((n, images, Nx, Ny, dx, dy, wav_len, dz, channel, depthlevel))
            raw_data += vol # Sum intensities (incoherent)

        # Normalize and Resize
        min_val = np.min(raw_data)
        max_val = np.max(raw_data)
        if max_val > min_val:
            raw_data = (raw_data - min_val) / (max_val - min_val)
        
        min_lbl = np.min(label_data)
        max_lbl = np.max(label_data)
        if max_lbl > min_lbl:
            label_data = (label_data - min_lbl) / (max_lbl - min_lbl)

        # Resizing to 128x128x128 if needed
        if Nx != target_Nx:
            resized_raw = np.zeros((depthlevel, target_Nx, target_Ny), dtype=float)
            resized_lbl = np.zeros((depthlevel, target_Nx, target_Ny), dtype=float)
            
            for z in range(depthlevel):
                resized_raw[z] = cv2.resize(raw_data[z], (target_Nx, target_Ny), interpolation=cv2.INTER_LINEAR)
                resized_lbl[z] = cv2.resize(label_data[z], (target_Nx, target_Ny), interpolation=cv2.INTER_LINEAR)
            
            raw_data = resized_raw
            label_data = resized_lbl
            
        # Save HDF5
        # HDF class takes output dir in init
        hdf_maker = HDF(target_Nx, target_Ny, depthlevel, dz, hdf_out_dir)
        hdf_maker.makeHDF(raw_data, label_data, f"SimData_{channel+1:05d}.h5")
        
        # Save a preview image of middle slice
        plt.imsave(os.path.join(images_out_dir, f"preview_{channel+1:05d}.png"), raw_data[depthlevel//2], cmap='gray')

    print("Done!")

if __name__ == "__main__":
    main()
