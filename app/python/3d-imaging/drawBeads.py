import os
import random
from PIL import Image, ImageDraw
import concurrent.futures
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 3D space dimensions
x_dim, y_dim, z_dim = 1024, 1024, 800
sphere_diameter = 12
sphere_radius = sphere_diameter // 2

# 保存用ディレクトリがない場合は作成
save_dir = f"C:\\Users\\Owner\\mizusaki\\3d-holography\\app\python\\3d-imaging\\src\\beads\\{x_dim}x{y_dim}x{z_dim}_{sphere_diameter}px"
print(save_dir)
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# Initialize 3D space with zeros
space = np.zeros((x_dim, y_dim, z_dim))

# Function to place a sphere at given center coordinates
def place_sphere(center, space):
    cx, cy, cz = center
    for x in range(cx - sphere_radius, cx + sphere_radius + 1):
        for y in range(cy - sphere_radius, cy + sphere_radius + 1):
            for z in range(cz - sphere_radius, cz + sphere_radius + 1):
                if (x - cx)**2 + (y - cy)**2 + (z - cz)**2 <= sphere_radius**2:
                    if 0 <= x < x_dim and 0 <= y < y_dim and 0 <= z < z_dim:
                        space[x, y, z] = 1

# Example: Randomly place 100 spheres in the 3D space
np.random.seed(0)  # For reproducibility
num_spheres = 100
for _ in range(num_spheres):
    center = (np.random.randint(sphere_radius, x_dim - sphere_radius),
              np.random.randint(sphere_radius, y_dim - sphere_radius),
              np.random.randint(sphere_radius, z_dim - sphere_radius))
    place_sphere(center, space)

# Save each slice in z direction
for z in range(z_dim):
    plt.imshow(space[:, :, z], cmap="gray")
    plt.axis("off")
    plt.savefig(f"{save_dir}/slice_{z:03d}.png", bbox_inches='tight', pad_inches=0)
    plt.close()

print("画像の生成が完了しました。")