"""
Submodule to supplement Pointcept with preprocessing scripts to handle heritage data.
"""
import os
import shutil
import laspy
import torch
import numpy as np
import pandas as pd
from dotenv import load_dotenv


def las_to_np_pth(input_las_path, scene_id, num_points=None, spoof_normal=True, spoof_gt=True):
    """This function takes an input .las file and outputs a PyTorch state dictionary that is compatible with
    Pointcept's data config for Scannet."""
    
    # Load environment variables from the .env file
    load_dotenv()

    # Check if DATA_ROOT is set in the environment
    data_root = os.getenv('DATA_ROOT')
    if data_root is None:
        print("ERROR: DATA_ROOT environment variable not found.")
        exit(1)

    # Make the DATA_ROOT directory if it doesn't exist
    try:
        os.makedirs(data_root, exist_ok=True)
    except Exception as e:
        print(f"ERROR: Unable to create directory {data_root}. {e}")
        exit(1)

    with laspy.open(input_las_path) as file:
        las = file.read()

        # Determine the number of points to process
        total_points = len(las.points)
        if num_points is None:
            num_points = total_points  # Use all points if num_points is not given

        max_points = min(num_points, total_points)

        # Set the output filename based on whether num_points was explicitly given
        if num_points == total_points:
            output_pth_path = os.path.join(data_root, f"{scene_id}.pth")
        else:
            output_pth_path = os.path.join(data_root, f"{scene_id}_n{max_points}.pth")

        # Read and adjust coordinates to start at zero
        x_adjusted = las.x[:max_points] - np.min(las.x[:max_points])
        y_adjusted = las.y[:max_points] - np.min(las.y[:max_points])
        z_adjusted = las.z[:max_points] - np.min(las.z[:max_points])
        
        # Create the coordinate array
        coord = np.stack((x_adjusted, y_adjusted, z_adjusted), axis=-1).astype(np.float32)
        
        # Check if RGB data exists
        if hasattr(las, 'red') and hasattr(las, 'green') and hasattr(las, 'blue'):
            color = np.stack((las.red[:max_points], las.green[:max_points], las.blue[:max_points]), axis=-1).astype(np.float32)
            color /= 256.0  # Normalize color
        else:
            color = np.zeros((max_points, 3), dtype=np.float32)  # Default to black
            print("RGB data not found in LAS file. Defaulting to black for color.")

        # Save numpy arrays into a dictionary and then to a .pth file
        data = {
            'coord': coord,
            'color': color,
            'scene_id': scene_id,
        }

        # Generate random normals if required
        if spoof_normal:
            normal = np.random.rand(max_points, 3).astype(np.float32)
            norm = np.linalg.norm(normal, axis=1, keepdims=True)
            normal /= norm
            data['normal'] = normal

        # Generate random ground truth if required
        if spoof_gt:
            semantic_gt20 = np.random.randint(0, 20, size=max_points, dtype=np.int64)
            data['semantic_gt20'] = semantic_gt20
        
        # Save the dictionary as a .pth file
        torch.save(data, output_pth_path)
        print(f"Saved {max_points} points to {output_pth_path}")


def clear_directory(directory):
    """Clears the specified directory of all files and subdirectories."""
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print(f"Failed to delete {file_path}. Reason: {e}")


def partition_pth_file(input_pth_filename, name_tag=None, num_voxels_x=None, num_voxels_y=None, num_voxels_z=None):
    """This function takes a processed .pth file from the data_root directory,
    optionally splits it into the requested number of sub-voxels (defaulting unspecified dimensions to 1),
    creates subdirectories for train, test, and val, and puts the subvoxels into the 'val' directory within the specified output directory."""
    
    # Load environment variables from the .env file
    load_dotenv()

    # Get DATA_ROOT from environment
    data_root = os.getenv('DATA_ROOT')
    if data_root is None:
        print("ERROR: DATA_ROOT environment variable not found.")
        exit(1)

    # Determine the output base directory name based on name_tag or default to structured naming
    # if name_tag:
    #     base_output_directory = os.path.join(data_root, name_tag)
    # else:
    #     base_output_directory = os.path.join(data_root, "output_subvoxels")



    # Load data from the .pth file
    input_pth_path = os.path.join(data_root, input_pth_filename)
    data = torch.load(input_pth_path)
    scene_id = data['scene_id']

    # Determine if any voxelization parameters are given
    voxelization_requested = num_voxels_x or num_voxels_y or num_voxels_z

    # Define the directory name based on voxelization
    if voxelization_requested:
        num_voxels_x = num_voxels_x or 1
        num_voxels_y = num_voxels_y or 1
        num_voxels_z = num_voxels_z or 1
        dir_name = f"data_{scene_id}_voxels_{num_voxels_x}x{num_voxels_y}x{num_voxels_z}"
    else:
        dir_name = f"data_{scene_id}"

    # Use name_tag if provided
    if name_tag:
        dir_name = name_tag

    base_output_directory = os.path.join(data_root, dir_name)

    # Check if the base output directory exists and clear it if it does
    if os.path.exists(base_output_directory):
        print(f"Clearing existing directory {base_output_directory}")
        clear_directory(base_output_directory)
    
    os.makedirs(base_output_directory, exist_ok=True)
    os.makedirs(os.path.join(base_output_directory, 'test'), exist_ok=True)
    os.makedirs(os.path.join(base_output_directory, 'train'), exist_ok=True)
    val_directory = os.path.join(base_output_directory, 'val')
    os.makedirs(val_directory, exist_ok=True)

    # Check if any voxelization parameter is given and default others to 1
    if voxelization_requested:
        coords = data['coord']

        # Calculate the bounds and voxel sizes
        min_coords = coords.min(axis=0)
        max_coords = coords.max(axis=0)
        voxel_size = (max_coords - min_coords) / np.array([num_voxels_x, num_voxels_y, num_voxels_z])

        # Calculate voxel indices for each point
        voxel_indices = np.floor((coords - min_coords) / voxel_size).astype(int)
        voxel_indices = np.clip(voxel_indices, 0, [num_voxels_x-1, num_voxels_y-1, num_voxels_z-1])

        # Loop over all voxels and save them in the 'val' directory
        for ix in range(num_voxels_x):
            for iy in range(num_voxels_y):
                for iz in range(num_voxels_z):
                    # Find points in this voxel
                    voxel_mask = (voxel_indices[:, 0] == ix) & (voxel_indices[:, 1] == iy) & (voxel_indices[:, 2] == iz)
                    if voxel_mask.any():
                        voxel_data = {key: data[key][voxel_mask] for key in data if isinstance(data[key], np.ndarray)}
                        unique_scene_id = f"{data['scene_id']}_voxel_{ix}_{iy}_{iz}"
                        voxel_data['scene_id'] = unique_scene_id  # Assign unique scene_id

                        # Path for the new split file
                        voxel_file_path = os.path.join(val_directory, f'{unique_scene_id}.pth')
                        torch.save(voxel_data, voxel_file_path)

        print(f"All voxel files saved in {val_directory}")
    else:
        # No voxelization requested, save the original data in the 'val' directory with a modified scene_id
        # original_scene_id = data['scene_id']
        # data['scene_id'] = f"{original_scene_id}_original"
        original_file_path = os.path.join(val_directory, f'{data["scene_id"]}.pth')
        torch.save(data, original_file_path)
        print(f"Original data saved in {val_directory} with file {original_file_path}")


if __name__ == "__main__":
    # las_to_np_pth('/data/sdd/training_v2.las', 'scene01', num_points=10000)
    partition_pth_file("scene01_n10000.pth", name_tag="my_experiment", num_voxels_x=3)
