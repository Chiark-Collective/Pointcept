"""
Submodule to supplement Pointcept with preprocessing scripts to handle heritage data.
"""
import os
import shutil
import laspy
import torch
import typing as ty
import numpy as np
import pandas as pd
import polars as pl
import xarray as xr

from pathlib import Path
from pointcept.supplemental.utils import *
from pyntcloud import PyntCloud


# Define the mappings from Polars to PCD data types
POLARS_PCD_TYPE_MAPPINGS = [
    (pl.Float32, ('F', 4)),
    (pl.Float64, ('F', 8)),
    (pl.UInt8, ('U', 1)),
    (pl.UInt16, ('U', 2)),
    (pl.UInt32, ('U', 4)),
    (pl.UInt64, ('U', 8)),
    (pl.Int16, ('I', 2)),
    (pl.Int32, ('I', 4)),
    (pl.Int64, ('I', 8)),
]
POLARS_TYPE_TO_PCD_TYPE = {dtype: mapping for dtype, mapping in POLARS_PCD_TYPE_MAPPINGS}


def read_pcd(file_path):
    cloud = PyntCloud.from_file(
        file_path
    )
    return cloud.points


def las_to_np_pth(input_las_path, category, scene_id, num_points=None, spoof_normal=True, spoof_gt=True, print_contents=True):
    """This function takes an input .las file and outputs a PyTorch state dictionary that is compatible with
    Pointcept's data config for Scannet."""
    
    category_root = ensure_category_dirs(category)

    if print_contents:
        read_las_file(input_las_path)

    with laspy.open(input_las_path) as file:
        las = file.read()

        # Determine the number of points to process
        total_points = len(las.points)
        if num_points is None:
            num_points = total_points  # Use all points if num_points is not given

        max_points = min(num_points, total_points)

        # Set the output filename based on whether num_points was explicitly given
        if num_points == total_points:
            output_pth_path = os.path.join(category_root, f"{scene_id}.pth")
        else:
            output_pth_path = os.path.join(category_root, f"{scene_id}_n{max_points}.pth")

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

        # Finally return the output filename
        return output_pth_path

def clear_directory(directory):
    """
    Removes all files and subdirectories within the specified directory.
    
    Parameters:
        directory (str): The path to the directory to be cleared.
        
    Raises:
        Exception: If any file or directory cannot be deleted, prints an error message.
    """   
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print(f"Failed to delete {file_path}. Reason: {e}")


def partition_pth_file(input_pth_filename, category, name_tag=None, num_voxels_x=None, num_voxels_y=None, num_voxels_z=None, print_contents=True):
    """This function takes a processed .pth file from the data_root directory,
    optionally splits it into the requested number of sub-voxels (defaulting unspecified dimensions to 1),
    creates subdirectories for train, test, and val, and puts the subvoxels into the 'val' directory within
    the specified output directory for inference."""
    
    # This dir is reserved for experiment results, stop the user
    # from specifying this name.
    if name_tag == "result":
        exit("Error: cannot specify 'result' as an experiment name, this dir is reserved for experiment \
            results. Please give another name.")


    category_root = ensure_category_dirs(category)

    # Load data from the .pth file
    # input_pth_path = os.path.join(category_root, input_pth_filename)
    data = torch.load(input_pth_filename)
    if print_contents:
        print_dict_structure(data)
    scene_id = data['scene_id']

    # Determine if any voxelization parameters are given
    voxelization_requested = num_voxels_x or num_voxels_y or num_voxels_z

    # Define the directory name based on voxelization
    if voxelization_requested:
        num_voxels_x = num_voxels_x or 1
        num_voxels_y = num_voxels_y or 1
        num_voxels_z = num_voxels_z or 1
        dir_name = f"{scene_id}_voxels_{num_voxels_x}x{num_voxels_y}x{num_voxels_z}"
    else:
        dir_name = f"{scene_id}"

    # Use name_tag if provided
    if name_tag:
        dir_name = name_tag

    base_output_directory = os.path.join(category_root, dir_name)

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
        original_file_path = os.path.join(val_directory, f'{data["scene_id"]}.pth')
        torch.save(data, original_file_path)
        print(f"Original data saved in {val_directory} with file {original_file_path}")


def create_pc_dataset(pc_dict: dict[str, ty.Any]) -> xr.Dataset:
    """
    Create an Xarray dataset from a point cloud dictionary.

    Args:
        pc_dict (Dict[str, Any]): A dictionary containing point cloud data with the following keys:
            - 'coord': The point coordinates as a 2D numpy array of shape (num_points, 3).
            - 'color': The point colors as a 2D numpy array of shape (num_points, 3).
            - 'normal': The point normals as a 2D numpy array of shape (num_points, 3).
            - 'semantic_gt20': The semantic labels (20 classes) as a 1D numpy array of shape (num_points,).
            - 'semantic_gt200': The semantic labels (200 classes) as a 1D numpy array of shape (num_points,).
            - 'instance_gt': The instance labels as a 1D numpy array of shape (num_points,).
            - 'scene_id': The scene ID as a string.

    Returns:
        xr.Dataset: An Xarray dataset representing the point cloud data.

    Raises:
        KeyError: If any of the required keys are missing from the input dictionary.
        ValueError: If the input arrays have inconsistent shapes or if the 'scene_id' is not a string.

    Example:
        pc_dict = {
            'coord': np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]),
            'color': np.array([[255, 0, 0], [0, 255, 0]]),
            'normal': np.array([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0]]),
            'semantic_gt20': np.array([1, 2]),
            'semantic_gt200': np.array([10, 20]),
            'instance_gt': np.array([1, 1]),
            'scene_id': 'scene_0001'
        }
        pc_dataset = create_pc_dataset(pc_dict)
    """
    # Check if all required keys are present in the input dictionary
    required_keys = [
        'coord', 'color', 'normal', 'gt', 'scene_id'
    ]
    for key in required_keys:
        if key not in pc_dict:
            raise KeyError(f"Missing required key '{key}' in the input dict")

    # Check if the input arrays have consistent shapes
    num_points = len(pc_dict['coord'])
    for key in ['color', 'normal']:
        if pc_dict[key].shape != (num_points, 3):
            raise ValueError(f"Array '{key}' has an inconsistent shape. Expected ({num_points}, 3).")
    for key in ['gt']: #, 'semantic_gt200', 'instance_gt']:
        if pc_dict[key].shape != (num_points,):
            raise ValueError(f"Array '{key}' has an inconsistent shape. Expected ({num_points},).")

    # Create an Xarray dataset from the point cloud dictionary
    pc_dataset = xr.Dataset(
        data_vars=dict(
            coord=(['point', 'coord_dim'], pc_dict['coord']),
            color=(['point', 'color_dim'], pc_dict['color']),
            normal=(['point', 'normal_dim'], pc_dict['normal']),
            gt=(['point'], pc_dict['gt']),
            # semantic_gt200=(['point'], pc_dict['semantic_gt200']),
            # instance_gt=(['point'], pc_dict['instance_gt'])
        ),
        coords=dict(
            point=np.arange(num_points),
            coord_dim=['x', 'y', 'z'],
            color_dim=['r', 'g', 'b'],
            normal_dim=['nx', 'ny', 'nz']
        ),
        attrs=dict(scene_id=pc_dict['scene_id'])
    )

    return pc_dataset


def write_pcd(filename, data, metadata=None, batch_size=500000, binary=False):
    """
    Writes a PCD file from a Polars DataFrame or LazyFrame in batches.

    Parameters
    ----------
    filename: str
        Path to the output PCD file.
    data: Polars DataFrame or LazyFrame
        DataFrame or LazyFrame containing the point cloud data.
    metadata: dict, optional
        Dictionary containing PCD metadata. If not provided, it will be
        generated from the data.
    batch_size: int, optional
        Size of each batch to be processed.
    binary: bool, optional
        If True, writes the PCD file in binary format. Otherwise, writes in ASCII format.
    """
    # Check if data is a LazyFrame
    is_lazy = isinstance(data, pl.LazyFrame)

    # Rename columns if they exist
    # rename_map = {'X': 'x', 'Y': 'y', 'Z': 'z'}
    # data = data.rename(rename_map)

    # Fill nans for eigenentropy
    if is_lazy:
        data = data.with_columns([
            pl.col("x").cast(pl.Float32),  # open3d seems to req single precision float
            pl.col("y").cast(pl.Float32),
            pl.col("z").cast(pl.Float32)
        ])
        data = data.select(pl.exclude(pl.String))
    else:
        data = data.with_columns([
            pl.col("x").cast(pl.Float32),  # open3d seems to req single precision float
            pl.col("y").cast(pl.Float32),
            pl.col("z").cast(pl.Float32)
        ])
        data = data.select(pl.exclude(pl.Utf8))

    if metadata is None:
        n_rows = data.height if not is_lazy else data.select(pl.col("*")).count().item()
        # Generate metadata from the first batch of data
        first_batch = data.limit(batch_size).collect() if is_lazy else data.head(batch_size)
        metadata = {
            'version': '.7',
            'fields': first_batch.columns,
            'size': [POLARS_TYPE_TO_PCD_TYPE[dtype][1] for dtype in first_batch.dtypes],
            'type': [POLARS_TYPE_TO_PCD_TYPE[dtype][0] for dtype in first_batch.dtypes],
            'count': [1] * len(first_batch.columns),
            'width': n_rows,
            'height': 1,
            'viewpoint': [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            'points': n_rows,
            'data': 'binary' if binary else 'ascii'
        }

    with open(filename, 'wb') as f:
        # Write metadata
        f.write(f"VERSION {metadata['version']}\n".encode())
        f.write(f"FIELDS {' '.join(metadata['fields'])}\n".encode())
        f.write(f"SIZE {' '.join(map(str, metadata['size']))}\n".encode())
        f.write(f"TYPE {' '.join(metadata['type'])}\n".encode())
        f.write(f"COUNT {' '.join(map(str, metadata['count']))}\n".encode())
        f.write(f"WIDTH {metadata['width']}\n".encode())
        f.write(f"HEIGHT {metadata['height']}\n".encode())
        f.write(f"VIEWPOINT {' '.join(map(str, metadata['viewpoint']))}\n".encode())
        f.write(f"POINTS {metadata['points']}\n".encode())
        f.write(f"DATA {metadata['data']}\n".encode())

        # Process and write data in batches
        for start_row in range(0, metadata["points"], batch_size):
            batch_df = data.slice(start_row, batch_size).collect() if is_lazy else data[start_row:start_row+batch_size]
            if binary:
                # Write batch in binary format
                data_buffer = batch_df.to_pandas().to_records(index=False).tobytes()
                f.write(data_buffer)
            else:
                # Write batch in ASCII format
                np.savetxt(f, batch_df.to_numpy(), fmt=' '.join(['%s'] * batch_df.width))

    return filename


def convert_pcd_to_las(
    pcd_file_path: Path,
    las_file_path: Path,
    pred_col: str = "pred",
    gt_col: str = "gt"
) -> Path:
    # Read the PCD file
    pcd_data = read_pcd(str(pcd_file_path))

    # Create a new LAS file header
    header = laspy.LasHeader(version="1.4", point_format=2)
    
    # Add extra dimensions for predicted_class and predicted_proba
    header.add_extra_dim(laspy.ExtraBytesParams(name="predicted_class", type="f"))
    header.add_extra_dim(laspy.ExtraBytesParams(name="gt_class", type="f"))
    
    # Create an empty LAS file with the header
    las = laspy.LasData(header)

    # Fill the LAS file with data
    las.x, las.y, las.z = pcd_data['x'], pcd_data['y'], pcd_data['z']
    if "intensity" in pcd_data.columns:
        las.intensity = pcd_data['intensity']
    las.red, las.green, las.blue = pcd_data["r"], pcd_data["g"], pcd_data["b"]

    # Set values for the extra dimensions
    las.predicted_class = pcd_data[pred_col]
    las.gt_class = pcd_data[gt_col]

    # Write the LAS file to disk
    las.write(las_file_path)
    return las_file_path


def create_las_with_results(scenes_dir, results_dir, output_dir=None):
    """
    """
    scenes_dir = Path(scenes_dir)
    results_dir = Path(results_dir)

    if output_dir is None:
        output_dir = './'
    else:
        output_dir = os.path.join(output_dir, '')
    os.makedirs(output_dir, exist_ok=True)

    predictions = {
        "library_val": np.load(path) 
        for path in results_dir.glob("*.npy")
    }
    scenes = {
        "library_val": torch.load(path) 
        for path in scenes_dir.glob("*.pth")
    }

    # classes = [
    # 'wall', 'floor', 'cabinet', 'bed', 'chair', 'sofa', 'table', 'door',
    # 'window', 'bookshelf', 'picture', 'counter', 'desk', 'curtain',
    # 'refridgerator', 'shower curtain', 'toilet', 'sink', 'bathtub',
    # 'otherfurniture'
    # ]

    classes = [
    "wall",
    "floor",
    "roof",
    "ceiling",
    "footpath",
    "grass",
    "column",
    "door",
    "window",
    "stair",
    "railing",
    "rainwater pipe",
    "other",
    ]

    for scene_id, pred in predictions.items():
        points = scenes[scene_id]
        labels = points["gt"]
        ds = create_pc_dataset(points)
        ds = ds.assign_coords(classes=xr.DataArray(np.array(classes), dims=["classes"]))
        ds["pred"] =(("point",), pred)
        ds["color"] = ds["color"].astype(np.uint8)
        ds["gt"] = ds["gt"]
        df = pd.concat(
            [
                ds["coord"].to_pandas(),
                ds["color"].to_pandas(),
                ds["pred"].to_pandas().rename("pred"),
                ds["gt"].to_pandas().rename("gt")
            ],
            axis=1
        )
        lf = pl.from_pandas(df)
        pcd_out = f"{scene_id}.pcd"
        las_out = output_dir+f"{scene_id}.las"
        write_pcd(pcd_out, lf, binary=True)
        print(f"Write: {las_out}")
        convert_pcd_to_las(pcd_out, las_out)
        os.remove(pcd_out)


if __name__ == "__main__":
    # pth_file = las_to_np_pth('/data/sdd/training_v2.las', 'qh', 'scene01', num_points=10000)
    try:
        partition_pth_file(pth_file, 'qh', num_voxels_x=3)
    except NameError:
        partition_pth_file('./data/qh/scene01_n10000.pth', 'qh',)
