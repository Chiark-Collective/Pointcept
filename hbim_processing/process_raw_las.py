"""
This script will take a category of hbim data ("park_row", "rog_north" etc) and 
do the following:

1. reads the finely sampled cloud .las files for each category present in the hbim set.
2. applies a scaling operation to all axes to convert from cm to m
3. subsamples the clouds according to the specified voxelisation
4. switches the Z and Y axes so Z is vertical, as expected by Pointcept
5. adds a new uniform scalar field to each input cloud called "gt", for the ground truth
   category.
6. merges the results
7. uses the chipper filter to create a set of large voxels across the file to partition
   the data into training and test samples
8. saves the resultant files and moves them to the appropriate directories

Needs to be run from the HBIM data root.
"""
import argparse
import pdal
import json
import glob
import os
import re
import shutil
import pprint
import sys
import laspy
import torch
import numpy as np

from pathlib import Path
from collections import namedtuple


# Constants
VALID_LABELS = [
    'park_row',
    'rog_north',
    'rog_south',
]
EXTRA_DIMS = "NormalX=float64, NormalY=float64, NormalZ=float64"
CATEGORIES = [
    '1_wall',
    '2_floor',
    '3_roof',
    '4_ceiling',
    '5_footpath',
    '6_grass',
    '7_column',
    '8_door',
    '9_window',
    '10_stair',
    '11_railing',
    '12_rwp',
    '13_other',
]


###############################################################################
# .las to .pth conversion
###############################################################################
def convert_all_las_to_pth(config):
    split_files = glob.glob(config.split_template)
    for f in split_files:
        las_to_pth(f)


def las_to_pth(in_f, num_points=None):
    """This function takes an input .las file and outputs a PyTorch state dictionary that is compatible with
    Pointcept's data config for HBIM data.
    """
    in_f = Path(in_f)

    output_dir = in_f.parent

    with laspy.open(str(in_f)) as file:
        las = file.read()

        # Determine the number of points to process
        total_points = len(las.points)
        if num_points is None:
            num_points = total_points  # Use all points if num_points is not given
        max_points = min(num_points, total_points)

        # Set the output filename
        output_pth_path = in_f.with_suffix('.pth')
        output_pth_path = output_pth_path.parent / 'pth' / output_pth_path.name
        if num_points != total_points:
            output_pth_path = output_pth_path.with_stem(f'{output_pth_path.stem}_n{num_points}')
        
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
            raise ValueError(
                "Error: input cloud does not contain RGB info. This is likely a .las version or \
                point format mismatch. Check how your .las clouds were saved."
                )

        # Extract normals if they exist
        if hasattr(las, 'NormalX') and hasattr(las, 'NormalY') and hasattr(las, 'NormalZ'):
            normal = np.stack((las.NormalX[:num_points], las.NormalY[:num_points], las.NormalZ[:num_points]), axis=-1)
            norm = np.linalg.norm(normal, axis=1, keepdims=True)
            normal = (normal / norm).astype(np.float32)  # Normalize normals
        else:
            raise ValueError("Error: missing normals. Check your .las format.")

        # Extract gt if it exists
        if hasattr(las, 'gt'):
            gt = las.gt[:num_points].astype(np.int64)
        else:
            raise ValueError("Error: input cloud lacks ground truth information.")

        # Save numpy arrays into a dictionary and then to a .pth file
        data = {
            'coord': coord,
            'color': color,
            'scene_id': output_pth_path.with_suffix(''),
            'normal': normal,
            'gt': gt,
        }
        
        # Save the dictionary as a .pth file
        torch.save(data, output_pth_path)
        print(f"- saved {max_points} points to {output_pth_path}")
    return


###############################################################################
# PDAL pipelines
###############################################################################
def run_pdal_single_category(label, category, resolution):
    gt_index = int(category.split('_')[0])
    pipeline_json = {
        "pipeline": [
            # Read the .las
            {
                "type": "readers.las",
                "filename": f"raw_clouds/{label}/{category}.las",
                "extra_dims": EXTRA_DIMS
            },
            # Subsample at requested resolution. Cloud is still in cm so we mult by 100.
            {
                "type": "filters.voxelcenternearestneighbor",
                "cell": resolution * 100
            },
            # Scale down
            {
                "type":"filters.transformation",
                "matrix":"0.01  0  0  0  0  0.01  0  0  0  0  0.01  0  0  0  0  1"
            },
            # Switch Z and Y axes
            {
                "type": "filters.transformation",
                "matrix": "1 0 0 0 0 0 -1 0 0 1 0 0 0 0 0 1"
            },
            # Add new scalar field called gt
            {
                "type": "filters.ferry",
                "dimensions": "=>gt"
            },
            # Uniformly assign the ground truth category to gt
            {
                "type": "filters.assign",
                "assignment": f"gt[:]={gt_index}"
            },
            # Write the output
            {
                "type": "writers.las",
                "filename": f"processed_clouds/resolution_{resolution}/{label}/{category}.las",
                "forward": "all",
                "extra_dims": "all",
                "minor_version": 4,
                "dataformat_id": 8
            }
        ]
        
    }
    pipeline = pdal.Pipeline(json.dumps(pipeline_json))
    print(f" - processing category: {category}")
    count = pipeline.execute()
    print(f"   - total points processed: {count}")   


def run_all_categories(label, resolution):
    for category in CATEGORIES:
        # First check the category file exists, some of the sets are nonexhaustive
        fname = f"raw_clouds/{label}/{category}.las"
        if not Path(fname).exists():
            continue
        run_pdal_single_category(label, category, resolution)


def run_merger_pipeline(config):
    label = config.label
    resolution = config.resolution
    input_files = []
    for category in CATEGORIES:       
        fname = f"processed_clouds/resolution_{resolution}/{label}/{category}.las"
        if Path(fname).exists():
            input_files.append(fname)
    
    pl = []
    for f in input_files:
        pl.append(
            {
                "type": "readers.las",
                "filename": f,
                "extra_dims": EXTRA_DIMS
            }
        )
    pl.append(
        {
           "type": "filters.merge" 
        }
    )
    pl.append(
        {
            "type": "writers.las",
            "filename": config.merged_filename,
            "forward": "all",
            "extra_dims": "all",
            "minor_version": 4,
            "dataformat_id": 8        
        }
    )
    pipeline_json = {
        "pipeline": pl
    }

    pipeline = pdal.Pipeline(json.dumps(pipeline_json))
    print("Merging categories...")
    count = pipeline.execute()
    print(f"Total points processed: {count}")
    print(f"Created merged file: {config.merged_filename}")
    print(f"Removing intermediate split category .las files.")
    for f in input_files:
        os.remove(f)


def run_scene_chipper_pipeline(config):
    """Run the initial chipper to create test/train/val scene splits."""
    pipeline_json = {
        "pipeline": [
            {
                "type": "readers.las",
                "filename": config.merged_filename,
                "extra_dims": EXTRA_DIMS
            },
            {
                "type": "filters.chipper",
                "capacity": config.scene_capacity
            },
            {
                "type": "writers.las",
                "filename": config.scene_template,
                "forward": "all",
                "extra_dims": "all",
                "minor_version": 4,
                "dataformat_id": 8
            }   
        ]
    }
    pipeline = pdal.Pipeline(json.dumps(pipeline_json))
    print(f"Running train/test/val scene chipper with capacity={config.scene_capacity}...")
    count = pipeline.execute()

    # Count number of outputs
    file_count = 0
    for root, dirs, files in os.walk(config.scene_dir):
        file_count += len(files)
    print(f" - chipper created {file_count} splits.")


def run_file_chipper_pipeline(config):
    """Run the chipper to split each scene split into manageable filesizes for Pointcept."""
    base_dir = config.scene_dir
    base_dir = Path(base_dir)
    f_template = base_dir / "scene*.las"
    scene_files = glob.glob(str(f_template))

    for f in scene_files:

        path = Path(f)
        template = config.split_dir + f"{path.stem}_split#.las"
        pipeline_json = {
            "pipeline": [
                # Read the .las
                {
                    "type": "readers.las",
                    "filename": f,
                    "extra_dims": EXTRA_DIMS
                },
                {
                    "type": "filters.chipper",
                    "capacity": config.file_capacity
                },
                {
                    "type": "writers.las",
                    "filename": str(template),
                    "forward": "all",
                    "extra_dims": "all",
                    "minor_version": 4,
                    "dataformat_id": 8
                }   
            ]
        }
        pipeline = pdal.Pipeline(json.dumps(pipeline_json))
        print(f" - running {path.stem} chipper with capacity={file_capacity}...")
        count = pipeline.execute()
        file_count = 0

    # Walk through all directories and files in directory_path
    for root, dirs, files in os.walk(config.split_dir):
        file_count += len(files)  # Add count of files in this directory
    print(f" - chipper created {file_count} splits across all scenes.")


def scene_files_exist(config):
    # Replace the # char in the pdal template with * for glob
    template = config.scene_template.replace('#', '*')
    scene_files = glob.glob(template)
    return True if scene_files else False


###############################################################################
# Arg parser and main script
###############################################################################
def parse_arguments():
    parser = argparse.ArgumentParser(description="Process point cloud data using PDAL.")
    parser.add_argument("dataset", type=str, help="The dataset tag to process, e.g., 'park_row'")
    parser.add_argument("resolution", type=float, help="The resolution to use in subsampling.")
    parser.add_argument(
        "--file-capacity", type=int, default=500000,
        help="The point capacity for each file used as input to Pointcept, default is 500k."
        )
    parser.add_argument(
        "--scene-capacity", type=int, default=2400000,
        help="The point capacity for each file train/test/val scene, default is 2.4 million."
    )
    parser.add_argument(
        '--merger-only', action='store_true', default=False,
        help='Enable merger only mode (default: %(default)s)'
    )
    return parser.parse_args()


if __name__ == '__main__':

    # Retrieve args
    args = parse_arguments()
    label = args.dataset
    if label not in VALID_LABELS:
        raise ValueError(f"Unknown dataset label: '{label}'.")

    resolution = args.resolution
    if not isinstance(resolution, float):
        raise ValueError(f"Provided resolution {resolution} is not a float.")
    if resolution < 0.005:
        raise ValueError(f"The requested resolution {resolution} is too fine for the pre-sampling.")

    scene_capacity = args.scene_capacity
    if not isinstance(scene_capacity, int):
        raise ValueError(f"Provided scene_capacity: {scene_capacity} must be an integer.")

    file_capacity = args.file_capacity
    if not isinstance(file_capacity, int):
        raise ValueError(f"Provided file_capacity: {file_capacity} must be an integer.")
    
    # Generate a config struct to be passed as arg.
    MyConfig = namedtuple("MyConfig", [
        "label", "resolution", "scene_capacity", "file_capacity",
        "base_dir", "merged_filename", "scene_dir", "scene_template", "split_dir", "split_template",
        "pth_dir"
    ])
    scene_dir = f"processed_clouds/resolution_{resolution}/{label}/scene_capacity_{scene_capacity}/"
    config = MyConfig(
        label = label, resolution=resolution, scene_capacity=scene_capacity, file_capacity=file_capacity,
        base_dir = f"processed_clouds/resolution_{resolution}/{label}",
        merged_filename = f"processed_clouds/resolution_{resolution}/{label}/{label}_merged.las",
        scene_dir = scene_dir,
        scene_template = scene_dir + "scene#.las",  # for PDAL
        split_dir = scene_dir + f'file_capacity_{file_capacity}/',
        split_template = scene_dir + f'file_capacity_{file_capacity}/scene*_split*.las',  # for glob
        pth_dir = scene_dir + f'file_capacity_{file_capacity}/pth/',
    )

    # Print the config to the terminal with formatting for alignment
    max_field_length = max(len(field) for field in MyConfig._fields) + 1
    print("\033[1mConfiguration Details:\033[0m")
    print("-" * 30)
    for field in MyConfig._fields:
        field_name_formatted = field.replace('_', ' ').title().ljust(max_field_length)
        print(f"\033[1m{field_name_formatted}:\033[0m \033[94m{getattr(config, field)}\033[0m")
    print("-" * 30)

    # Ensure the relevant dirs
    os.makedirs(config.scene_dir, exist_ok=True)
    os.makedirs(config.pth_dir, exist_ok=True)

    # If necessary, run sampling correction and merging.
    if not Path(config.merged_filename).exists():
        print("Subsampling individual category clouds.")
        run_all_categories(label, resolution)
        run_merger_pipeline(config)
    else:
        print(f"Merged .las output already exists for {label} at this resolution.")
    if args.merger_only:
        print("Merger only mode enabled, exiting now.")
        sys.exit()

    # If necessary, run the scene splitter
    if not scene_files_exist(config):
        run_scene_chipper_pipeline(config)
    else:
        print("Scene files exist for this scene capacity. Skipping to file chipper...")

    # Run the file splitter and .pth conversion
    run_file_chipper_pipeline(config)
    convert_all_las_to_pth(config)

    print("Data pipeline complete!")
    print("-" * 30)
