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
import os
import re
import shutil
import pprint

from pathlib import Path

# Constants
VALID_LABELS = [
    'park_row',
    'rog_north',
]
EXTRA_DIMS = "NormalX=float64, NormalY=float64, NormalZ=float64"

categories = [
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
                "matrix": "1 0 0 0 0 0 1 0 0 1 0 0 0 0 0 1"
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
    for category in categories:

        # First check the category file exists, some of the sets are nonexhaustive
        fname = f"raw_clouds/{label}/{category}.las"
        if not Path(fname).exists():
            continue

        run_pdal_single_category(label, category, resolution)


def run_merger_pipeline(label, resolution):
    input_files = []
    for category in categories:       
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
            "filename": get_merged_filename(resolution, label),
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
    print(f"Created merged file: {output_filename}")
    print(f"Removing intermediate split category .las files.")
    for f in input_files:
        os.remove(f)


def run_chipper_pipeline(label, resolution, capacity):
    pipeline_json = {
        "pipeline": [
            # Read the .las
            {
                "type": "readers.las",
                "filename": get_merged_filename(resolution, label),
                "extra_dims": EXTRA_DIMS
            },
            {
                "type": "filters.chipper",
                "capacity": capacity
            },
            {
                "type": "writers.las",
                "filename": f"processed_clouds/resolution_{resolution}/{label}/chipper_{capacity}/{label}_split#.las",
                "forward": "all",
                "extra_dims": "all",
                "minor_version": 4,
                "dataformat_id": 8
            }   
        ]
    }
    pipeline = pdal.Pipeline(json.dumps(pipeline_json))
    print(f"Running chipper with capacity={capacity}...")
    count = pipeline.execute()
    file_count = 0
    # Walk through all directories and files in directory_path
    chipper_dir = f"processed_clouds/resolution_{resolution}/{label}/chipper_{capacity}"
    for root, dirs, files in os.walk(chipper_dir):
        file_count += len(files)  # Add count of files in this directory
    print(f" - chipper created {file_count} splits.")


def parse_arguments():
    parser = argparse.ArgumentParser(description="Process point cloud data using PDAL.")
    parser.add_argument("dataset", type=str, help="The dataset tag to process, e.g., 'park_row'")
    parser.add_argument("resolution", type=float, help="The resolution to use in subsampling.")
    parser.add_argument("--capacity", type=int, default=2400000, help="The capacity for each train/test split, default is 2.4 million.")
    return parser.parse_args()


def get_merged_filename(resolution, label):
    return f"processed_clouds/resolution_{resolution}/{label}/{label}_merged.las"


def merged_file_exists(resolution, label):
    f = get_merged_filename(resolution, label)
    if Path(f).exists():
        return True
    return False


if __name__ == '__main__':

    # Retrieve args
    args = parse_arguments()
    label = args.dataset
    if label not in VALID_LABELS:
        raise ValueError(f"Unknown dataset label: '{label}'.")

    resolution = args.resolution
    if not isinstance(resolution, float):
        raise ValueError(f"Provided resolution {resolution} is not a float.")
    if resolution < 0.01:
        raise ValueError(f"The requested resolution {resolution} is too fine for the pre-sampling.")

    capacity = args.capacity
    if not isinstance(capacity, int):
        raise ValueError(f"Provided capacity: {capacity} must be an integer.")

    print("\033[1mConfiguration Details:\033[0m")  # Bold header
    print("-" * 30)  # Separator
    print(f"\033[1mDataset Tag:\033[0m \033[94m{args.dataset}\033[0m")  # Bold key, blue value
    print(f"\033[1mResolution:\033[0m \033[94m{args.resolution}\033[0m")
    print(f"\033[1mCapacity:\033[0m \033[94m{args.capacity}\033[0m")
    print("-" * 30)  # Footer separator
    
    # Ensure the relevant dirs
    base_dir = f"processed_clouds/resolution_{resolution}/{label}"
    chipper_dir = f"processed_clouds/resolution_{resolution}/{label}/chipper_{capacity}"
    os.makedirs(base_dir, exist_ok=True)
    os.makedirs(chipper_dir, exist_ok=True)

    if not merged_file_exists(resolution, label):
        # Run the pipelines to create the processed individual clouds
        print("Subsampling individual category clouds.")
        run_all_categories(label, resolution)

        # Run the pipeline to merge the clouds
        run_merger_pipeline(label, resolution)
    else:
        print("Merged .las output already exists for this config. Skipping to chipper...")

    # Run the chipper pipeline
    run_chipper_pipeline(label, resolution, capacity)