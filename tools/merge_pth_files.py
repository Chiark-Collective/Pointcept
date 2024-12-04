#!/usr/bin/env python3
"""
merge_pth_files.py

A command-line utility to merge multiple combined*.pth files from train, test, and eval directories
for specified site labels into unified .pth files stored in newly created 'combined' directories.

Usage:
    python merge_pth_files.py --input_dir /path/to/main_dir --labels queens_house park_row

Arguments:
    --input_dir: Path to the main directory containing site label subdirectories (e.g., 'park_row', 'queens_house').
    --labels: List of site labels to process (e.g., queens_house, park_row).

Example:
    python merge_pth_files.py --input_dir data/clouds/res0.02_pr0.05 --labels queens_house park_row
"""

import argparse
import torch
import numpy as np
import os
import random
from pathlib import Path
from typing import List

def parse_arguments():
    """
    Parse command-line arguments.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="Merge combined*.pth files from train, test, and eval directories for specified site labels into unified .pth files."
    )
    parser.add_argument(
        '--input_dir',
        type=str,
        required=True,
        help='Path to the main directory containing site label subdirectories (e.g., "park_row", "queens_house").'
    )
    parser.add_argument(
        '--labels',
        type=str,
        nargs='+',
        required=True,
        help='List of site labels to process (e.g., queens_house, park_row).'
    )
    return parser.parse_args()

def generate_random_scene_id():
    """
    Generate a new random integer to override the scene_id.

    Returns:
        int: A random integer between 1,000,000 and 9,999,999.
    """
    return random.randint(1_000_000, 9_999_999)

def merge_combined_pth_custom(label_dir: Path, output_path: Path):
    """
    Merge all combined*.pth files from train, test, eval directories into a unified .pth file.

    Args:
        label_dir (Path): Path to the site label directory containing train/test/eval subdirectories.
        output_path (Path): Path where the unified .pth file will be saved.
    """
    # Define the subdirectories to search
    subdirs = ['train', 'test', 'eval']
    
    # Initialize lists to hold data from all files
    merged_coord: List[np.ndarray] = []
    merged_color: List[np.ndarray] = []
    merged_normal: List[np.ndarray] = []
    merged_gt: List[np.ndarray] = []
    
    for subdir in subdirs:
        dir_path = label_dir / subdir
        if not dir_path.exists():
            print(f"Warning: Subdirectory '{subdir}' does not exist under '{label_dir}'. Skipping.")
            continue

        # Find all combined*.pth files in the subdirectory
        pth_files = list(dir_path.glob('combined*.pth'))
        if not pth_files:
            print(f"No combined*.pth files found in '{dir_path}'.")
            continue

        for pth_file in pth_files:
            print(f"Loading '{pth_file}'...")
            try:
                data = torch.load(pth_file, map_location='cpu')
                
                if not isinstance(data, dict):
                    print(f"Warning: '{pth_file}' does not contain a dictionary. Skipping.")
                    continue
                
                # Extract and append each key's data
                coord = data.get('coord')
                color = data.get('color')
                normal = data.get('normal')
                gt = data.get('gt')
                
                if coord is not None and isinstance(coord, np.ndarray):
                    merged_coord.append(coord)
                else:
                    print(f"Warning: 'coord' missing or not a numpy.ndarray in '{pth_file}'.")
                
                if color is not None and isinstance(color, np.ndarray):
                    merged_color.append(color)
                else:
                    print(f"Warning: 'color' missing or not a numpy.ndarray in '{pth_file}'.")
                
                if normal is not None and isinstance(normal, np.ndarray):
                    merged_normal.append(normal)
                else:
                    print(f"Warning: 'normal' missing or not a numpy.ndarray in '{pth_file}'.")
                
                if gt is not None and isinstance(gt, np.ndarray):
                    merged_gt.append(gt)
                else:
                    print(f"Warning: 'gt' missing or not a numpy.ndarray in '{pth_file}'.")
            
            except Exception as e:
                print(f"An error occurred while loading '{pth_file}': {e}")
                continue

    # Concatenate all numpy arrays along the first axis
    def concatenate_arrays(array_list: List[np.ndarray], key_name: str):
        if not array_list:
            print(f"Warning: No data found for '{key_name}'.")
            return None
        try:
            return np.concatenate(array_list, axis=0)
        except Exception as e:
            print(f"Error concatenating '{key_name}': {e}")
            return None

    unified_data = {}
    unified_data['coord'] = concatenate_arrays(merged_coord, 'coord')
    unified_data['color'] = concatenate_arrays(merged_color, 'color')
    unified_data['normal'] = concatenate_arrays(merged_normal, 'normal')
    unified_data['gt'] = concatenate_arrays(merged_gt, 'gt')
    
    # Override 'scene_id' with a new random integer
    new_scene_id = generate_random_scene_id()
    unified_data['scene_id'] = new_scene_id
    print(f"Overriding 'scene_id' with new random integer: {new_scene_id}")

    # Save the merged data
    try:
        torch.save(unified_data, output_path)
        print(f"Unified .pth file saved to '{output_path}'.\n")
    except Exception as e:
        print(f"Failed to save the unified .pth file: {e}")

def main():
    # Parse command-line arguments
    args = parse_arguments()
    input_dir = Path(args.input_dir).resolve()
    labels = args.labels

    # Verify input directory exists
    if not input_dir.exists():
        print(f"Error: The input directory '{input_dir}' does not exist.")
        exit(1)
    
    # Define the combined directory path
    combined_dir = input_dir / 'combined'
    try:
        combined_dir.mkdir(exist_ok=True)
        print(f"Combined directory: '{combined_dir}'\n")
    except Exception as e:
        print(f"Failed to create combined directory '{combined_dir}': {e}")
        exit(1)

    # Process each label
    for label in labels:
        print(f"Processing label: '{label}'")
        label_dir = input_dir / label
        if not label_dir.exists():
            print(f"Error: Label directory '{label_dir}' does not exist. Skipping.")
            continue
        
        # Define the output file path
        output_filename = f"combined_{label}_unified.pth"
        output_pth = combined_dir / output_filename

        # Merge the .pth files for this label
        merge_combined_pth_custom(label_dir, output_pth)

    print("All specified labels have been processed.")

if __name__ == '__main__':
    main()
