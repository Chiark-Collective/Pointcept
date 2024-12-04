#!/usr/bin/env python3
"""
chunk_pth_files.py

A command-line utility to chunk a unified .pth file into spatially contiguous regions based on x and y bins.
Each chunk spans the entire z-axis and is saved as a separate .pth file with an updated scene_id.

Usage:
    python chunk_pth_files.py --input_file /path/to/combined_queens_house_unified.pth --x_bins 4 --y_bins 4

Arguments:
    --input_file: Path to the unified .pth file to be chunked.
    --x_bins: Number of bins along the x-axis.
    --y_bins: Number of bins along the y-axis.

Example:
    python chunk_pth_files.py --input_file data/clouds/res0.02_pr0.05/combined/combined_queens_house_unified.pth --x_bins 4 --y_bins 4
"""

import argparse
import torch
import numpy as np
import os
import random
from pathlib import Path
from typing import List, Tuple

def parse_arguments():
    """
    Parse command-line arguments.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="Chunk a unified .pth file into spatially contiguous regions based on x and y bins."
    )
    parser.add_argument(
        '--input_file',
        type=str,
        required=True,
        help='Path to the unified .pth file to be chunked.'
    )
    parser.add_argument(
        '--x_bins',
        type=int,
        default=4,
        help='Number of bins along the x-axis. Default is 4.'
    )
    parser.add_argument(
        '--y_bins',
        type=int,
        default=4,
        help='Number of bins along the y-axis. Default is 4.'
    )
    return parser.parse_args()

def generate_random_scene_id() -> int:
    """
    Generate a new random integer to override the scene_id.

    Returns:
        int: A random integer between 1,000,000 and 9,999,999.
    """
    return random.randint(1_000_000, 9_999_999)

def define_bins(coordinates: np.ndarray, num_bins: int) -> Tuple[np.ndarray, float]:
    """
    Define bin edges for a given set of coordinates and number of bins.

    Args:
        coordinates (np.ndarray): Array of coordinate values (either x or y).
        num_bins (int): Number of bins to create.

    Returns:
        Tuple[np.ndarray, float]: Bin edges and bin widths.
    """
    min_val = coordinates.min()
    max_val = coordinates.max()
    bins = np.linspace(min_val, max_val, num_bins + 1)
    bin_width = bins[1] - bins[0]
    return bins, bin_width

def chunk_data(
    coord: np.ndarray,
    color: np.ndarray,
    normal: np.ndarray,
    gt: np.ndarray,
    x_bins: int,
    y_bins: int
) -> List[dict]:
    """
    Chunk the data into spatially contiguous regions based on x and y bins.

    Args:
        coord (np.ndarray): Array of coordinates with shape (N, 3).
        color (np.ndarray): Array of colors with shape (N, ...).
        normal (np.ndarray): Array of normals with shape (N, ...).
        gt (np.ndarray): Array of ground truth labels with shape (N, ...).
        x_bins (int): Number of bins along the x-axis.
        y_bins (int): Number of bins along the y-axis.

    Returns:
        List[dict]: List of dictionaries, each representing a chunk.
    """
    # Extract x and y coordinates
    x = coord[:, 0]
    y = coord[:, 1]

    # Define bin edges
    x_edges, x_width = define_bins(x, x_bins)
    y_edges, y_width = define_bins(y, y_bins)

    chunks = []

    for i in range(x_bins):
        for j in range(y_bins):
            # Define the current bin range
            x_min = x_edges[i]
            x_max = x_edges[i + 1]
            y_min = y_edges[j]
            y_max = y_edges[j + 1]

            # Find indices of points within the current bin
            if i < x_bins -1:
                in_x = (x >= x_min) & (x < x_max)
            else:
                in_x = (x >= x_min) & (x <= x_max)  # Include the max value in the last bin

            if j < y_bins -1:
                in_y = (y >= y_min) & (y < y_max)
            else:
                in_y = (y >= y_min) & (y <= y_max)  # Include the max value in the last bin

            indices = np.where(in_x & in_y)[0]

            if len(indices) == 0:
                print(f"Warning: No data points found in bin x:{i+1}/{x_bins} y:{j+1}/{y_bins}. Skipping this chunk.")
                continue

            # Extract data for the current chunk
            chunk_coord = coord[indices]
            chunk_color = color[indices]
            chunk_normal = normal[indices]
            chunk_gt = gt[indices]

            # Override scene_id
            new_scene_id = generate_random_scene_id()

            chunk = {
                'coord': chunk_coord,
                'color': chunk_color,
                'normal': chunk_normal,
                'gt': chunk_gt,
                'scene_id': new_scene_id
            }

            chunks.append({
                'chunk_id': f"x{i+1}_y{j+1}",
                'data': chunk
            })

    return chunks

def save_chunks(chunks: List[dict], output_dir: Path, label: str):
    """
    Save each chunk as a separate .pth file.

    Args:
        chunks (List[dict]): List of chunk dictionaries.
        output_dir (Path): Directory where chunked .pth files will be saved.
        label (str): The label name, used in the output filenames.
    """
    for chunk in chunks:
        chunk_id = chunk['chunk_id']
        data = chunk['data']
        output_filename = f"combined_{label}_chunk_{chunk_id}_unified.pth"
        output_path = output_dir / output_filename

        try:
            torch.save(data, output_path)
            print(f"Saved chunk '{chunk_id}' to '{output_path}'.")
        except Exception as e:
            print(f"Failed to save chunk '{chunk_id}' to '{output_path}': {e}")

def main():
    # Parse command-line arguments
    args = parse_arguments()
    input_file = Path(args.input_file).resolve()
    x_bins = args.x_bins
    y_bins = args.y_bins

    # Validate input file
    if not input_file.exists():
        print(f"Error: The input file '{input_file}' does not exist.")
        exit(1)
    
    # Define the output directory as the same as input file's directory
    output_dir = input_file.parent / 'chunks'
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"Output directory set to '{output_dir}'.\n")
    except Exception as e:
        print(f"Error: Failed to create output directory '{output_dir}': {e}")
        exit(1)

    # Load unified .pth file
    try:
        data = torch.load(input_file, map_location='cpu')
        print(f"Successfully loaded '{input_file}'.")
    except Exception as e:
        print(f"Error: Failed to load '{input_file}': {e}")
        exit(1)

    # Extract data components
    coord = data.get('coord')
    color = data.get('color')
    normal = data.get('normal')
    gt = data.get('gt')

    # Validate data components
    if coord is None or color is None or normal is None or gt is None:
        print("Error: One or more required keys ('coord', 'color', 'normal', 'gt') are missing in the .pth file.")
        exit(1)
    
    # Ensure coord has shape (N, 3)
    if coord.ndim != 2 or coord.shape[1] != 3:
        print(f"Error: 'coord' array must have shape (N, 3). Current shape: {coord.shape}")
        exit(1)

    # Extract label from filename (assuming format: combined_<label>_unified.pth)
    filename = input_file.stem  # e.g., 'combined_queens_house_unified'
    parts = filename.split('_')
    if len(parts) < 3:
        print("Error: Unexpected filename format. Expected format: combined_<label>_unified.pth")
        exit(1)
    label = '_'.join(parts[1:-1])  # Handles labels with underscores, e.g., 'park_row'

    print(f"Processing label: '{label}' with {x_bins} x-bins and {y_bins} y-bins.\n")

    # Chunk the data
    chunks = chunk_data(coord, color, normal, gt, x_bins, y_bins)

    if not chunks:
        print("No chunks were created. Exiting.")
        exit(0)

    # Save the chunks
    save_chunks(chunks, output_dir, label)

    print("\nAll chunks have been processed and saved.")

if __name__ == '__main__':
    main()
