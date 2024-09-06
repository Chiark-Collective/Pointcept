"""
Submodule to supplement Pointcept utils for heritage processing and development.
"""
import os
import laspy
import torch
import numpy as np
import pandas as pd
from dotenv import load_dotenv


_current_label = 'v1'
_category_dict = {
    'v1': [
        "1_WALL",
        "2_FLOOR",
        "3_ROOF",
        "4_CEILING",
        "5_FOOTPATH",
        "6_GRASS",
        "7_COLUMN",
        "8_DOOR",
        "9_WINDOW",
        "10_STAIR",
        "11_RAILING",
        "12_RWP",
        "13_OTHER",
        ],
}

def get_category_list(label=_current_label):
    return _category_dict[label]


def in_docker():
    """
    Check if the code is running inside a Docker container managed by the Pointcept project.

    Returns:
        bool: True if running inside Docker, False otherwise.
    """
    return os.getenv('INSIDE_POINTCEPT_DOCKER', 'false') == 'true'


def get_data_root2():
    """
    Loads environment variables, ensures 'DATA_ROOT' exists, and returns its path.
    Exits with an error if 'DATA_ROOT' is not set or the directory cannot be created.
    
    Returns:
        str: Path to the 'DATA_ROOT' directory.
    """
    load_dotenv()  # Ensure environment variables are loaded
    data_root = os.getenv('DATA_ROOT')
    if data_root is None:
        print("ERROR: DATA_ROOT environment variable not found.")
        exit(1)  # Exit if DATA_ROOT is not found
        # Make the DATA_ROOT directory if it doesn't exist
    try:
        os.makedirs(data_root, exist_ok=True)
        os.makedirs(data_root+'/results', exist_ok=True)
    except Exception as e:
        print(f"ERROR: Unable to create directory {data_root}. {e}")
        exit(1)
    return data_root

def ensure_data_root():
    data_root = './data'
    try:
        os.makedirs(data_root, exist_ok=True)
    except Exception as e:
        print(f"ERROR: Unable to create directory {data_root}. {e}")
        exit(1)
    return data_root

def ensure_category_dirs(category):
    """
    Creates necessary directories for a specified category within a fixed root data path.
    
    Args:
        category (str): The name of the category for which to create directories.

    Returns:
        str: The path to the newly created category directory, which includes a results subdirectory.

    Raises:
        Exception: If directory creation fails, prints an error message and exits the program.
    """
    data_root = './data'
    category_root = data_root + f'/{category}'
    try:
        os.makedirs(data_root, exist_ok=True)
        os.makedirs(category_root, exist_ok=True)
    except Exception as e:
        print(f"ERROR: Unable to create directory {data_root}. {e}")
        exit(1)
    return category_root

def print_dict_structure(data, indent=0):
    """
    Prints the structure of a dictionary, emphasizing the layout of nested dictionaries,
    numpy arrays, and PyTorch tensors. It details types, shapes, and data types, and includes a preview of elements.

    Parameters:
        data (dict): Dictionary to print, potentially containing complex data structures.
        indent (int): Indentation level for pretty printing, increasing with each dictionary level.

    Returns:
        None: Outputs directly to the console.
    """
    for key, value in data.items():
        print('    ' * indent + f'{key}: ', end='')
        if isinstance(value, dict):
            print()
            print_structure(value, indent+1)
        elif isinstance(value, np.ndarray):
            print(f'{type(value)}')
            print('    ' * (indent + 1) + f'Shape: {value.shape}')
            print('    ' * (indent + 1) + f'Dtype: {value.dtype}')
            # Improved presentation of first few elements respecting the shape
            preview_elements = value[:min(5, value.shape[0])] if value.ndim > 1 else value[:min(5, value.size)]
            print('    ' * (indent + 1) + 'First few elements:')
            for elem in preview_elements:
                print('    ' * (indent + 2) + str(elem))
        elif isinstance(value, torch.Tensor):
            print(f'{type(value)}')
            print('    ' * (indent + 1) + f'Shape: {value.shape}')
            print('    ' * (indent + 1) + f'Dtype: {value.dtype}')
            print('    ' * (indent + 1) + 'First few elements:')
            # Ensure tensor is on CPU for numpy conversion and print
            preview_tensor = value[:min(5, value.size(0))] if value.dim() > 1 else value[:min(5, value.numel())]
            if value.requires_grad:
                preview_tensor = preview_tensor.detach()
            preview_tensor = preview_tensor.cpu().numpy()  # Convert to numpy array for easier handling
            for elem in preview_tensor:
                print('    ' * (indent + 2) + str(elem))
        if isinstance(value, str):
            print(type(value), value)
        else:
            print(type(value))


def read_las_file(file_path):
    """
    Reads and prints key information from a LAS file, including metadata, header details, and a sample of point data.
    
    This function opens a LAS file, reads its header to extract file metadata such as the LAS version and point format,
    and then reads point data to provide a sample of coordinates, intensities, and classifications.
    
    Parameters:
        file_path (str): The path to the LAS file to be read.
    
    Returns:
        None: Outputs directly to the console.
    """
    with laspy.open(file_path) as file:
        # Get the header to access metadata
        header = file.header

        # Print file version and general header information
        print(f"LAS File Version: {header.version}")
        if hasattr(header, 'file_signature'):
            print(f"File Signature: {header.file_signature}")
        print(f"Point Format: {header.point_format}")
        print(f"Number of Point Records: {header.point_count}")
        print(f"Number of Points by Return: {header.number_of_points_by_return}")
        
        # Safely print bounding box if it exists
        if hasattr(header, 'bounds'):
            print(f"Bounding Box: {header.bounds}\n")
        else:
            print("Bounding Box: Not available\n")

        # Initial point format details
        points = next(file.chunk_iterator(1))
        print("Point Format Details:")
        print(f"  - Dimension names: {', '.join(points.point_format.dimension_names)}")
        print(f"  - Extra dimension names: {', '.join(points.point_format.extra_dimension_names)}")
        print(f"  - Point size in bytes: {points.point_size}\n")

        # Read the point records from the file
        las = file.read()

        # Accessing specific data dimensions
        points = las.points
        x_coordinates = las.x
        y_coordinates = las.y
        z_coordinates = las.z
        red_coordinates = las.red

        # Optionally, access other attributes like intensity, classification, etc.
        intensity = las.intensity
        classifications = las.classification

        # Print some basic information about the point data
        print("Sample Data Points:")
        print(f"  - Sample X coordinates: {x_coordinates[:10]}")  # Print first 10 for brevity
        print(f"  - Sample Y coordinates: {y_coordinates[:10]}")
        print(f"  - Sample Z coordinates: {z_coordinates[:10]}")
        print(f"  - Sample red values: {red_coordinates[:10]}")
        print(f"  - Sample intensity values: {intensity[:10]}")
        print(f"  - Sample classifications: {classifications[:10]}")

def read_las_file2(file_path):
    """
    Reads and prints key information from a LAS file, including metadata, header details, and a sample of point data,
    with additional handling for normals and ground truth scalar fields if present.
    
    Parameters:
        file_path (str): The path to the LAS file to be read.
    
    Returns:
        None: Outputs directly to the console.
    """
    with laspy.open(file_path) as file:
        header = file.header

        # Print file version and general header information
        print(f"LAS File Version: {header.version}")
        if hasattr(header, 'file_signature'):
            print(f"File Signature: {header.file_signature}")
        print(f"Point Format: {header.point_format}")
        print(f"Number of Point Records: {header.point_count}")
        print(f"Number of Points by Return: {header.number_of_points_by_return}")
        
        if hasattr(header, 'bounds'):
            print(f"Bounding Box: {header.bounds}\n")
        else:
            print("Bounding Box: Not available\n")

        # Print initial point format details
        points = next(file.chunk_iterator(1))
        print("Point Format Details:")
        print(f"  - Dimension names: {', '.join(points.point_format.dimension_names)}")
        print(f"  - Extra dimension names: {', '.join(points.point_format.extra_dimension_names)}")
        print(f"  - Point size in bytes: {points.point_size}\n")

        # Read the entire file
        las = file.read()

        # Accessing and printing specific data dimensions
        print("Sample Data Points:")
        sample_count = 10  # Define how many samples to show
        for name in ["x", "y", "z", "red", "intensity", "classification", "gt", "NormalX", "NormalY", "NormalZ"]:
            if hasattr(las, name):
                print(f"  - Sample {name.capitalize()} values: {getattr(las, name)[:sample_count]}")



if __name__ == "__main__":
    fp = '/data/sdd/training_v2.las'
    read_las_file(fp)
