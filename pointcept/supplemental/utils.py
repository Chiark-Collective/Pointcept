"""
Submodule to supplement Pointcept utils for heritage processing and development.
"""
import laspy
import torch
import numpy as np
import pandas as pd


def print_dict_structure(data, indent=0):
    """This function takes a dict, like those loaded from a numpy or torch file,
    and prints its contents in a human-readable format. It specifically looks out for 
    numpy arrays and torch tensors."""
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
    # Open the LAS file
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


if __name__ == "__main__":
    fp = '/data/sdd/training_v2.las'
    read_las_file(fp)
