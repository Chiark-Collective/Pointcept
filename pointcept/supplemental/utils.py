import os
import vtk
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




def read_ply_mesh(file_path, compute_normals=True):
    """
    Reads a .ply mesh from a file path and optionally computes normals.

    Args:
        file_path (str): Path to the .ply file.
        compute_normals (bool): If True, computes normals for the mesh. Default is True.

    Returns:
        vtk.vtkPolyData: A VTK object representing the loaded mesh, optionally with normals computed.
    """
    # Read the PLY file
    reader = vtk.vtkPLYReader()
    reader.SetFileName(file_path)
    reader.Update()

    # Get the output as vtkPolyData
    polyData = reader.GetOutput()

    # Optionally compute normals for the mesh
    if compute_normals:
        normals_filter = vtk.vtkPolyDataNormals()
        normals_filter.SetInputData(polyData)
        normals_filter.ComputePointNormalsOn()
        normals_filter.Update()
        return normals_filter.GetOutput()

    # Return the vtkPolyData object without normals
    return polyData

def render_vtk_mesh(mesh, window_name="VTK Mesh Viewer", background_color=(0.1, 0.2, 0.4)):
    """
    Renders a vtkPolyData mesh using VTK in a Jupyter notebook.

    Args:
        mesh (vtk.vtkPolyData): The vtkPolyData mesh to render.
        window_name (str): The title of the render window.
        background_color (tuple): Background color in RGB format.
    """
    # Create a mapper and actor for the mesh
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputData(mesh)

    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetColor(1.0, 1.0, 1.0)  # Set mesh color to white for better visibility
    actor.GetProperty().SetEdgeVisibility(1)  # Optional: show edges
    actor.GetProperty().SetLineWidth(1.0)  # Optional: line width for edges

    # Create a renderer and render window
    renderer = vtk.vtkRenderer()
    render_window = vtk.vtkRenderWindow()
    render_window.AddRenderer(renderer)
    render_window.SetSize(800, 600)  # Set window size
    render_window.SetWindowName(window_name)

    # Set the background color
    renderer.SetBackground(*background_color)

    # Add actor to the renderer
    renderer.AddActor(actor)

    # Create a render window interactor
    render_window_interactor = vtk.vtkRenderWindowInteractor()
    render_window_interactor.SetRenderWindow(render_window)

    # Start the rendering loop
    render_window.Render()
    render_window_interactor.Start()


def read_las_file(file_path):
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


