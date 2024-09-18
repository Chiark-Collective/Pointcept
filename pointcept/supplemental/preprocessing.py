import subprocess
import shutil
import logging
import sys
import random
import re

import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
import pyvista as pv

from pathlib import Path

from pointcept.supplemental.utils import get_category_list
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering


# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', datefmt='%H:%M:%S')
logger = logging.getLogger(__name__)

CATEGORIES = get_category_list()


def merge_and_save_cells(dh, splits):
    """
    Merge cells for each fold-category permutation and save them to disk.

    Args:
        dh (DataHandler): An object with a split_dirs attribute, containing directories for 'train', 'test', 'eval'.
        splits (dict): A dictionary containing fold-category mappings of cells. 
                       Example: {'train': {'category1': [cell1, cell2, ...], 'category2': [...]}}

    Returns:
        None: The function saves the merged meshes directly to disk.
    """
    # Iterate over the splits dictionary (e.g., 'train', 'test', 'eval')
    for fold, categories in splits.items():
        # Get the appropriate directory for the fold from DataHandler
        fold_dir = dh.split_dirs.get(fold)
        cell_counter = 0
        # Iterate over each category in the fold
        for category, cells in categories.items():
            # Initialize an empty PolyData object to merge the cells
            combined_mesh = pv.PolyData()
            
            # Merge the cells
            for cell in cells:
                # mesh.set_active_scalars("RGB")
                combined_mesh = combined_mesh.merge(cell)
                cell_counter += 1

            combined_mesh.GetPointData().SetActiveNormals('Normals')

            output_file = fold_dir / f"{category.lower()}.ply"           
            # Save the merged mesh to disk using vtk for control over color storage
            writer = vtk.vtkPLYWriter()
            writer.SetFileName(output_file.as_posix())
            writer.SetInputData(combined_mesh)
            writer.SetColorModeToDefault()  # Ensure colors are written from the Scalars
            writer.SetArrayName('RGB')
            writer.Write()
        print(f"for fold {fold}, processed {cell_counter} cells")


def process_splits_pyvista(splits, cell_width, seed=None):
    """
    Process all splits (train, test, eval), arranging cells from all categories into a single spiral pattern
    while maintaining category information.

    Args:
        splits (dict): Dictionary with categories containing lists of cells for 'train', 'test', 'eval'.
        cell_width (float): The width of each cell.
        seed (int): Seed for random shuffling and transformations.

    Returns:
        dict: Dictionary with combined meshes for 'train', 'test', 'eval', maintaining category distinctions.
    """
    for split_name, categories in splits.items():
        all_cells = []
        for category, cells in categories.items():
            all_cells.extend(cells)
            # Apply the appropriate centering transform with conditional z offsets
            for cell in cells:
                recenter_library_cell(cell, category)
        if all_cells:
            transform_spiral_cells_pyvista(all_cells, cell_width, seed)


def recenter_library_cell(cell, category):
    """
    Recenter the mesh in x and y dimensions to the origin. Set the minimum z-coordinate to zero for all
    categories except '3_ROOF' and '4_CEILING', which receive a specific Gaussian z-offset.

    Args:
        mesh (open3d.geometry.TriangleMesh): Mesh to be recentered.
        category (str): The category of the mesh to determine z offset behavior.
    """

    # Compute the axis-aligned bounding box (AABB) of the mesh
    bounds = cell.bounds  # Returns (xmin, xmax, ymin, ymax, zmin, zmax)
    # Compute the center in X and Y directions and min in Z
    x_center = (bounds[0] + bounds[1]) / 2  # (xmin + xmax) / 2
    y_center = (bounds[2] + bounds[3]) / 2  # (ymin + ymax) / 2
    min_z = bounds[4]  # zmin
    
    # Conditional offsets for '3_ROOF' and '4_CEILING'
    if category == '3_ROOF':
        min_z -= np.random.normal(loc=4, scale=0.5)
    elif category == '4_CEILING':
        min_z -= np.random.normal(loc=3, scale=0.5)
    
    cell.translate([-x_center, -y_center, -min_z], inplace=True)


def transform_spiral_cells_pyvista(cells, cell_width, seed=12525352):
    """
    Apply a transformation to arrange the cells in a spiral pattern.
    Apply the spiral transformation to all cells collectively.

    Args:
        cells (list): List of PyVista PolyData objects.
        cell_width (float): The width of each cell.
        seed (int, optional): Seed for random shuffling.

    Returns:
        None: The cells are transformed in place.
    """
    # Shuffle cells if a seed is provided
    random.seed(seed)
    random.shuffle(cells)

    # Generate spiral positions for arranging cells
    positions = generate_spiral_positions(len(cells))

    # Apply transformations directly to the original meshes
    for cell, (dx, dy) in zip(cells, positions):
          
        # Calculate the new translation vector
        translation = np.array([dx * cell_width, dy * cell_width, 0])
        
        # Translate the cell in place
        cell.translate(translation, inplace=True)


def divide_all_categories_into_cells_pyvista(meshes_dict, cell_width):
    """
    Divides each category's mesh into grid cells.

    Args:
        meshes_dict (dict): Dictionary of category meshes.
        cell_width (float): The desired width for each cell.

    Returns:
        dict: A dictionary with each category mapping to a list of its cell meshes.
    """
    category_cells = {}
    for category, mesh in meshes_dict.items():
        category_cells[category] = split_mesh_into_cells_pyvista(mesh, cell_width)
    return category_cells


def split_mesh_into_cells_pyvista(mesh, cell_width):
    """
    Splits a mesh into smaller cells based on the specified cell width.

    Args:
        mesh (vtk.vtkPolyData): The mesh to be split (wrapped with PyVista).
        cell_width (float): The width of each cell.

    Returns:
        list: A list of PyVista PolyData objects representing the cells.
    """
    # Wrap the VTK mesh with PyVista
    pv_mesh = pv.wrap(mesh)

    # Compute the axis-aligned bounding box (AABB) of the mesh
    bounds = pv_mesh.bounds  # Returns (xmin, xmax, ymin, ymax, zmin, zmax)
    min_bound = np.array([bounds[0], bounds[2], bounds[4]])
    max_bound = np.array([bounds[1], bounds[3], bounds[5]])

    # Determine the number of cells in each dimension
    num_cells_x = int(np.ceil((max_bound[0] - min_bound[0]) / cell_width))
    num_cells_y = int(np.ceil((max_bound[1] - min_bound[1]) / cell_width))

    cells = []

    # Generate bounding boxes for each cell and perform sequential clipping
    for i in range(num_cells_x):
        for j in range(num_cells_y):
            cell_min = np.array([min_bound[0] + i * cell_width, min_bound[1] + j * cell_width, min_bound[2]])
            cell_max = np.array([min_bound[0] + (i + 1) * cell_width, min_bound[1] + (j + 1) * cell_width, max_bound[2]])

            # Create six planes for the AABB
            planes = [
                ('x', cell_min[0], False),  # xmin plane
                ('x', cell_max[0], True), # xmax plane
                ('y', cell_min[1], False),  # ymin plane
                ('y', cell_max[1], True), # ymax plane
            ]

            # Start with the original mesh and sequentially clip with each plane
            clipped_mesh = pv_mesh
            for axis, origin, invert in planes:
                clipped_mesh = clipped_mesh.clip(
                    normal=axis,
                    origin=[origin, 0, 0] if axis == 'x' else [0, origin, 0],
                    invert=invert
                )            
            # Add the resulting clipped cell mesh if it's not empty
            if clipped_mesh.n_points > 0:
                cells.append(clipped_mesh)

    return cells


def combine_category_meshes(splits):
    """
    Combines all meshes within each category for every split (train, test, eval).

    Args:
        splits (dict): Dictionary with keys 'train', 'test', 'eval', each containing a dict of category: list of meshes.

    Returns:
        dict: Dictionary with keys 'train', 'test', 'eval', each containing a dict of category: combined mesh.
    """
    combined_splits = {split_name: {} for split_name in splits}
    for split_name, categories in splits.items():
        for category, meshes in categories.items():
            combined_mesh = o3d.geometry.TriangleMesh()
            for mesh in meshes:
                combined_mesh += mesh
            combined_splits[split_name][category] = combined_mesh
            # logger.info(f"Combined {len(meshes)} meshes for {split_name}/{category}")
    return combined_splits

## Helper funcs to generate library splits.
def split_mesh_into_cells(mesh, cell_width):
    """
    Splits a mesh into smaller cells based on the specified cell width.

    Args:
        mesh (open3d.geometry.TriangleMesh): The mesh to be split.
        cell_width (float): The width of each cell.

    Returns:
        list: A list of TriangleMesh objects representing the cells.
    """
    cells = []
    aabb = mesh.get_axis_aligned_bounding_box()
    min_bound = aabb.get_min_bound()
    max_bound = aabb.get_max_bound()

    # Determine the number of cells in each dimension
    num_cells_x = int(np.ceil((max_bound[0] - min_bound[0]) / cell_width))
    num_cells_y = int(np.ceil((max_bound[1] - min_bound[1]) / cell_width))

    # Generate bounding boxes for each cell
    for i in range(num_cells_x):
        for j in range(num_cells_y):
            cell_min = np.array([min_bound[0] + i * cell_width, min_bound[1] + j * cell_width, min_bound[2]])
            cell_max = np.array([min_bound[0] + (i + 1) * cell_width, min_bound[1] + (j + 1) * cell_width, max_bound[2]])
            cell_aabb = o3d.geometry.AxisAlignedBoundingBox(min_bound=cell_min, max_bound=cell_max)
            cell_mesh = mesh.crop(cell_aabb)
            if not cell_mesh.is_empty():
                cells.append(cell_mesh)

    return cells

def divide_all_categories_into_cells(meshes_dict, cell_width):
    """
    Divides each category's mesh into grid cells.

    Args:
        meshes_dict (dict): Dictionary of category meshes.
        cell_width (float): The desired width for each cell.

    Returns:
        dict: A dictionary with each category mapping to a list of its cell meshes.
    """
    category_cells = {}
    for category, mesh in meshes_dict.items():
        category_cells[category] = split_mesh_into_cells(mesh, cell_width)
    return category_cells

def recenter_cell(mesh, category):
    """
    Recenter the mesh in x and y dimensions to the origin. Set the minimum z-coordinate to zero for all
    categories except '3_ROOF' and '4_CEILING', which receive a specific Gaussian z-offset.

    Args:
        mesh (open3d.geometry.TriangleMesh): Mesh to be recentered.
        category (str): The category of the mesh to determine z offset behavior.

    Returns:
        open3d.geometry.TriangleMesh: Recentered mesh with appropriate z offset.
    """
    aabb = mesh.get_axis_aligned_bounding_box()
    center = aabb.get_center()
    min_z = aabb.get_min_bound()[2]

    # Base translation to zero the mesh in z and center x, y
    base_translation = np.array([-center[0], -center[1], -min_z])
    mesh.translate(base_translation, relative=True)

    # Conditional offsets for '3_ROOF' and '4_CEILING'
    if category == '3_ROOF':
        additional_offset = np.random.normal(loc=6, scale=1)  # Gaussian centered at 6 with std deviation of 1
        mesh.translate([0, 0, additional_offset], relative=True)
    elif category == '4_CEILING':
        additional_offset = np.random.normal(loc=4, scale=1)  # Gaussian centered at 4 with std deviation of 1
        mesh.translate([0, 0, additional_offset], relative=True)

    return mesh

def transform_cells(cells_dict):
    """
    Transforms all cells in the dictionary by recentering them. The recentering includes 
    specific z-coordinate adjustments for categories '3_ROOF' and '4_CEILING'.

    Args:
        cells_dict (dict): Dictionary of categories, each with a list of TriangleMesh objects.

    Returns:
        dict: Dictionary with transformed cells for each category.
    """
    transformed_cells = {}
    for category, cells in cells_dict.items():
        transformed_cells[category] = [recenter_cell(cell, category) for cell in cells]
    return transformed_cells

def split_category_cells(cells, weights=(0.65, 0.2, 0.15), seed=None):
    """
    Splits a list of cells into train, test, and eval based on specified weights.

    Args:
        cells (list): List of TriangleMesh objects representing the cells.
        weights (tuple): Tuple of three elements representing the proportion for train, test, and eval.
        seed (int): Random seed for shuffling.

    Returns:
        dict: Dictionary containing lists of cells for 'train', 'test', 'eval'.
    """
    if seed is not None:
        random.seed(seed)
    random.shuffle(cells)

    total = len(cells)
    num_train = int(total * weights[0])
    num_test = int(total * weights[1])

    train_cells = cells[:num_train]
    test_cells = cells[num_train:num_train + num_test]
    eval_cells = cells[num_train + num_test:]

    return {'train': train_cells, 'test': test_cells, 'eval': eval_cells}

def split_all_categories(cells_dict, weights=(0.65, 0.2, 0.15), seed=None):
    """
    Splits cells from all categories into train, test, and eval splits based on weights.
    Organizes the splits by category.

    Args:
        cells_dict (dict): Dictionary of cells by category.
        weights (tuple): Weights for splitting into train, test, and eval.
        seed (int): Seed for random shuffling.

    Returns:
        dict: Nested dictionary with keys 'train', 'test', 'eval', each containing sub-dictionaries for each category with their respective cells.
    """
    splits = {'train': {}, 'test': {}, 'eval': {}}
    for category, cells in cells_dict.items():
        category_splits = split_category_cells(cells, weights, seed)
        splits['train'][category] = category_splits['train']
        splits['test'][category] = category_splits['test']
        splits['eval'][category] = category_splits['eval']

    return splits

def generate_spiral_positions(n):
    """
    Generate coordinates in a spiral order.

    Args:
        n (int): Number of positions to generate.

    Returns:
        list of tuples: List of (x, y) coordinates.
    """
    x, y = 0, 0
    dx, dy = 0, -1
    positions = []
    for _ in range(n):
        if (-n/2 < x <= n/2) and (-n/2 < y <= n/2):
            positions.append((x, y))
        if x == y or (x < 0 and x == -y) or (x > 0 and x == 1-y):
            dx, dy = -dy, dx
        x, y = x + dx, y + dy
    return positions

def recenter_and_transform_cells(cells, cell_width, seed, category_labels):
    """
    Recenter each cell to origin, then apply a transformation to arrange them in a spiral pattern.
    Apply the spiral transformation to all cells across categories collectively.

    Args:
        cells (list): List of TriangleMesh objects from all categories.
        cell_width (float): The width of each cell.
        seed (int): Seed for random shuffling.
        category_labels (list): List of category labels corresponding to each cell.

    Returns:
        open3d.geometry.TriangleMesh: The recombined mesh across all categories.
    """
    if seed is not None:
        random.seed(seed)
        combined_list = list(zip(cells, category_labels))
        random.shuffle(combined_list)
        cells, category_labels = zip(*combined_list)

    positions = generate_spiral_positions(len(cells))
    
    combined_mesh = o3d.geometry.TriangleMesh()
    for cell, (dx, dy), category in zip(cells, positions, category_labels):
        # Translate cells to new positions
        translation = np.array([dx * cell_width, dy * cell_width, 0])
        cell.translate(translation, relative=True)
        # Here, you might want to encode the category information within the mesh
        # For example, by assigning a scalar field or coloring based on 'category'
        combined_mesh += cell

    return combined_mesh


def process_splits(splits, cell_width, seed=None):
    """
    Process all splits (train, test, eval), arranging cells from all categories into a single spiral pattern
    while maintaining category information.

    Args:
        splits (dict): Dictionary with categories containing lists of cells for 'train', 'test', 'eval'.
        cell_width (float): The width of each cell.
        seed (int): Seed for random shuffling and transformations.

    Returns:
        dict: Dictionary with combined meshes for 'train', 'test', 'eval', maintaining category distinctions.
    """
    combined_meshes = {}
    for split_name, categories in splits.items():
        all_cells = []
        category_labels = []
        for category, cells in categories.items():
            all_cells.extend(cells)
            category_labels.extend([category] * len(cells))

        if all_cells:  # Ensure there are cells to process
            combined_meshes[split_name] = recenter_and_transform_cells(all_cells, cell_width, seed, category_labels)
        else:
            combined_meshes[split_name] = o3d.geometry.TriangleMesh()  # Empty mesh if no cells

    return combined_meshes

def plot_mesh_inline(mesh):
    """
    Visualize an Open3D mesh inline in a Jupyter Notebook using Matplotlib.

    Args:
        mesh (open3d.geometry.TriangleMesh): The mesh to visualize.
    """
    # Create a visualizer
    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=False)  # Create a non-visible window
    vis.add_geometry(mesh)

    # Set up the view control and update the camera parameters
    ctr = vis.get_view_control()
    cam_params = ctr.convert_to_pinhole_camera_parameters()
    
    # Camera setup
    cam_eye = np.array([0, -10, 10])  # Position
    cam_lookat = np.array([0, 0, 0])  # Look at
    cam_up = np.array([0, 0, 1])      # Up vector

    cam_params.extrinsic = np.linalg.inv(np.array([
        [1, 0, 0, -cam_eye[0]],
        [0, 1, 0, -cam_eye[1]],
        [0, 0, 1, -cam_eye[2]],
        [0, 0, 0, 1]
    ]))
    cam_params.intrinsic.set_intrinsics(800, 600, 1000, 1000, 400, 300)  # Adjust focal length and principal point
    ctr.convert_from_pinhole_camera_parameters(cam_params)
    ctr.set_lookat(cam_lookat)
    ctr.set_front(cam_eye - cam_lookat)
    ctr.set_up(cam_up)
    ctr.set_zoom(0.35)

    # Update the renderer
    vis.update_geometry(mesh)
    vis.poll_events()
    vis.update_renderer()

    # Capture the image
    image = vis.capture_screen_float_buffer(do_render=True)
    plt.figure(figsize=(9, 5))  # Small figure size
    plt.imshow(np.asarray(image))
    plt.axis('off')
    plt.show()

    vis.destroy_window()
