import logging
import shutil
import subprocess
import re
import psutil
import os
import gc

import vtk
import laspy
import torch
import open3d as o3d
import pyvista as pv
import numpy as np

from pathlib import Path

from vtk.util.numpy_support import vtk_to_numpy

from pointcept.supplemental.utils import get_category_list, read_ply_mesh
from pointcept.supplemental.library_funcs import *


# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', datefmt='%H:%M:%S')
logger = logging.getLogger(__name__)

CATEGORIES = get_category_list()

_data_root = None
def set_data_root(path):
    """
    Set the data root path for the module using a Path object.

    Args:
        path (str or Path): The filesystem path to be set as data root.
    """
    global _data_root
    if isinstance(path, str):
        _data_root = Path(path)
    elif isinstance(path, Path):
        _data_root = path
    else:
        raise TypeError("path must be a string or a Path object.")

def get_data_root():
    """
    Get the data root path for the module as a Path object.

    Returns:
        Path: The filesystem path used as the data root.
    """
    if _data_root is None:
        raise ValueError("Data root has not been set. Please set it using set_data_root().")
    return _data_root


#################################################################################
# DataHandler - knows where everything is
#################################################################################
class DataHandler:
    """
    Handles the organization and management of mesh data for processing.
    
    This class manages directories for storing and retrieving mesh data, ensuring
    proper structure and accessibility for operations such as extraction and analysis.

    Attributes:
        root_dir (Path): The root directory for mesh data storage.
        label (str): Label specifying a particular subset of data.
        cc_path (str): Path to the CloudCompare executable.
        mesh_dir (Path): Directory containing all mesh data.
        raw_mesh_dir (Path): Directory containing raw mesh files.
        extraction_dir (Path): Directory where extracted mesh data is stored.
        split_dir (Path): Directory for storing data splits.
        split_dirs (dict): Directories for train, test, and evaluation splits.
        raw_mesh_path (Path): Path to the raw mesh file specific to the label.
        raw_mesh_path_temp (Path): Temporary path for processing raw mesh data.
        _extracted_mesh_dict (dict): Dictionary storing extracted mesh data (private).
    """
    def __init__(self, label, cc_path="org.cloudcompare.CloudCompare"):
        """
        Initializes a new DataHandler instance with specific label and path settings.

        Args:
            label (str): A label to identify and categorize the mesh data being managed.
            cc_path (str, optional): Path to the CloudCompare executable used for
                                     handling mesh files. Defaults to "org.cloudcompare.CloudCompare".
        """
        self.root_dir = get_data_root()
        self.label = label
        self.cc_path = cc_path
        
        self.mesh_dir = self.root_dir / "meshes"
        self.raw_mesh_dir = self.mesh_dir / "raw"
        self.extraction_dir = self.mesh_dir / "extracted" / label
        self.transformed_dir = self.extraction_dir / 'transformed'

        self.split_dir = self.extraction_dir / "splits"
        self.split_dirs = {'train': self.extraction_dir / "train", 'test': self.extraction_dir / "test", 'eval': self.extraction_dir / "eval"}
        self.raw_mesh_path = self.raw_mesh_dir / f"{self.label}.bin"
        self.raw_mesh_path_temp = self.extraction_dir / f"{label}.bin"
        self._extracted_mesh_dict = {}

        self.clouds_root = self.root_dir / "clouds"

        # Ensure all necessary mesh directories are created
        for path in [self.mesh_dir, self.raw_mesh_dir, self.extraction_dir,
                     self.split_dir, self.clouds_root] + list(self.split_dirs.values()):
            path.mkdir(parents=True, exist_ok=True)

        self._ensure_split_dirs()

    def _ensure_split_dirs(self):
        for dir_path in self.split_dirs.values():
            dir_path.mkdir(parents=True, exist_ok=True)

    def set_bin_file(self, bin_file):
        """
        Only needs to be called if the .bin file is in a non-standard location outside the data_root.
        """
        try:
            if bin_file:
                self.raw_mesh_path = Path(bin_file)
                if not self.raw_mesh_path.exists():
                    raise FileNotFoundError(f"No file found at the specified path: {self.raw_mesh_path}")
                if not self.raw_mesh_path.is_file():
                    raise ValueError(f"The specified path is not a file: {self.raw_mesh_path}")
                logging.info(f"Input .bin specified and set: {self.raw_mesh_path}")
            else:
                raise ValueError("No bin file path provided.")
        except (FileNotFoundError, PermissionError, ValueError) as e:
            logging.error(f"Failed to set .bin file: {e}")
            raise

    def _verify_input_bin(self):
        try:
            assert self.raw_mesh_path.exists()
        except AssertionError:
            logging.error("Input .bin file for this config does not exist:")
            logging.error(f" {self.raw_mesh_path.as_posix()}")
            raise
    
    @property
    def extracted_meshes(self):
        if not self._extracted_mesh_dict: return {}
        return {category: data['mesh'] for category, data in self._extracted_mesh_dict.items()}

    def _label_has_extracted_meshes(self):
        files = list(self.transformed_dir.glob('*.ply'))
        return len(files) > 0   
    
    def extract_meshes(self):
        self._extract_meshes_from_bin()
        self._transform_meshes()
    
    def _prepare_mesh_extraction(self):
        """
        Prepare necessary dirs and copy raw .bin mesh file over to get
        around CC limitations.
        """
        self._verify_input_bin()
        if self.extraction_dir.exists():
            logger.info("Cleaning .bin extraction dirs.")
            shutil.rmtree(self.extraction_dir.as_posix())
        else:
            logger.info("Creating .bin extraction dirs.")
        self.extraction_dir.mkdir(parents=True, exist_ok=True)
        # Copy raw mesh file over.
        shutil.copy(self.raw_mesh_path.as_posix(), self.raw_mesh_path_temp.as_posix())    

    def _split_bin_by_category(self):
        logger.info(f"Splitting bin file {self.raw_mesh_path_temp} by category...")
        for category in CATEGORIES:
            command_regex = [
                self.cc_path,
                "-SILENT",
                "-O", self.raw_mesh_path_temp.name,
                "-SELECT_ENTITIES",
                "-REGEX", category,
                "-RENAME_ENTITIES", category.lower(),
                "-NO_TIMESTAMP", "-SAVE_MESHES"
            ]
    
            logger.info(f"  extracting category: {category}")
            try:
                result = subprocess.run(command_regex, cwd=self.extraction_dir, check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                # print("CloudCompare output:", result.stdout.decode())
                # print("CloudCompare errors:", result.stderr.decode())
            except subprocess.CalledProcessError as e:
                print(f"CloudCompare failed with error: {e.stderr.decode()}")
    
    def _convert_bin_to_ply(self):
        """
        Extract .ply files from the raw .bin files.
        """  
        # Cloudcompare appends an index suffix to renamed files when multiple entities are loaded
        # So we'll strip this suffix for deterministic file names going forward.
        # Regular expression to match filenames ending with '_<integer>.bin'        
        pattern = re.compile(r"^(.*?_\d+)\.(bin)$")
       
        for file_path in self.extraction_dir.iterdir():
            if file_path.is_file() and pattern.match(file_path.name):
                new_file_stem = re.sub(r'_\d+$', '', file_path.stem)
                new_file_path = file_path.with_name(f"{new_file_stem}{file_path.suffix}")
                file_path.rename(new_file_path)

                # Now convert the file to a .ply file for use outside of CloudCompare.
                command_convert = [
                    self.cc_path,
                    "-SILENT",
                    "-O", new_file_path.name,
                    "-M_EXPORT_FMT", "PLY",
                    "-NO_TIMESTAMP", "-SAVE_MESHES",
                ]
                try:
                    # Run the command as a subprocess
                    logger.info(f"  converting {new_file_path.name}")
                    result = subprocess.run(command_convert, cwd=self.extraction_dir, check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                except subprocess.CalledProcessError as e:
                    print(f"CloudCompare failed with error: {e.stderr.decode()}")

                # Finally unlink the intermediary .bin file
                new_file_path.unlink()

        # Unlink the original .bin file duplicate
        self.raw_mesh_path_temp.unlink()
        
        # Summarise created files
        logger.info("Converted .bin files to .ply files.")

    def _transform_meshes_open3d(self, recenter_origin=True):
        """
        Takes the extracted .ply meshes, swaps z-y axes while maintaining chirality, rescales from cm to m,
        and recenters all meshes so that the center of their combined AABB is at (0, 0, 0) and the lowest point in z is at 0.
        Saves the transformed meshes in a subdirectory 'transformed'.
        """
        # List of input .ply files with relative paths
        ply_files = [f for f in self.extraction_dir.iterdir() if f.suffix == '.ply']
        logger.info("Transforming meshes...")

        # Transformation matrix to switch Y and Z axes, reverse X axis, and scale by 0.01
        transform_matrix = np.array([
            [-0.01,  0,     0,    0],  # Reverse X axis and scale by 0.01
            [0,      0,  0.01,    0],  # Y becomes Z and scale by 0.01
            [0,   0.01,     0,    0],  # Z becomes Y and scale by 0.01
            [0,      0,     0,    1]   # Homogeneous coordinate
        ])

        # Initialize variables to compute the combined AABB and global minimum z value
        global_min_bound = np.array([np.inf, np.inf, np.inf])
        global_max_bound = np.array([-np.inf, -np.inf, -np.inf])
        global_min_z = np.inf  # Initialize the global minimum Z value

        # List to store loaded meshes
        meshes = []

        # Create 'transformed' subdirectory if it doesn't exist
        transformed_dir = self.transformed_dir
        transformed_dir.mkdir(exist_ok=True)

        # Step 1: Load meshes, apply initial transformations, and compute global bounds
        for ply_file in ply_files:
            mesh = o3d.io.read_triangle_mesh(ply_file.as_posix())

            # print("Before transformation:")
            # print(mesh)
            # print("Bounding Box:", mesh.get_axis_aligned_bounding_box())

            mesh.compute_vertex_normals()
            mesh.transform(transform_matrix)  # Apply initial transformation

            # Update global AABB and minimum Z value
            min_bound = np.asarray(mesh.get_min_bound())
            max_bound = np.asarray(mesh.get_max_bound())
            global_min_bound = np.minimum(global_min_bound, min_bound)
            global_max_bound = np.maximum(global_max_bound, max_bound)
            global_min_z = min(global_min_z, min_bound[2])  # Update the global minimum Z

            # Store the mesh object in the list
            meshes.append(mesh)

        # Calculate the center of the global AABB
        aabb_center = (global_min_bound + global_max_bound) / 2

        # Step 2: Translate each mesh, recenter, adjust Z, and save to the 'transformed' directory
        for mesh, ply_file in zip(meshes, ply_files):
            # Translate mesh to recenter the global AABB and set minimum Z to 0
            if recenter_origin:
                mesh.translate(-aabb_center)  # Recenter to (0, 0, 0)
                mesh.translate([0, 0, -global_min_z])  # Set minimum Z to 0
            # print("After transformation:")
            # print(mesh)
            # print("Bounding Box:", mesh.get_axis_aligned_bounding_box())
            # Save the transformed mesh to the 'transformed' directory
            output_file = transformed_dir / ply_file.name
            o3d.io.write_triangle_mesh(output_file.as_posix(), mesh)
        
        logger.info(f"Mesh transformation complete! Transformed files saved in '{transformed_dir}'.")

    def create_meshes(self):
        """
        Run the full mesh extraction and transformation pipeline.
        """
        self._prepare_mesh_extraction()
        self._split_bin_by_category()
        self._convert_bin_to_ply()
        self._transform_meshes_open3d()
 
    def load_extracted_meshes(self):
        """
        Loads the processed meshes and stores the file paths and loaded meshes in the category dict.
        """
        ply_files = [f for f in self.transformed_dir.iterdir() if f.suffix == '.ply']
        for file_path in ply_files:           
            file_stem = file_path.stem.upper()
            for category in CATEGORIES:
                if category in file_stem:  # Match based on the category prefix
                    self._extracted_mesh_dict[category] = {}
                    self._extracted_mesh_dict[category]["file"] = file_path
                    mesh = o3d.io.read_triangle_mesh(file_path.as_posix())
                    self._extracted_mesh_dict[category]["mesh"] = mesh
                    self._extracted_mesh_dict[category]["surface_area"] = mesh.get_surface_area()
                    break                  

    def load_extracted_meshes_vtk(self):
        """
        Loads the processed meshes using VTK and stores the file paths and loaded meshes in the category dict.
        """
        ply_files = [f for f in self.transformed_dir.iterdir() if f.suffix == '.ply']
        for file_path in ply_files:           
            file_stem = file_path.stem.upper()
            for category in CATEGORIES:
                if category in file_stem and category not in self._extracted_mesh_dict:  # Match based on the category prefix
                    self._extracted_mesh_dict[category] = {}
                    self._extracted_mesh_dict[category]["file"] = file_path

                    # Load the mesh using VTK
                    reader = vtk.vtkPLYReader()
                    reader.SetFileName(file_path.as_posix())
                    reader.Update()
                    mesh = reader.GetOutput()  # vtkPolyData object

                    # Compute normals for proper rendering and further processing
                    normals_filter = vtk.vtkPolyDataNormals()
                    normals_filter.SetInputData(mesh)
                    normals_filter.ComputePointNormalsOn()
                    normals_filter.Update()
                    mesh_with_normals = normals_filter.GetOutput()

                    # Store the mesh in the dictionary
                    self._extracted_mesh_dict[category]["mesh"] = mesh_with_normals

                    # Calculate the surface area using VTK's MassProperties
                    mass_properties = vtk.vtkMassProperties()
                    mass_properties.SetInputData(mesh_with_normals)
                    surface_area = mass_properties.GetSurfaceArea()

                    # Store the surface area in the dictionary
                    self._extracted_mesh_dict[category]["surface_area"] = surface_area
                    break

    def ensure_meshes(self, bin_file=None):
        """
        Loads any existing extracted meshes, or runs extraction if necessary.
        """
        # If necessary, convert .bin files for this label, else load the converted meshes.
        if self._label_has_extracted_meshes():
            logger.info(f"Label {self.label} already has extracted meshes.")
        else:
            logger.info(f"Label {self.label} does not currently have extracted meshes. Attempting now.")
            self.create_meshes()
        self.load_extracted_meshes_vtk()
           
    def save_splits(self, split_dict):
        """
        Func to save any splits as .ply mesh files in train/test/eval dirs.
        """
        self._ensure_split_dirs()

        combined_dict = combine_category_meshes(split_dict)
        for split in combined_dict:
            save_dir = self.split_dirs[split]
            logger.info(f"saving {split} files to {save_dir}")
            for category, mesh in combined_dict[split].items():
                p = save_dir / f"{category}.ply"
                o3d.io.write_triangle_mesh(p.as_posix(), mesh)

    def generate_and_save_fold_clouds(self, resolution, poisson_radius, output_format='.las', save_all_formats=False):
        """
        Generates points from the fold meshes and saves them in the specified format.
        If save_all_formats=True, it saves both .las and .pth formats.
        """
        valid_formats = ['.las', '.pth']
        if output_format not in valid_formats:
            raise ValueError(f"extension {output_format} not supported. Valid formats: {valid_formats}")

        resolution_root = self.clouds_root / f"res{resolution}_pr{poisson_radius}"
        label_root = resolution_root / self.label
        
        # Scrub the label_root directory (remove it if it exists and recreate it)
        if label_root.exists():
            shutil.rmtree(label_root)

        split_cloud_dirs = {
            'train': label_root / "train",
            'test': label_root / "test",
            'eval': label_root / "eval"
        }
        # resolution_root.mkdir(parents=True, exist_ok=True)
        for path in [resolution_root, label_root] + list(split_cloud_dirs.values()):
            path.mkdir(parents=True, exist_ok=True)

        for fold, output_dir in split_cloud_dirs.items():
            mesh_dir = self.split_dirs[fold]
            for in_f in mesh_dir.glob('*.ply'):

                gt_value = int(in_f.stem.split('_')[0])
                sampler = MeshSampler(mesh_path=in_f, gt_value=gt_value)
                sampler.generate_cloud(resolution=resolution, poisson_radius=poisson_radius)

                # Save in the specified output format
                if output_format == '.las' or save_all_formats:
                    las_output_name = output_dir / f"{in_f.stem}.las"
                    sampler.save(las_output_name)

                if output_format == '.pth' or save_all_formats:
                    pth_output_name = output_dir / f"{in_f.stem}.pth"
                    sampler.save(pth_output_name)

            # Also generate a combined file for the fold in the requested format(s)
            if output_format == '.las' or save_all_formats:
                las_in_files = list(output_dir.glob('*.las'))
                combined_las_name = output_dir / "combined.las"
                combine_las_files(input_paths=las_in_files, output_path=combined_las_name)

            if output_format == '.pth' or save_all_formats:
                pth_in_files = list(output_dir.glob('*.pth'))
                combined_pth_name = output_dir / "combined.pth"
                combine_pth_files(label=f"{self.label}_{fold}", input_paths=pth_in_files, output_path=combined_pth_name)


#################################################################################
# MeshAnalyser - tools for manipulating meshes and evaluating folds
#################################################################################
class MeshAnalyser:
    def __init__(self, data_handler=None):
        if not isinstance(data_handler, DataHandler):
            raise TypeError("data_handler must be an instance of DataHandler")
        self.dh = data_handler
        self._cached_aabb = None
        self._mesh_dict = {}
        self._mesh_area_dict = {}

        # Compute surface area for each mesh using vtkMassProperties
        for category, mesh in data_handler.extracted_meshes.items():
            if mesh is not None and mesh.GetNumberOfPoints() > 0:  # Ensure mesh is valid and not empty
                mass_properties = vtk.vtkMassProperties()
                mass_properties.SetInputData(mesh)
                mass_properties.Update()
                self._mesh_area_dict[category] = mass_properties.GetSurfaceArea()
            else:
                self._mesh_area_dict[category] = 0.0  # Assign 0.0 if the mesh is empty or None

        # Compute the total surface area
        self.total_mesh_surface_area = sum(area for area in self._mesh_area_dict.values())


    @property
    def aabb_all_meshes(self):
        """
        Returns the min and max AABB bounds across all meshes in the form of a dict,
        and caches these values after first computation.
        """
        if self._cached_aabb is not None:
            return self._cached_aabb  # Return cached value if available

        global_min_bound = np.array([np.inf, np.inf, np.inf])  # Initialize to positive infinity
        global_max_bound = np.array([-np.inf, -np.inf, -np.inf])  # Initialize to negative infinity
    
        # Compute the combined AABB over all meshes
        for category, mesh in self.meshes.items():
            if mesh is not None and not mesh.is_empty():
                # Get the min and max bounds of the current mesh
                min_bound = np.asarray(mesh.get_min_bound())
                max_bound = np.asarray(mesh.get_max_bound())
                
                # Update the global AABB bounds
                global_min_bound = np.minimum(global_min_bound, min_bound)
                global_max_bound = np.maximum(global_max_bound, max_bound)

        # Cache the result before returning
        self._cached_aabb = {
            "min": global_min_bound,
            "max": global_max_bound
        }
        return self._cached_aabb

    @property
    def aabb_all_meshes_vtk(self):
        """
        Returns the min and max AABB bounds across all vtkPolyData meshes in the form of a dict,
        and caches these values after first computation.
        """
        if self._cached_aabb is not None:
            return self._cached_aabb  # Return cached value if available

        # Initialize global bounds to extreme values
        global_min_bound = np.array([np.inf, np.inf, np.inf])  # Initialize to positive infinity
        global_max_bound = np.array([-np.inf, -np.inf, -np.inf])  # Initialize to negative infinity
        
        # Compute the combined AABB over all meshes
        for category, mesh in self.meshes.items():
            if mesh is not None and mesh.GetNumberOfPoints() > 0:  # Ensure mesh is valid and not empty
                # Get the bounds of the current mesh as a tuple (xmin, xmax, ymin, ymax, zmin, zmax)
                bounds = mesh.GetBounds()

                # Convert bounds to min and max bound arrays
                min_bound = np.array([bounds[0], bounds[2], bounds[4]])  # (xmin, ymin, zmin)
                max_bound = np.array([bounds[1], bounds[3], bounds[5]])  # (xmax, ymax, zmax)
                
                # Update the global AABB bounds
                global_min_bound = np.minimum(global_min_bound, min_bound)
                global_max_bound = np.maximum(global_max_bound, max_bound)

        # Cache the result before returning
        self._cached_aabb = {
            "min": global_min_bound,
            "max": global_max_bound
        }
        return self._cached_aabb

    @property
    def meshes(self):
        return self.dh.extracted_meshes

    #################################################################################
    # Toy pointcloud generation
    #################################################################################  
    def generate_toy_pcds(self, resolution=0.05, categories=None, excluded_categories=None):
        """
        Generate point clouds from the meshes stored in category_dict using MeshSampler.

        Args:
            resolution (float): The resolution to use for sampling points.
            categories (list, optional): List of categories to generate point clouds for. Defaults to None (all categories).

        Returns:
            dict: A dictionary with categories as keys and sampled point clouds (numpy arrays) as values.
        """
        dh = self.dh
        # Set categories to all keys in meshes if not provided
        if categories is None:
            categories = list(self.meshes.keys())

        if not excluded_categories:
            excluded_categories = []

        logger.info(f"Generating toy pointclouds for categories {categories}.")
        logger.info(f"Sampling with resolution {resolution}.")
        pcd_dict = {}

        # Iterate over the specified categories
        for category, mesh in self.meshes.items():
            if category in categories and category not in excluded_categories:
                # Check if mesh is valid
                if mesh is not None and mesh.GetNumberOfPoints() > 0:
                    # Initialize MeshSampler with the current mesh
                    mesh_sampler = MeshSampler(mesh_data=mesh, gt_value=0)  # 'gt_value' can be any placeholder, not used here.
                    
                    # Generate point cloud with the specified resolution
                    mesh_sampler.generate_cloud(resolution=resolution, calculate_normals_colors=False)
                    
                    # Get the sampled points directly from the _points attribute
                    pcd_dict[category] = mesh_sampler._points

                    # logger.info(f"  Sampled {len(sampled_points)} points for category {category}")
                else:
                    logger.warning(f"Mesh for category '{category}' is invalid or empty.")
            else:
                logger.warning(f"Category '{category}' not found in meshes.")

        logger.info("Finished generating toy PCDs.")
        self.toy_pcds = pcd_dict
        return pcd_dict

    def evaluate_binning(self, pcd_dict, x_cell_width=3.0, y_cell_width=3.0):
        """
        Evaluates a binning schema for a dictionary of Open3D point clouds using AABB from meshes for histogram limits,
        with bins specified by x and y cell widths. The schema will cover the entire AABB and may extend beyond it.
    
        Args:
            pcd_dict (dict): Dictionary where each key is a category and each value is an Open3D point cloud.
            x_cell_width (float): Width of each cell along the x-axis.
            y_cell_width (float): Width of each cell along the y-axis.
    
        Returns:
            dict: A dictionary with keys as categories and values as a 2D array of counts per bin.
        """
        # Use the AABB from all meshes to define the bin limits
        aabb = self.aabb_all_meshes_vtk
        min_pt = aabb['min']
        max_pt = aabb['max']
    
        # Compute the number of bins needed to cover the AABB, potentially extending beyond it
        num_bins_x = int(np.ceil((max_pt[0] - min_pt[0]) / x_cell_width))
        num_bins_y = int(np.ceil((max_pt[1] - min_pt[1]) / y_cell_width))
        logger.info(f"Binning is using {num_bins_x} bins in X and {num_bins_y} bins in Y based on cell widths of {x_cell_width} and {y_cell_width}.")
    
        # Create bin edges based on the AABB, ensuring coverage beyond the AABB if necessary
        x_edges = np.linspace(min_pt[0], min_pt[0] + num_bins_x * x_cell_width, num_bins_x + 1)
        y_edges = np.linspace(min_pt[1], min_pt[1] + num_bins_y * y_cell_width, num_bins_y + 1)
    
        # Evaluate binning for each point cloud
        bin_counts = {}
        for category, points in pcd_dict.items():
            # points = np.asarray()
            hist, _, _ = np.histogram2d(points[:, 0], points[:, 1], bins=(x_edges, y_edges))
            bin_counts[category] = hist
    
        return {
            'counts': bin_counts,
            'x_edges': x_edges,
            'y_edges': y_edges,
        }

    # def generate_library_splits(self, cell_width=3.0, weights=(0.65, 0.2, 0.15), random_seed=9039501):
    #     """
    #     Function to generate splits with a special algorithm for the library dataset.
    #     """
    #     category_cells = divide_all_categories_into_cells_pyvista(self.meshes, cell_width)
    #     # transformed_category_cells = transform_cells(category_cells)
    #     logger.info("Cell division complete, now assigning to folds and transforming...")
    #     splits = split_all_categories(category_cells, weights=weights)
    #     process_splits_pyvista(splits, cell_width=cell_width, seed=random_seed)
    #     logger.info("Fold allocation complete!")
    #     return splits


#################################################################################
# MeshSampler - generates production .las and .pth files from the meshes
#################################################################################
class MeshSampler:
    """
    A class for sampling point clouds from mesh files and saving them in various formats.
    
    Methods:
        generate_cloud(resolution): Generates a point cloud from the mesh at a specified resolution.
        save(output_path): Saves the generated point cloud to a specified path.
    """
    
    def __init__(self, mesh_path=None, mesh_data=None, gt_value=None):
        """
        Initializes the MeshSampler object with the specified mesh file or vtkPolyData and ground truth value.

        Args:
            mesh_path (str, optional): Path to the mesh file.
            mesh_data (vtk.vtkPolyData, optional): A vtkPolyData object representing the mesh.
            gt_value (int): Ground truth value to assign to each point.
        
        Raises:
            ValueError: If neither mesh_path nor mesh_data is provided.
            ValueError: If both mesh_path and mesh_data are provided.
            FileNotFoundError: If the mesh_path does not exist or is not a file.
            TypeError: If gt_value is not an integer.
        """
        # Validate input arguments
        if mesh_path is None and mesh_data is None:
            raise ValueError("Either mesh_path or mesh_data must be provided.")
        if mesh_path is not None and mesh_data is not None:
            raise ValueError("Only one of mesh_path or mesh_data should be provided.")

        if gt_value is None or not isinstance(gt_value, int):
            raise TypeError(f"gt_value must be an integer, got {type(gt_value).__name__} instead.")
        
        self._gt_value = gt_value
        
        # Initialize mesh based on provided input
        if mesh_path is not None:
            # Load mesh from file path
            self.mesh_path = Path(mesh_path)
            if not self.mesh_path.exists() or not self.mesh_path.is_file():
                raise FileNotFoundError(f"Provided mesh path does not exist or is not a file: {mesh_path}")
            
            # Read the mesh using VTK
            reader = vtk.vtkPLYReader()
            reader.SetFileName(self.mesh_path.as_posix())
            reader.Update()
            self.mesh = reader.GetOutput()
        else:
            # Use provided vtkPolyData object directly
            if not isinstance(mesh_data, vtk.vtkPolyData):
                raise TypeError(f"mesh_data must be a vtkPolyData object, got {type(mesh_data).__name__} instead.")
            self.mesh = mesh_data

        # Ensure the mesh is valid vtkPolyData
        if not isinstance(self.mesh, vtk.vtkPolyData):
            raise TypeError("Mesh must be a vtkPolyData object.")

        self._points = None
        self._colors = None
        self._normals = None
    
    @staticmethod
    def check_output_path_viability(output_path):
        """
        Static method to check the viability of an output path.

        Args:
            output_path (str or Path): The output path to validate.

        Raises:
            ValueError: If the output path is None or empty.
            FileNotFoundError: If the directory does not exist and cannot be created.
        """
        if not output_path:
            raise ValueError("Output path is not provided or is empty.")

        # Convert to Path object if necessary
        path = Path(output_path) if not isinstance(output_path, Path) else output_path

        # Check if the directory exists or try to create it
        if not path.parent.exists():
            try:
                path.parent.mkdir(parents=True, exist_ok=True)
                logger.info(f"Directory created: {path.parent}")
            except Exception as e:
                logger.error(f"Failed to create directory: {path.parent}")
                raise FileNotFoundError(f"Failed to create directory at {path.parent}: {e}")
        
    def generate_cloud(
        self,
        poisson_radius=None,  # Radius for Poisson Disk Sampling
        resolution=0.02,  # Sampling distance, computed as a function of poisson_radius if not provided
        calculate_normals_colors=True,  # Whether to calculate normals and interpolate colors
        apply_poisson_disk=True  # Whether to apply Poisson Disk Sampling after initial sampling
    ):
        """
        Generates a sampled point cloud from the mesh at the given resolution.

        Args:
            poisson_radius (float): Radius to use for Poisson Disk Sampling.
            resolution (float): The distance between sampled points; defaults to poisson_radius / 2.
            calculate_normals_colors (bool): If True, calculate normals and interpolate colors.
            apply_poisson_disk (bool): If True, apply Poisson Disk Sampling after the initial sampling.
        """
        # Determine resolution based on poisson_radius if not explicitly provided
        if poisson_radius is None:
            poisson_radius = 7/8 * resolution  # This seems a good compromise for point spatial and density uniformity

        # Step 1: Setup the reader and load the mesh
        input_mesh = self.mesh

        # Compute normals on the original mesh if interpolating normals
        if calculate_normals_colors:
            normals_filter = vtk.vtkPolyDataNormals()
            normals_filter.SetInputData(input_mesh)
            normals_filter.ComputePointNormalsOn()
            normals_filter.ConsistencyOn()
            normals_filter.AutoOrientNormalsOn()
            normals_filter.Update()
            mesh_with_normals = normals_filter.GetOutput()
        else:
            mesh_with_normals = input_mesh

        # Step 2: Setup Poisson Disk Sampler to sample the mesh
        sampler = vtk.vtkPolyDataPointSampler()
        sampler.SetInputData(mesh_with_normals)
        sampler.SetDistance(resolution)
        sampler.Update()
        sampled_point_cloud = sampler.GetOutput()
        num_points_initial = sampled_point_cloud.GetNumberOfPoints()
        logger.info(f"Number of points after initial sampling: {num_points_initial}")

        # Step 3: Optionally apply Poisson Disk Sampling
        if apply_poisson_disk:
            poisson_disk_sampler = vtk.vtkPoissonDiskSampler()
            poisson_disk_sampler.SetInputData(sampled_point_cloud)
            poisson_disk_sampler.SetRadius(poisson_radius)
            poisson_disk_sampler.Update()
            sampled_point_cloud = poisson_disk_sampler.GetOutput()
            num_points_poisson = sampled_point_cloud.GetNumberOfPoints()
            logger.info(f"Number of points after Poisson Disk Sampling: {num_points_poisson}")

        # Step 4: Interpolate colors and normals from the original mesh to the sampled points
        if calculate_normals_colors:
            probe_filter = vtk.vtkProbeFilter()
            probe_filter.SetInputData(sampled_point_cloud)
            probe_filter.SetSourceData(mesh_with_normals)
            probe_filter.Update()

            # Get the output with interpolated normals and colors
            interpolated_point_cloud = probe_filter.GetOutput()

            # Convert the interpolated colors and points to numpy arrays
            self._points = vtk_to_numpy(interpolated_point_cloud.GetPoints().GetData())
            self._colors = vtk_to_numpy(interpolated_point_cloud.GetPointData().GetScalars()) if interpolated_point_cloud.GetPointData().GetScalars() else None
            self._normals = vtk_to_numpy(interpolated_point_cloud.GetPointData().GetNormals())
        else:
            # If normals and colors are not to be calculated, simply extract points
            self._points = vtk_to_numpy(sampled_point_cloud.GetPoints().GetData())
            self._colors = None
            self._normals = None


    def save(self, output_path=None):
        """
        Saves the generated point cloud to a specified file path. The format is determined by the file extension.

        Args:
            output_path (str): Path where the point cloud will be saved.

        Raises:
            ValueError: If an unsupported file extension is provided.
        """
        p = Path(output_path)
        self.check_output_path_viability(p)      
        file_extension = p.suffix
        if file_extension == '.las':
            self._save_as_las(p)
        elif file_extension == '.pth':
            self._save_as_pth(p)
        else:
            raise ValueError(f"Unsupported file extension: {file_extension}. Please use '.las' or '.pth'.")

    def _save_as_las(self, output_path):
        """
        Saves the point cloud data in LAS format.

        Args:
            output_path (Path): Path to save the LAS file.
        """
        points = self._points
        normals = self._normals
        logger.info("Saving cloud as .las format...")
        header = laspy.LasHeader(version="1.4", point_format=8)
        header.x_scale = 0.001  # Adjust the scale to match the units of your points
        header.y_scale = 0.001
        header.z_scale = 0.001
        
        # Set the offset to the minimum values to avoid loss of precision
        header.offsets = [np.min(points[:, 0]), np.min(points[:, 1]), np.min(points[:, 2])]
        las = laspy.LasData(header)
        # Rescale the colors to .las format
        las_colors = np.round(self._colors * (65535/255)).astype(np.uint16)
        
        # Fill in the point records
        las.x = points[:, 0]
        las.y = points[:, 1]
        las.z = points[:, 2]
        las.red = las_colors[:, 0]
        las.green = las_colors[:, 1]
        las.blue = las_colors[:, 2]
        
        # Add normals as extra dimensions to the LAS file
        las.add_extra_dim(laspy.ExtraBytesParams(name="NormalX", type=np.float32, description="X component of normal vector"))
        las.add_extra_dim(laspy.ExtraBytesParams(name="NormalY", type=np.float32, description="Y component of normal vector"))
        las.add_extra_dim(laspy.ExtraBytesParams(name="NormalZ", type=np.float32, description="Z component of normal vector"))        
        las['NormalX'] = normals[:,0]
        las['NormalY'] = normals[:,1]
        las['NormalZ'] = normals[:,2]

        # Add the ground truth as a scalar field
        las.add_extra_dim(laspy.ExtraBytesParams(name="gt", type=np.float32, description="Ground truth field"))
        las['gt'] = np.broadcast_to(self._gt_value, las.header.point_count)

        las.write(output_path.as_posix())
        logger.info(f"Saved as: {output_path}")
    
    def _save_as_pth(self, output_path):
        """
        Saves the point cloud data in PyTorch .pth format.

        Args:
            output_path (Path): Path to save the .pth file.
        """
        # Extract points, colors, normals, and other relevant data from the class
        points = self._points  # Nx3 array of coordinates (x, y, z)
        colors = self._colors  # Nx3 array of colors (r, g, b) normalized [0, 1]
        normals = self._normals  # Nx3 array of normal vectors
        gt_value = self._gt_value  # Ground truth value or placeholder

        # Prepare the coordinate array (float32)
        coord = points.astype(np.float32)

        # Prepare the color array (float32)
        color = colors.astype(np.float32)

        # Dictionary to store the data
        data = {
            'coord': coord,
            'color': color,
            'scene_id': str(output_path.stem)  # Use the filename (without extension) as scene_id
        }

        # Add normals if present
        if normals is not None:
            data['normal'] = normals.astype(np.float32)

        # Optionally, add semantic ground truth data if available
        if gt_value is not None:
            # Assuming gt_value is scalar, we broadcast it for each point (replace if different per point)
            semantic_gt20 = np.broadcast_to(gt_value, (points.shape[0],)).astype(np.int64)
            data['gt'] = semantic_gt20

        # Save the dictionary as a .pth file
        torch.save(data, output_path.as_posix())
        logger.info(f"Saved point cloud to {output_path} in .pth format.")


def combine_pth_files(label, input_paths, output_path):
    """
    Combines multiple .pth files into one .pth file while handling the process incrementally.

    Args:
        output_path (str or Path): The path where the combined .pth file will be saved.
        input_paths (list of str or Path): List of paths (either str or Path) to .pth files to be combined.
    """
    # Normalize the output_path to a Path object
    output_path = Path(output_path)
    
    combined_data = {
        'coord': [],
        'color': [],
        'normal': [],
        'gt': [],
        'scene_id': label,
    }

    # Process each .pth file incrementally
    for file_path in input_paths:
        # Normalize the input path to a Path object
        file_path = Path(file_path)

        # Load the .pth file
        data = torch.load(file_path)

        # Append the data from this file to the combined data
        combined_data['coord'].append(data['coord'])
        combined_data['color'].append(data['color'])
        
        if 'normal' in data:
            combined_data['normal'].append(data['normal'])
        
        if 'gt' in data:
            combined_data['gt'].append(data['gt'])
        
    # Concatenate the arrays (out-of-core behavior depends on NumPy's internal optimizations)
    combined_data['coord'] = np.concatenate(combined_data['coord'], axis=0)
    combined_data['color'] = np.concatenate(combined_data['color'], axis=0)
    
    if combined_data['normal']:
        combined_data['normal'] = np.concatenate(combined_data['normal'], axis=0)
    
    if combined_data['gt']:
        combined_data['gt'] = np.concatenate(combined_data['gt'], axis=0)

    # Save the combined data to a new .pth file
    torch.save(combined_data, output_path.as_posix())
    logger.info(f"Combined data saved to {output_path}")


def combine_las_files(input_paths, output_path):
    """
    Combines multiple LAS files into one, ensuring that point positions are distinct by adjusting for local offsets.
    
    Args:
        output_path (str or Path): The path where the combined LAS file will be saved.
        input_paths (list of str or Path): List of paths (either str or Path) to LAS files to be combined.
    """
    output_path = Path(output_path)

    # Open the first LAS file to initialize the header for the output file
    with laspy.open(input_paths[0], mode='r') as first_file:
        header = laspy.LasHeader(version=first_file.header.version, point_format=first_file.header.point_format)
        # Keep the scaling and offsets from the first file as the reference for the combined file
        header.x_scale = first_file.header.x_scale
        header.y_scale = first_file.header.y_scale
        header.z_scale = first_file.header.z_scale
        header.offsets = first_file.header.offsets  # Keep the first file's offsets

    # Open the output file in write mode
    with laspy.open(output_path, mode='w', header=header) as out_file:
        total_points = 0

        # Process each input LAS file incrementally
        for file_path in input_paths:
            file_path = Path(file_path)

            # Open the LAS file for reading
            with laspy.open(file_path, mode='r') as in_file:
                # Get the scale and offset of the current LAS file
                x_offset, y_offset, z_offset = in_file.header.x_offset, in_file.header.y_offset, in_file.header.z_offset

                for points in in_file.chunk_iterator(1000000):  # Process in chunks of 1,000,000 points
                    # Convert the local coordinates to global coordinates using this file's offsets
                    points.x = points.x + (x_offset - header.x_offset)
                    points.y = points.y + (y_offset - header.y_offset)
                    points.z = points.z + (z_offset - header.z_offset)

                    # Write the points in global coordinates to the output file
                    out_file.write_points(points)
                    total_points += len(points)

        logger.info(f"Combined LAS file saved to {output_path} with {total_points} points.")
