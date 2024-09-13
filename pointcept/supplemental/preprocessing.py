import subprocess
import shutil
import logging
import sys
import random
import re
import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path

from pointcept.supplemental.utils import get_category_list
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering


# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', datefmt='%H:%M:%S')
logger = logging.getLogger(__name__)

CATEGORIES = get_category_list()


class ParametrixPreprocessor():

    ##########################################################################
    # Init and data preparation methods
    ##########################################################################
    def __init__(
        self, label="park_row", cc_path="org.cloudcompare.CloudCompare",
        root_dir="/home/sogilvy/repos/Pointcept/data/parametrix",
        bin_file=None,
    ):
            
        self.label = label
        self.cc_path = cc_path
        
        # Required directories
        self.root_dir = Path(root_dir)
        self.mesh_dir = self.root_dir / "meshes"
        self.raw_mesh_dir = self.mesh_dir / "raw"
        self.extraction_dir = self.mesh_dir / "extracted" / label
        self.split_dir = self.extraction_dir / "splits"
        self.split_dirs = {}
        self.split_dirs['train'] = self.extraction_dir / "train"
        self.split_dirs['test'] = self.extraction_dir / "test"
        self.split_dirs['eval'] = self.extraction_dir / "eval"

        # File paths
        self._set_bin_file(bin_file)
        self.raw_mesh_path_temp = self.extraction_dir / f"{label}.bin"

        # A dict to hold loaded and derived per-category information.
        # This can be non-exhaustive based on present categories!
        self.category_dict = {}

    def _set_bin_file(self, bin_file):
        if bin_file is None:
            self.raw_mesh_path = self.raw_mesh_dir / f"{self.label}.bin"
        else:
            logger.info(f"Input .bin specified: {bin_file}")
            self.raw_mesh_path = Path(bin_file)

    def _label_has_meshes(self):
        files = list(self.extraction_dir.glob('*.ply'))
        return len(files) > 0

    @property
    def meshes(self):
        if not self.category_dict: return {}
        return {category: data['mesh'] for category, data in self.category_dict.items()}

    @property
    def total_mesh_surface_area(self):
        return sum(data['surface_area'] for data in self.category_dict.values()) 

    @property
    def surface_areas(self):
        return {category: data['surface_area'] for category, data in self.category_dict.items()}
    
    def _verify_input_bin(self):
        try:
            assert self.raw_mesh_path.exists()
        except AssertionError:
            logging.error("Input .bin file for this config does not exist:")
            logging.error(f" {self.raw_mesh_path.as_posix()}")
            raise

    def _ensure_split_dirs(self):
        for dir_path in self.split_dirs.values():
            dir_path.mkdir(parents=True, exist_ok=True)

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
    
    def _extract_meshes_from_bin(self):
        """
        Extract .ply files from the raw .bin files.
        """
        self._prepare_mesh_extraction()
        self._split_bin_by_category()
    
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
        # for file in self.extraction_dir.iterdir():
        #     logger.info(f"  {file.name}")

    def _transform_meshes(self):
        """
        Takes the extracted .ply meshes, swaps z-y axes while maintaining chirality, rescales from cm to m,
        and recenters all meshes so that the center of their combined AABB is at (0, 0, 0).
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
    
        # Initialize variables to compute the combined AABB
        global_min_bound = np.array([np.inf, np.inf, np.inf])
        global_max_bound = np.array([-np.inf, -np.inf, -np.inf])
    
        # Step 1: Compute the combined AABB over all meshes
        meshes = []  # Store meshes for later processing
        for ply_file in ply_files:
            mesh = o3d.io.read_triangle_mesh(ply_file.as_posix())
            mesh.compute_vertex_normals()
            mesh.transform(transform_matrix)  # Apply initial transformation
            
            # Update global AABB
            min_bound = np.asarray(mesh.get_min_bound())
            max_bound = np.asarray(mesh.get_max_bound())
            global_min_bound = np.minimum(global_min_bound, min_bound)
            global_max_bound = np.maximum(global_max_bound, max_bound)
            
            meshes.append((mesh, ply_file))  # Store the mesh and its corresponding file for later processing
    
        # Step 2: Calculate the center of the global AABB
        aabb_center = (global_min_bound + global_max_bound) / 2
    
        # Step 3: Recenter each mesh so that the AABB center is at (0, 0, 0)
        for mesh, ply_file in meshes:
            # Translate mesh to center of global AABB
            mesh.translate(-aabb_center)
            # Overwrite the .ply file with the transformed mesh
            o3d.io.write_triangle_mesh(ply_file.as_posix(), mesh)
        
        logger.info("Mesh transformation complete!")
    
    def create_meshes(self):
        """
        Run the full mesh extraction and transformation pipeline.
        """
        self._extract_meshes_from_bin()
        self._transform_meshes()

    def load_processed_meshes(self):
        """
        Loads the processed meshes and stores the file paths and loaded meshes in the category dict.
        """
        ply_files = [f for f in self.extraction_dir.iterdir() if f.suffix == '.ply']
        for file_path in ply_files:           
            file_stem = file_path.stem.upper()
            for category in CATEGORIES:
                if category in file_stem:  # Match based on the category prefix
                    self.category_dict[category] = {}
                    self.category_dict[category]["file"] = file_path
                    mesh = o3d.io.read_triangle_mesh(file_path.as_posix())
                    self.category_dict[category]["mesh"] = mesh
                    self.category_dict[category]["surface_area"] = mesh.get_surface_area()
                    break                  

    def ensure_meshes(self, bin_file=None):
        """
        Loads any existing extracted meshes, or runs extraction if necessary.
        """
        self._set_bin_file(bin_file)
        # If necessary, convert .bin files for this label, else load the converted meshes.
        if self._label_has_meshes():
            logger.info(f"Label {self.label} already has extracted meshes.")
        else:
            logger.info(f"Label {self.label} does not currently have extracted meshes. Attempting now.")
            self.create_meshes()
        self.load_processed_meshes()

    ##########################################################################
    # Analysis methods
    ##########################################################################
    def get_aabb_all_meshes(self):
        """
        Returns the min and max AABB bounds across all meshes in the form of a dict.
        """
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

        # Return the combined AABB
        return {
            "min": global_min_bound,
            "max": global_max_bound
        }

    def get_least_populous_categories(self, n):
        """
        Returns a list of the n least populous mesh categories based on surface area.

        Args:
            n (int): The number of least populous categories to return.

        Returns:
            list: A list of the n least populous category names.
        """
        # Access surface areas from the class instance
        surface_areas = self.surface_areas
        
        # Sort categories by surface area in ascending order and get the first n categories
        sorted_categories = sorted(surface_areas, key=surface_areas.get)
        return sorted_categories[:n]
    
    def generate_toy_pcds(self, total_points=5000, normalize_by_area=True, categories=None):
        """
        Generate point clouds from the meshes stored in category_dict.
        
        Args:
            total_points (int): The total number of points to generate.
            normalize_by_area (bool): If True, normalize the number of points generated per category based on surface area.
            categories (list, optional): List of categories to generate point clouds for. Defaults to None (all categories).
        
        Returns:
            dict: A dictionary with categories as keys and sampled point clouds as values.
        """
        # Set categories to all keys in category_dict if not provided
        all_cats = False
        total = self.total_mesh_surface_area
        if categories is None:
            all_cats = True
            categories = list(self.category_dict.keys())
        else:
            total = sum(data['surface_area'] for cat, data in self.category_dict.items() if cat in categories)
            
        logger.info(f"Generating toy pointclouds for categories {categories}.")
        logger.info(f"Sampling {total_points} total points. Normalize_by_area = {normalize_by_area}")       
        pcd_dict = {}
        
        # Iterate over the specified categories
        for category in categories:
            if category in self.category_dict:
                data = self.category_dict[category]
                mesh = data['mesh']
                surface_area = data['surface_area']
                
                # Check if mesh is valid
                if mesh is not None and not mesh.is_empty():
                    # Compute the number of points to sample
                    if normalize_by_area:
                        # Normalizing by surface area
                        points_to_sample = int((surface_area / total) * total_points)
                    else:
                        # Evenly distribute points across all categories
                        points_to_sample = total_points // len(categories)  # Use the length of the selected categories
                    
                    # Ensure the mesh has vertex normals computed for better sampling
                    mesh.compute_vertex_normals()
    
                    # Sample points from the mesh based on the computed number
                    if points_to_sample > 0:  # Ensure positive number of points
                        sampled_pcd = mesh.sample_points_poisson_disk(number_of_points=points_to_sample)
                        pcd_dict[category] = sampled_pcd
                        logger.info(f"  Sampled {points_to_sample} points for category {category}")
            else:
                logger.warning(f"Category '{category}' not found in category_dict.")
        
        logger.info("Finished generating toy PCDs.")
        self.recent_toy_pcds = pcd_dict
        return pcd_dict

    def get_cluster_densities(self, pcd_dict, k_clusters):
        """
        Generates cluster density arrays for the n least populous categories using K-means clustering.

        Args:
            n (int): The number of least populous categories to consider.
            k_clusters (int): The number of K-means clusters to generate.
            total_points (int): The total number of points to generate for each category.
            normalize_by_area (bool): If True, normalize the number of points generated per category based on surface area.

        Returns:
            dict: A dictionary mapping each category to its array of cluster centers.
        """
        # Step 1: Get the n least populous categories
        # least_populous_categories = self.get_least_populous_categories(n)

        # Step 2: Generate point clouds for the least populous categories
        # pcd_dict = self.generate_toy_pcds(total_points=total_points, normalize_by_area=normalize_by_area, categories=least_populous_categories)

        # Step 3: Initialize a dictionary to store cluster means for each category
        category_to_cluster_means = {}

        # Step 4: Perform K-means clustering for each category and store the cluster means
        for category in pcd_dict:
            # Extract point cloud data from the Open3D PointCloud object
            sampled_pcd = pcd_dict[category]
            sampled_points = np.asarray(sampled_pcd.points)  # Convert to NumPy array

            # Collapse the point clouds to the XY plane
            xy_points = sampled_points[:, :2]

            # Normalize the data
            scaler = StandardScaler()
            xy_points_normalized = scaler.fit_transform(xy_points)  # Normalize the data

            # Perform K-means Clustering
            kmeans = KMeans(n_clusters=k_clusters, random_state=0, init='k-means++')  # Use k-means++ for better initialization
            kmeans.fit(xy_points_normalized)

            # Transform cluster centers back to original scale
            cluster_centers = scaler.inverse_transform(kmeans.cluster_centers_)

            # Store cluster means in the dictionary
            category_to_cluster_means[category] = cluster_centers

        # Step 5: Return the dictionary with cluster density arrays
        return category_to_cluster_means

    def create_figure_base(self, figure_scale):
        """
        Plot a figure with appropriate dimensions.
        """
        # Step 2: Get the AABB (Axis-Aligned Bounding Box) across all meshes
        aabb = self.get_aabb_all_meshes()
        min_x, max_x = aabb['min'][0], aabb['max'][0]
        min_y, max_y = aabb['min'][1], aabb['max'][1]
        
        # Calculate the range for preserving aspect ratio
        range_x = max_x - min_x
        range_y = max_y - min_y
        
        # Use figure_scale as the minimum x or y edge size
        scale_x, scale_y = 1, 1
        if range_x > range_y:
            scale_x = (range_x / range_y) * figure_scale
            scale_y = figure_scale
        else:
            scale_y = (range_y / range_x) * figure_scale
            scale_x = figure_scale

        plt.figure(figsize=(scale_x, scale_y))  # Preserve the aspect ratio of the AABB
        plt.xlim(min_x, max_x)
        plt.ylim(min_y, max_y)

    def plot_cluster_means(self, category_to_cluster_means, figure_scale=3):
        # Setup the figure using the custom helper function
        self.create_figure_base(figure_scale=figure_scale)

        # Fetch categories and prepare the colormap
        categories = list(category_to_cluster_means.keys())
        cmap = plt.get_cmap('tab20')  # Using 'tab20' to allow for up to 20 unique colors
        n_categories = len(categories)
        color_norm = plt.Normalize(vmin=0, vmax=n_categories - 1)  # Normalize color mapping

        # Plot cluster means for each category
        handles = []  # To collect legend handles
        for idx, category in enumerate(categories):
            cluster_means = category_to_cluster_means[category]
            scatter = plt.scatter(cluster_means[:, 0], cluster_means[:, 1],
                                color=cmap(color_norm(idx)), label=category,
                                s=100, alpha=0.7, edgecolor='k', marker='o')
            handles.append(scatter)

        # Add legend to differentiate categories
        legend = plt.legend(handles, categories, loc='upper left', bbox_to_anchor=(1.05, 1.0), title='Categories')
        for handle in legend.legend_handles:
            handle.set_alpha(1)  # Set alpha to 1 for legend handles to show full color

        # Set plot labels and title
        plt.xlabel('X Coordinate')
        plt.ylabel('Y Coordinate')
        plt.title('Cluster Means for Each Category')

        # Ensure aspect ratio is maintained from create_figure_base
        plt.gca().set_aspect('equal', adjustable='box')

        # Show grid for better visibility
        plt.grid(True)

        # Display the plot
        plt.show()

    def plot_overlay_2d_distributions(self, pcd_dict, figure_scale):
        """
        Plots an overlay of 2D distributions for each category's point cloud on the same plot with different colors,
        using a custom figure setup to respect the global AABB aspect ratio and a color map for many categories.
        Fixes legend colors to match the plot colors vividly.

        Args:
            pcd_dict (dict): Dictionary mapping each category to its Open3D point cloud object.
            figure_scale (float): Scale factor for figure size.
        """
        # Setup the figure using the custom helper function
        self.create_figure_base(figure_scale)

        # Use a color map that supports more categories. `tab20` or `viridis` can be good choices.
        cmap = plt.get_cmap('tab20')  # 'tab20' supports 20 unique colors
        n_categories = len(pcd_dict.keys())
        color_norm = plt.Normalize(vmin=0, vmax=n_categories - 1)

        # Create a plot and collect labels and handles for legend that correctly reflect the colors
        handles = []
        labels = []
        
        # Plot points for each category
        for idx, (category, pcd) in enumerate(pcd_dict.items()):
            points = np.asarray(pcd.points)  # Convert Open3D PointCloud to NumPy array
            x_coords = points[:, 0]  # X coordinates
            y_coords = points[:, 1]  # Y coordinates
            
            scatter = plt.scatter(x_coords, y_coords, alpha=0.5, color=cmap(color_norm(idx)), label=category)
            handles.append(scatter)
            labels.append(category)

        # Set the aspect ratio to be equal, maintaining the AABB proportions
        plt.gca().set_aspect('equal', adjustable='box')

        # Finalize plot settings
        plt.title('Overlay of 2D Point Distributions')
        plt.xlabel('X Coordinate')
        plt.ylabel('Y Coordinate')
        plt.grid(True)

        # Fix legend to display the correct color intensity
        legend = plt.legend(handles, labels, title='Category', bbox_to_anchor=(1.05, 1), loc='upper left')
        for handle in legend.legend_handles:
            handle.set_alpha(1)  # Set alpha to 1 for legend handles to show full color

        plt.show()
        return scatter

    def evaluate_binning(self, pcd_dict, num_bins_x=None, num_bins_y=None):
        """
        Evaluates a binning schema for a dictionary of Open3D point clouds using AABB from meshes for histogram limits,
        with evenly spaced bins in x and y. Defaults to creating bins approximately 8 units in size when not specified.

        Args:
            pcd_dict (dict): Dictionary where each key is a category and each value is an Open3D point cloud.
            num_bins_x (int): Optional. Number of bins to use in x dimension.
            num_bins_y (int): Optional. Number of bins to use in y dimension.

        Returns:
            dict: A dictionary with keys as categories and values as a 2D array of counts per bin.
        """
        # Use the AABB from all meshes to define the bin limits
        aabb = self.get_aabb_all_meshes()
        min_pt = aabb['min']
        max_pt = aabb['max']

        # Desired size of each bin in terms of units (approximately 8x8 units if not specified)
        desired_bin_size = 8

        # Compute default number of bins if not specified
        if num_bins_x is None:
            num_bins_x = max(1, int(np.round((max_pt[0] - min_pt[0]) / desired_bin_size)))
        if num_bins_y is None:
            num_bins_y = max(1, int(np.round((max_pt[1] - min_pt[1]) / desired_bin_size)))
        logger.info(f"Binning is using {num_bins_x} bins in X, and {num_bins_y} bins in Y.")
        # Create bin edges based on the AABB
        x_edges = np.linspace(min_pt[0], max_pt[0], num_bins_x + 1)
        y_edges = np.linspace(min_pt[1], max_pt[1], num_bins_y + 1)

        # Evaluate binning for each point cloud
        bin_counts = {}
        for category, pcd in pcd_dict.items():
            points = np.asarray(pcd.points)
            hist, _, _ = np.histogram2d(points[:, 0], points[:, 1], bins=(x_edges, y_edges))
            bin_counts[category] = hist

        return bin_counts

    def generate_library_splits(self, cell_width=3.0, weights=(0.65, 0.2, 0.15), random_seed=9039501):
        """
        Function to generate splits with a special algorithm for the library dataset.
        """
        cell_width = 2.5
        category_cells = divide_all_categories_into_cells(self.meshes, cell_width)
        transformed_category_cells = transform_cells(category_cells)
        splits = split_all_categories(transformed_category_cells, weights=weights)
        processed_meshes = process_splits(splits, cell_width, seed=random_seed)
        return splits, processed_meshes

    def save_splits(self, split_dict):
        """
        Func to save any splits as .ply mesh files in train/test/eval dirs.
        """
        self._ensure_split_dirs()
        # self.split_dir.mkdir(parents=True, exist_ok=True)

        combined_dict = combine_category_meshes(split_dict)
        for split in combined_dict:
            save_dir = self.split_dirs[split]
            logger.info(f"saving {split} files to {save_dir}")
            for category, mesh in combined_dict[split].items():
                p = save_dir / f"{category}.ply"
                o3d.io.write_triangle_mesh(p.as_posix(), mesh)


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
