import subprocess
import shutil
import logging
import sys
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
