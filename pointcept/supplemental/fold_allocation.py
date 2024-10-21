import numpy as np
import random
import heapq
import matplotlib.pyplot as plt
import logging
import joblib
import itertools
import pyvista as pv
import vtk
from pointcept.supplemental.utils import *
import numpy as np
import heapq
import random
from copy import deepcopy
import logging
from itertools import count
import joblib
from matplotlib.patches import Rectangle
import matplotlib.lines as mlines
from matplotlib.collections import PatchCollection
from matplotlib import cm
from pathlib import Path


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', datefmt='%H:%M:%S')
logger = logging.getLogger(__name__)


class Fold:
    def __init__(self, fold_id, weight, region_count, categories):
        self.fold_id = fold_id
        self.weight = weight
        self.region_count = region_count
        self.categories = categories
        self.order_counter = 1  # Initialize per-region order counter

        # Initialize counts
        self.total_counts = 0
        self.category_counts = {cat: 0 for cat in categories}

        # Regions within the fold
        self.regions = []

        # Intended counts and cell counts will be set later
        self.intended_total_counts = 0
        self.intended_cell_counts = 0

    def add_region(self, region):
        self.regions.append(region)
        region.fold = self
        # Compute intended counts for the region
        region.compute_intended_counts()

    def update_counts(self, counts):
        self.total_counts += counts['total']
        for cat in self.categories:
            self.category_counts[cat] += counts[cat]

    def compute_intended_counts(self, total_counts, total_cells):
        self.intended_total_counts = self.weight * total_counts
        self.intended_cell_counts = self.weight * total_cells

    def reset(self):
        """Called between iterations."""
        self.total_counts = 0
        self.category_counts = {cat: 0 for cat in self.categories}
        for region in self.regions:
            region.reset()


class Region:
    def __init__(self, region_id, categories, weight):
        self.region_id = region_id
        self.categories = categories
        self.fold = None  # Will be set when added to a fold
        self.weight = weight
        
        # Initialize counts
        self.total_counts = 0
        self.category_counts = {cat: 0 for cat in categories}
        self.cell_count = 0  # Number of populated cells in the region

        # Bounding box for compactness calculations
        self.bounding_box = {'min_i': None, 'max_i': None, 'min_j': None, 'max_j': None}

        # Intended counts and cell counts will be set later
        self.intended_total_counts = 0
        self.intended_cell_counts = 0


    def reset(self):
        """Called between iterations."""
        self.order_counter = 1
        self.total_counts = 0
        self.category_counts = {cat: 0 for cat in self.categories}
        self.cell_count = 0
        self.bounding_box = {'min_i': None, 'max_i': None, 'min_j': None, 'max_j': None}

    def update_counts(self, counts):
        self.total_counts += counts['total']
        self.cell_count += counts['cell']
        for cat in self.categories:
            self.category_counts[cat] += counts[cat]

    def compute_intended_counts(self):
        if self.fold:
            self.intended_total_counts = self.fold.intended_total_counts / self.fold.region_count
            self.intended_cell_counts = self.fold.intended_cell_counts / self.fold.region_count
        else:
            raise ValueError("Region is not associated with a fold.")


class FoldConfiguration:
    def __init__(self, iteration, grid, folds, categories, total_category_counts, total_counts,
                 total_populated_cells, x_edges, y_edges, region_cell_order):
        self.iteration = iteration
        self.grid = grid
        self.folds = folds
        self.categories = categories
        self.total_category_counts = total_category_counts
        self.total_counts = total_counts
        self.total_populated_cells = total_populated_cells
        self.x_edges = x_edges
        self.y_edges = y_edges
        self.region_cell_order = region_cell_order
        self.compute_category_percentages()
        self.prepare_summary()

    def compute_category_percentages(self):
        self.fold_percentages = {}
        self.region_percentages = {}

        # Compute fold percentages
        for fold_id, fold in self.folds.items():
            percentages = {}
            for cat in self.categories:
                count = fold.category_counts[cat]
                total_cat_count = self.total_category_counts[cat]
                if total_cat_count > 0:
                    p = count / total_cat_count
                    percentages[cat] = {'percentage': p * 100}
                else:
                    percentages[cat] = {'percentage': 0}
            self.fold_percentages[fold_id] = percentages

            # Compute region percentages within each fold
            for region in fold.regions:
                region_percentages = {}
                for cat in self.categories:
                    count = region.category_counts[cat]
                    total_cat_count = self.total_category_counts[cat]
                    if total_cat_count > 0:
                        p = count / total_cat_count
                        region_percentages[cat] = {'percentage': p * 100}
                    else:
                        region_percentages[cat] = {'percentage': 0}
                self.region_percentages[region.region_id] = region_percentages

    def prepare_summary(self):
        self.summary = {
            'iteration': self.iteration,
            'folds': {}
        }

        for fold_id, fold in self.folds.items():
            fold_data = {
                'fold_id': fold_id,
                'total_counts': fold.total_counts,
                'intended_total_counts': fold.intended_total_counts,
                'category_counts': fold.category_counts.copy(),
                'percentages': self.fold_percentages[fold_id],
                'regions': {}
            }
            for region in fold.regions:
                region_data = {
                    'region_id': region.region_id,
                    'total_counts': region.total_counts,
                    'intended_total_counts': region.intended_total_counts,
                    'category_counts': region.category_counts.copy(),
                    'percentages': self.region_percentages[region.region_id],
                    'cell_count': region.cell_count,
                    'intended_cell_counts': region.intended_cell_counts,
                    'bounding_box': region.bounding_box.copy(),
                }
                fold_data['regions'][region.region_id] = region_data
            self.summary['folds'][fold_id] = fold_data

    def print_summary(self):
        print(f"Iteration: {self.iteration}")
        print("\nFold Summaries:")
        for fold_data in self.summary['folds'].values():
            fold_id = fold_data['fold_id']
            num_regions = len(fold_data['regions'])
            print(f"\nFold {fold_id}:")
            print(f"  Intended total counts: {fold_data['intended_total_counts']:.2f}")
            print(f"  Actual total counts: {fold_data['total_counts']}")
            print(f"  Category Counts:")
            for cat in self.categories:
                count = fold_data['category_counts'][cat]
                percentage_info = fold_data['percentages'][cat]
                percentage = percentage_info['percentage']
                print(f"    {cat}: {count} ({percentage:.2f}% of total {cat})")
            if num_regions == 1:
                # If only one region, append aspect ratio info here
                region_data = next(iter(fold_data['regions'].values()))
                bb = region_data['bounding_box']
                height = bb['max_i'] - bb['min_i'] + 1
                width = bb['max_j'] - bb['min_j'] + 1
                aspect_ratio = max(height / width, width / height) if width > 0 and height > 0 else 1
                print(f"  Aspect Ratio: {aspect_ratio:.2f}")
            else:
                print(f"  Regions:")
                for region_data in fold_data['regions'].values():
                    region_id = region_data['region_id']
                    print(f"    Region {region_id}:")
                    print(f"      Intended total counts: {region_data['intended_total_counts']:.2f}")
                    print(f"      Actual total counts: {region_data['total_counts']}")
                    print(f"      Category Counts:")
                    for cat in self.categories:
                        count = region_data['category_counts'][cat]
                        percentage_info = region_data['percentages'][cat]
                        percentage = percentage_info['percentage']
                        print(f"        {cat}: {count} ({percentage:.2f}% of total {cat})")
                    bb = region_data['bounding_box']
                    height = bb['max_i'] - bb['min_i'] + 1
                    width = bb['max_j'] - bb['min_j'] + 1
                    aspect_ratio = max(height / width, width / height) if width > 0 and height > 0 else 1
                    print(f"      Aspect Ratio: {aspect_ratio:.2f}")

    def extract_scaled_cell_boundaries(self, mask):
        x_edges = self.x_edges
        y_edges = self.y_edges
        boundaries = []
        rows, cols = mask.shape
        for i in range(rows):
            for j in range(cols):
                if mask[i, j] == 1:  # If the cell is part of the region
                    # Top boundary
                    if i == 0 or mask[i-1, j] == 0:
                        boundaries.append([[y_edges[i], x_edges[j]], [y_edges[i], x_edges[j+1]]])
                    # Bottom boundary
                    if i == rows-1 or mask[i+1, j] == 0:
                        boundaries.append([[y_edges[i+1], x_edges[j]], [y_edges[i+1], x_edges[j+1]]])
                    # Left boundary
                    if j == 0 or mask[i, j-1] == 0:
                        boundaries.append([[y_edges[i], x_edges[j]], [y_edges[i+1], x_edges[j]]])
                    # Right boundary
                    if j == cols-1 or mask[i, j+1] == 0:
                        boundaries.append([[y_edges[i], x_edges[j+1]], [y_edges[i+1], x_edges[j+1]]])
        return boundaries
    
    def plot(self, cmap='tab20'):   
        # Plotting the grid and the sub-areas
        fig, ax = plt.subplots(figsize=(10, 8))
        grid_size_y, grid_size_x = self.grid.shape
        x_edges = self.x_edges
        y_edges = self.y_edges
    
        ax.set_xlim(x_edges[0], x_edges[-1])
        ax.set_ylim(y_edges[0], y_edges[-1])
        ax.set_aspect('equal')
        
        # For plotting, we need to know the x and y positions of the grid cells
        x_positions = x_edges[:-1]
        y_positions = y_edges[:-1]
    
        # Build a mapping from fold IDs to region IDs
        fold_to_regions = {}
        for fold_id, fold in self.folds.items():
            fold_to_regions[fold_id] = [region.region_id for region in fold.regions]
    
        # Assign colors to folds
        num_folds = len(fold_to_regions)
        fold_colors = plt.get_cmap(cmap, num_folds*2)
        fold_id_to_color = {}
        for idx, fold_id in enumerate(sorted(fold_to_regions.keys())):
            fold_id_to_color[fold_id] = fold_colors(idx)
    
        # Assign base colors to regions (subregions)
        region_id_to_base_color = {}
        region_id_to_subregion_number = {}
        for fold_id, region_ids in fold_to_regions.items():
            fold_base_color = fold_id_to_color[fold_id]
            num_regions = len(region_ids)
            region_ids_sorted = sorted(region_ids)
            for idx, region_id in enumerate(region_ids_sorted, start=1):
                # Generate desaturated colors for subregions
                amount = 0.01 + 0.3 * (1 - (idx - 1) / max(num_regions - 1, 1))
                region_base_color = desaturate_color(fold_base_color, amount=amount)
                region_id_to_base_color[region_id] = region_base_color
                # Assign subregion number within the fold
                region_id_to_subregion_number[region_id] = idx
    
        # Build mappings for cell orders
        cell_to_order = {}
        for region_id, cell_orders in self.region_cell_order.items():
            for (i, j), global_order, region_order in cell_orders:
                cell_to_order[(i, j)] = {
                    'region_id': region_id,
                    'global_order': global_order,
                    'region_order': region_order
                }
    
        region_id_to_max_order = {}
        for region_id, cell_orders in self.region_cell_order.items():
            region_order_numbers = [region_order for (_, _), _, region_order in cell_orders]
            max_region_order = max(region_order_numbers)
            region_id_to_max_order[region_id] = max_region_order
    
        # Collect patches for each region
        region_patches = {}
    
        # Now plot the grid and collect patches
        width = 0
        height = 0
        for i in range(grid_size_y):
            for j in range(grid_size_x):
                region_id = self.grid[i, j]
                if region_id > 0:
                    x = x_positions[j]
                    y = y_positions[i]
                    width = x_edges[j + 1] - x_edges[j]
                    height = y_edges[i + 1] - y_edges[i]
    
                    # Get the cell order information
                    order_info = cell_to_order.get((i, j), None)
                    if order_info:
                        global_order = order_info['global_order']
                        region_order = order_info['region_order']
                        max_region_order = region_id_to_max_order[region_id]
    
                        # Compute the color based on region order
                        if max_region_order > 1:
                            amount = 0.7 + 0.3 * (1 - (region_order - 1) / (max_region_order - 1))
                        else:
                            amount = 1.0  # Only one cell, use darkest color
    
                        base_color = region_id_to_base_color[region_id]
                        facecolor = lighten_color(base_color, amount=amount)
                    else:
                        # Default color if no order info
                        facecolor = region_id_to_base_color[region_id]
    
                    rect = Rectangle((x, y), width, height)
                    region_patches.setdefault(region_id, []).append((rect, facecolor))
    
                    # Annotate the cell with order numbers
                    if order_info:
                        # Global order number at bottom right
                        text_x = x + width * 0.95
                        text_y = y + height * 0.05
                        ax.text(text_x, text_y, f'{order_info["global_order"]}', ha='right', va='bottom',
                                fontsize=6, color='black', weight='bold',
                                bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.2'))
    
                        # Region order number at top right
                        text_x_top = x + width * 0.05
                        text_y_top = y + height * 0.05
                        # ax.text(text_x_top, text_y_top, f'{order_info["region_order"]}', ha='left', va='bottom',
                        #         fontsize=6, color='black', weight='bold',
                        #         bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.2'))
    
        # Add patches to the plot
        for region_id, patches_and_colors in region_patches.items():
            patches = [p for p, c in patches_and_colors]
            facecolors = [c for p, c in patches_and_colors]
            patch_collection = PatchCollection(patches, facecolors=facecolors,
                                               edgecolor='black', linewidth=0)
            ax.add_collection(patch_collection)

        # Now, for each region, find its perimeter and draw it
        for region_id in region_patches.keys():
            region_mask = (self.grid == region_id).astype(np.uint8)          
            scaled_cell_boundaries = self.extract_scaled_cell_boundaries(region_mask)
            for boundary in scaled_cell_boundaries:
                x_coords = [boundary[0][1], boundary[1][1]]  # Columns (x-axis)
                y_coords = [boundary[0][0], boundary[1][0]]  # Rows (y-axis)
                ax.plot(x_coords, y_coords, color='black', linewidth=2)
        
        # Plot seed cells with a special marker and label them
        seed_cells = {}
        for region_id, cells in self.region_cell_order.items():
            if cells:
                (i, j), _, _ = cells[0]  # The first cell is the seed cell
                seed_cells[region_id] = (i, j)
        for region_id, (i, j) in seed_cells.items():
            x = x_positions[j] + (x_edges[j + 1] - x_edges[j]) / 2
            y = y_positions[i] + (y_edges[i + 1] - y_edges[i]) / 2
            y_upper = y + height * 0.2
            y_lower = y - height * 0.2

            fold_id = self.region_id_to_fold_id(region_id)
            subregion_number = region_id_to_subregion_number[region_id]
            ax.plot(x, y_lower, marker='*', markersize=10, color='red', markeredgecolor='black', markeredgewidth=0.5)
            annotation = fold_id
            if len(fold_to_regions[fold_id]) > 1:
                annotation = f'{fold_id}.{subregion_number}'
            ax.text(x, y_upper, annotation, ha='center', va='center',
                    fontsize=7, color='black', weight='bold',
                    bbox=dict(facecolor='white', alpha=0.5, edgecolor='none'))

        # Create legend for folds
        seed_cell_marker = mlines.Line2D([], [], color='red', marker='*', linestyle='None',
                                 markersize=10, markeredgecolor='black', markeredgewidth=0.5)
        legend_elements = [seed_cell_marker]
        legend_labels = ['Seed Cells']
        for fold_id in sorted(fold_to_regions.keys()):
            fold_color = fold_id_to_color[fold_id]
            fold_patch = plt.Rectangle((0, 0), 1, 1, facecolor=fold_color)
            legend_elements.append(fold_patch)
            legend_labels.append(f'Fold {fold_id}')
        legend = ax.legend(
            legend_elements, legend_labels, loc='upper right',
            bbox_to_anchor=(1.39, 1), borderaxespad=0., frameon=True, fontsize=10,
        )
        legend.get_frame().set_facecolor('white')
        legend.get_frame().set_edgecolor('black')
        legend.get_frame().set_alpha(1.0)

        plt.title('Fold Geometry')
        plt.xlabel('X (m)')
        plt.ylabel('Y (m)')
        plt.tight_layout()
        plt.show()
        return fig, ax

    def region_id_to_fold_id(self, region_id):
        for fold_id, fold in self.folds.items():
            for region in fold.regions:
                if region.region_id == region_id:
                    return fold_id
        return None

    def save(self, filename):
        # Save the FoldConfiguration instance to disk using joblib
        joblib.dump(self, filename)

    @staticmethod
    def load(filename):
        # Load a FoldConfiguration instance from disk using joblib
        return joblib.load(filename)


    def generate_fold_rectangles(self, combine_subregions=True, plot=True):
        """
        Generate rectangles for each fold or its subregions based on the combine_subregions flag.

        Parameters:
            combine_subregions (bool): 
                If True, combine all subregions within a fold into a single set of rectangles.
                If False, process each subregion's rectangles separately.
            plot (bool): Whether to plot the rectangles. Default is True.

        Returns:
            dict: A dictionary containing the combine_subregions flag and fold rectangles.
                  {
                      'combine_subregions': bool,
                      'fold_rectangles': {
                          fold_id: [rect1, rect2, ...],  # If combined
                          fold_id: {region_id: [rect1, rect2, ...], ...},  # If separate
                          ...
                      }
                  }
        """
        fold_rectangles = {}
        grid = self.grid
        folds = self.folds

        for fold_id, fold in folds.items():
            if combine_subregions:
                # Combine all subregions into one binary grid
                region_ids = [region.region_id for region in fold.regions]
                binary_grid = np.isin(grid, region_ids).astype(int)
                rectangles = find_rectangles_by_maximal_rectangle(binary_grid)
                fold_rectangles[fold_id] = rectangles

                # Calculate total perimeter
                perimeter = sum(
                    2 * ((rect['max_row'] - rect['min_row'] + 1) + (rect['max_col'] - rect['min_col'] + 1))
                    for rect in rectangles
                )
                logging.info(f"Fold {fold_id}: Found {len(rectangles)} rectangles with total perimeter {perimeter}")

                if plot:
                    plot_fold_rectangles(binary_grid, rectangles, f"Fold {fold_id} (Combined Subregions)")
            else:
                # Process each subregion separately
                fold_rectangles[fold_id] = {}
                for region in fold.regions:
                    region_id = region.region_id
                    binary_grid = (grid == region_id).astype(int)
                    rectangles = find_rectangles_by_maximal_rectangle(binary_grid)
                    fold_rectangles[fold_id][region_id] = rectangles

                    # Calculate total perimeter
                    perimeter = sum(
                        2 * ((rect['max_row'] - rect['min_row'] + 1) + (rect['max_col'] - rect['min_col'] + 1))
                        for rect in rectangles
                    )
                    logging.info(f"Fold {fold_id}, Region {region_id}: Found {len(rectangles)} rectangles with total perimeter {perimeter}")

                    if plot:
                        plot_fold_rectangles(binary_grid, rectangles, f"Fold {fold_id}, Region {region_id}")

        return {
            'combine_subregions': combine_subregions,
            'fold_rectangles': fold_rectangles
        }


class GridSplitter:
    def __init__(self, counts, x_edges, y_edges, weights, region_counts, iterations=100):
        """
        Initializes the GridSplitter.

        Args:
            counts (dict): Dictionary mapping categories to arrays of counts.
            x_edges (array): Array of x-axis bin edges.
            y_edges (array): Array of y-axis bin edges.
            weights (dict): Dictionary mapping fold IDs to weights.
            region_counts (dict): Dictionary mapping fold IDs to the number of regions in each fold.
            iterations (int): Number of iterations to run the algorithm.
            min_percentage_threshold (float): Minimum percentage threshold for category representation.
            size_penalty_multiplier (float): Multiplier for the size deviation in the priority function.
            aspect_ratio_penalty_multiplier (float): Multiplier for the aspect ratio penalty.
        """
        self.counts = counts
        self.x_edges = x_edges
        self.y_edges = y_edges
        self.weights = weights
        self.region_counts = region_counts
        self.iterations = iterations

        self.min_percentage_threshold = 3.0
        self.size_penalty_multiplier = 1.0
        self.aspect_ratio_penalty_multiplier = 1.0
        self.size_incentive_multiplier = 1000.0
        
        self.categories = list(counts.keys())
        self.grid_size_x = len(x_edges) - 1
        self.grid_size_y = len(y_edges) - 1

        # Initialize total category counts
        self.total_category_counts = {cat: np.sum(counts[cat]) for cat in self.categories}
        self.total_counts = sum(self.total_category_counts.values())

        # Prepare category counts grid
        self.category_counts_grid = np.zeros((self.grid_size_y, self.grid_size_x, len(self.categories)))
        for idx, cat in enumerate(self.categories):
            self.category_counts_grid[:, :, idx] = counts[cat].T

        # Initialize folds
        self.folds = {}
        for fold_id in weights:
            fold = Fold(
                fold_id=fold_id,
                weight=weights[fold_id],
                region_count=region_counts[fold_id],
                categories=self.categories
            )
            self.folds[fold_id] = fold

        # Compute total counts and total populated cells
        self.total_populated_cells = np.count_nonzero(np.sum(self.category_counts_grid, axis=2))

        # Compute intended counts for folds
        for fold in self.folds.values():
            fold.compute_intended_counts(self.total_counts, self.total_populated_cells)

        # Initialize regions within folds
        self.region_id_to_region = {}
        region_id = 1
        for fold in self.folds.values():
            for _ in range(fold.region_count):
                region_weight = fold.weight / fold.region_count
                region = Region(region_id=region_id, categories=self.categories, weight=region_weight)               
                fold.add_region(region)
                self.region_id_to_region[region_id] = region
                region_id += 1

        self.best_equality_score = float('inf')
        self.best_iteration = None
        self.best_configuration = None
        self.logger = logging.getLogger(__name__)
   
    def get_neighbors(self, i, j):
        neighbors = []
        if i > 0:
            neighbors.append((i - 1, j))
        if i < self.grid_size_y - 1:
            neighbors.append((i + 1, j))
        if j > 0:
            neighbors.append((i, j - 1))
        if j < self.grid_size_x - 1:
            neighbors.append((i, j + 1))
        return neighbors

    def assign_unallocated_cells(self, grid):
        """
        Assigns any unallocated cells to the nearest region based on counts and updates counts accordingly.
        """
        grid_size_y, grid_size_x = grid.shape
        unallocated_cells = []

        # Find unallocated cells
        for i in range(grid_size_y):
            for j in range(grid_size_x):
                if grid[i, j] == 0:
                    unallocated_cells.append((i, j))
            
        # For each unallocated cell, find the nearest allocated neighbor
        for cell in unallocated_cells:
            i, j = cell
            neighbors = self.get_neighbors(i, j)
            found = False
            for n in neighbors:
                ni, nj = n
                if grid[ni, nj] != 0:
                    region_id = grid[ni, nj]
                    region = self.region_id_to_region[region_id]
                    fold = region.fold
                    # Assign the cell to this region
                    grid[i, j] = region.region_id
                    cell_counts = self.category_counts_grid[i, j, :]
                    cell_total_counts = np.sum(cell_counts)
                    counts = {
                        'total': cell_total_counts,
                        'cell': 1
                    }
                    for idx, cat in enumerate(self.categories):
                        counts[cat] = cell_counts[idx]
                    region.update_counts(counts)
                    fold.update_counts(counts)
                    # Update bounding box
                    region.bounding_box['min_i'] = min(region.bounding_box['min_i'], i)
                    region.bounding_box['max_i'] = max(region.bounding_box['max_i'], i)
                    region.bounding_box['min_j'] = min(region.bounding_box['min_j'], j)
                    region.bounding_box['max_j'] = max(region.bounding_box['max_j'], j)
                    found = True
                    break  # Stop after assigning to the first found neighbor
            if not found:
                # If no allocated neighbors, assign to the closest region
                min_distance = float('inf')
                closest_region = None
                for region in self.region_id_to_region.values():
                    bb = region.bounding_box
                    if bb['min_i'] is not None and bb['min_j'] is not None:
                        center_i = (bb['min_i'] + bb['max_i']) / 2
                        center_j = (bb['min_j'] + bb['max_j']) / 2
                        distance = (i - center_i) ** 2 + (j - center_j) ** 2
                        if distance < min_distance:
                            min_distance = distance
                            closest_region = region
                if closest_region is not None:
                    grid[i, j] = closest_region.region_id
                    fold = closest_region.fold
                    cell_counts = self.category_counts_grid[i, j, :]
                    cell_total_counts = np.sum(cell_counts)
                    counts = {
                        'total': cell_total_counts,
                        'cell': 1
                    }
                    for idx, cat in enumerate(self.categories):
                        counts[cat] = cell_counts[idx]
                    closest_region.update_counts(counts)
                    fold.update_counts(counts)
                    # Update bounding box
                    closest_region.bounding_box['min_i'] = min(closest_region.bounding_box['min_i'], i)
                    closest_region.bounding_box['max_i'] = max(closest_region.bounding_box['max_i'], i)
                    
    def compute_equality_score(self):
        total_penalty = 0
        all_region_penalties = {}      

        for region_id, region in self.region_id_to_region.items():
            region_penalties = {}
            
            # First, determine a penalty from how much this category deviates from its intended
            # cell area and its overall share of the count across all categories.
            total_count_difference = abs(region.total_counts - region.intended_total_counts)
            total_count_penalty = total_count_difference**0.85
            region_penalties['total_count'] = total_count_penalty

            # Now similar, but for overall region area.
            size_diff = abs(region.cell_count - region.intended_cell_counts)
            size_penalty = (size_diff / self.total_populated_cells) * 1e2  # Adjust the multiplier as needed
            region_penalties['region_area'] = size_penalty
            
            # Now apply penalty terms based on underrepresented or missing categories
            category_penalties = {}
            for cat in self.categories:
                total_count = self.total_category_counts[cat]
                if total_count > 0:
                    count = region.category_counts.get(cat, 0)
                    percentage = (count / total_count) * 100
                    deficit = region.weight * 100 - percentage       
                    penalty = 0
                    if deficit > 0:
                        # Apply a scale factor to prioritise avoiding very low percentages
                        scaling_factor = 1 / max(percentage, 1)
                        penalty += deficit * 1e2 * scaling_factor                        
                        # If below the minimum percentage threshold, apply increasingly harsh penalty scaling
                        if percentage < self.min_percentage_threshold:
                            below_threshold_deficit = self.min_percentage_threshold - percentage
                            penalty = penalty ** below_threshold_deficit
                        # Large penalty if the category is completely missing
                        if percentage == 0:
                            penalty += 1e6
                            
                    category_penalties[cat] = penalty
            region_penalties['per-category'] = category_penalties

            # Now a penalty term based on the region aspect ratio.
            bb = region.bounding_box
            width = bb['max_j'] - bb['min_j'] + 1
            height = bb['max_i'] - bb['min_i'] + 1
            aspect_ratio_penalty = 0
            if width > 0 and height > 0:
                aspect_ratio = max(width / height, height / width)
                aspect_ratio_penalty = (aspect_ratio - 1) ** 2 * 1e3  # Adjust multiplier as needed
            else:
                aspect_ratio_penalty += 1e6  # Large penalty if width or height is zero
            region_penalties['aspect_ratio'] = aspect_ratio_penalty

            all_region_penalties[region_id] = region_penalties
        
        # Now sum all penalty terms
        total_penalty = 0
        for region_penalties in all_region_penalties.values():
            total_penalty += region_penalties['total_count']
            total_penalty += region_penalties['region_area']
            for category_penalty in region_penalties['per-category'].values():
                total_penalty += category_penalty
            total_penalty += region_penalties['aspect_ratio']
            
        return total_penalty, all_region_penalties

    def compute_priority(self, cell, region):
        i, j = cell
        cell_counts = self.category_counts_grid[i, j, :]
        cell_total_counts = np.sum(cell_counts)
    
        # Start with zero priority
        priority = 0
    
        # Calculate the deficit in region size
        region_size_deficit = region.intended_cell_counts - region.cell_count
    
        # Adjust priority based on size deficit
        if region_size_deficit > 0:
            # Region is under its intended size; increase priority
            size_incentive = region_size_deficit / region.intended_cell_counts
            priority += size_incentive * self.size_incentive_multiplier
        else:
            # Region has reached or exceeded intended size; reduce priority
            size_penalty = abs(region_size_deficit) / region.intended_cell_counts
            priority -= size_penalty * self.size_penalty_multiplier
    
        # Category underrepresentation reinforcement (as before)
        for idx, cat in enumerate(self.categories):
            total_cat_count = self.total_category_counts[cat]
            region_cat_count = region.category_counts.get(cat, 0)
            desired_cat_count = region.intended_total_counts * (self.total_category_counts[cat] / self.total_counts)
            cell_cat_count = cell_counts[idx]
    
            remaining = desired_cat_count - region_cat_count
            if remaining > 0 and cell_cat_count > 0:
                # Compute the deficit percentage
                region_percentage = (region_cat_count / total_cat_count) * 100 if total_cat_count > 0 else 0
                deficit = self.min_percentage_threshold - region_percentage
                if deficit > 0:
                    # Increase priority if the category is underrepresented
                    priority += cell_cat_count * (deficit ** 2)  # Square to emphasize underrepresentation
                else:
                    priority += cell_cat_count
    
        # Penalize worsening aspect ratio (as before)
        # Compute new bounding box if the cell is added
        min_i = min(region.bounding_box['min_i'], i) if region.bounding_box['min_i'] is not None else i
        max_i = max(region.bounding_box['max_i'], i) if region.bounding_box['max_i'] is not None else i
        min_j = min(region.bounding_box['min_j'], j) if region.bounding_box['min_j'] is not None else j
        max_j = max(region.bounding_box['max_j'], j) if region.bounding_box['max_j'] is not None else j
    
        width = max_j - min_j + 1
        height = max_i - min_i + 1
    
        # Compute aspect ratio before and after adding the cell
        old_width = region.bounding_box['max_j'] - region.bounding_box['min_j'] + 1 if region.bounding_box['max_j'] is not None else 1
        old_height = region.bounding_box['max_i'] - region.bounding_box['min_i'] + 1 if region.bounding_box['max_i'] is not None else 1
    
        old_aspect_ratio = max(old_width / old_height, old_height / old_width)
        new_aspect_ratio = max(width / height, height / width)
    
        # Penalize if aspect ratio worsens
        if new_aspect_ratio > old_aspect_ratio:
            aspect_ratio_increase = new_aspect_ratio - old_aspect_ratio
            priority -= aspect_ratio_increase * self.aspect_ratio_penalty_multiplier  # Adjust multiplier as needed
    
        # Return negative priority because heapq is a min-heap
        return -priority

    def run(self):
        """
        Runs the grid splitting algorithm across the specified number of iterations.
        """
        self.all_configurations = []
        self.all_seeds = []
        for iteration in range(1, self.iterations + 1):
            self.logger.debug(f"Iteration {iteration}")
    
            # Initialize grid and data structures
            self.grid = np.zeros((self.grid_size_y, self.grid_size_x), dtype=int)
            grid = self.grid  # For convenience

            # Initialize a dictionary to track cell addition order for each region
            region_cell_order = {region.region_id: [] for fold in self.folds.values() for region in fold.regions}
            global_order_counter = 1  # Global counter to assign order numbers
            
            # Reset regions and folds
            for fold in self.folds.values():
                fold.reset()
    
            # Create a list of all populated cells
            populated_cells = [(i, j) for i in range(self.grid_size_y) for j in range(self.grid_size_x)
                               if np.sum(self.category_counts_grid[i, j, :]) > 0]
            if not populated_cells:
                raise ValueError("No populated cells found in the grid.")
    
            # Shuffle the list for randomness
            random.shuffle(populated_cells)
    
            # Assign seed cells to regions
            used_cells = set()
            cell_index = 0
            for fold in self.folds.values():
                for region in fold.regions:
                    while cell_index < len(populated_cells):
                        i, j = populated_cells[cell_index]
                        cell_index += 1
                        if (i, j) not in used_cells:
                            seed_cell = (i, j)
                            used_cells.add((i, j))
                            # Assign cell to region
                            grid[i, j] = region.region_id
                            cell_counts = self.category_counts_grid[i, j, :]
                            counts = {
                                'total': np.sum(cell_counts),
                                'cell': 1
                            }
                            for idx, cat in enumerate(self.categories):
                                counts[cat] = cell_counts[idx]
                            region.update_counts(counts)
                            fold.update_counts(counts)
                            # Update bounding box
                            region.bounding_box['min_i'] = i
                            region.bounding_box['max_i'] = i
                            region.bounding_box['min_j'] = j
                            region.bounding_box['max_j'] = j
                            # Record the seed cell with order number
                            region_cell_order[region.region_id].append(((i, j), global_order_counter, region.order_counter))
                            global_order_counter += 1
                            region.order_counter += 1
                            break
                    else:
                        raise ValueError("Not enough populated cells to assign to all regions.")
            self.all_seeds.append(tuple(used_cells))

            # Initialize priority queue and counter
            heap = []
            counter = count()
    
            # Add neighbors of seed cells to the heap
            for fold in self.folds.values():
                for region in fold.regions:
                    i = region.bounding_box['min_i']
                    j = region.bounding_box['min_j']
                    neighbors = self.get_neighbors(i, j)
                    for n in neighbors:
                        ni, nj = n
                        if grid[ni, nj] == 0:
                            priority = self.compute_priority(
                                cell=(ni, nj),
                                region=region
                            )
                            heapq.heappush(heap, (priority, next(counter), (ni, nj), region))
    
            # Grow regions
            while heap:
                _, _, (i, j), region = heapq.heappop(heap)
                if grid[i, j] != 0:
                    continue  # Skip if already assigned
                grid[i, j] = region.region_id
                fold = region.fold
                cell_counts = self.category_counts_grid[i, j, :]
                counts = {
                    'total': np.sum(cell_counts),
                    'cell': 1
                }
                for idx, cat in enumerate(self.categories):
                    counts[cat] = cell_counts[idx]
                region.update_counts(counts)
                fold.update_counts(counts)
                # Update bounding box
                region.bounding_box['min_i'] = min(region.bounding_box['min_i'], i)
                region.bounding_box['max_i'] = max(region.bounding_box['max_i'], i)
                region.bounding_box['min_j'] = min(region.bounding_box['min_j'], j)
                region.bounding_box['max_j'] = max(region.bounding_box['max_j'], j)
                # Record the cell with order number
                region_cell_order[region.region_id].append(((i, j), global_order_counter, region.order_counter))
                global_order_counter += 1
                region.order_counter += 1
                # Add neighbors to the heap
                neighbors = self.get_neighbors(i, j)
                for n in neighbors:
                    ni, nj = n
                    if grid[ni, nj] == 0:
                        priority = self.compute_priority(
                            cell=(ni, nj),
                            region=region
                        )
                        heapq.heappush(heap, (priority, next(counter), (ni, nj), region))
            
            # Assign unallocated cells
            self.assign_unallocated_cells(grid)
    
            # Compute equality score
            equality_score, equality_breakdown = self.compute_equality_score()
    
            # Create a FoldConfiguration instance for this iteration
            configuration = FoldConfiguration(
                iteration=iteration,
                grid=grid.copy(),
                folds=deepcopy(self.folds),
                categories=self.categories,
                total_category_counts=self.total_category_counts,
                total_counts=self.total_counts,
                total_populated_cells=self.total_populated_cells,
                x_edges=self.x_edges,
                y_edges=self.y_edges,
                region_cell_order=deepcopy(region_cell_order),
            )
            configuration.equality_score = equality_score
            configuration.equality_breakdown = equality_breakdown
            
            self.all_configurations.append(configuration)
            
            # Keep track of the best configuration
            if equality_score < self.best_equality_score:
                self.best_equality_score = equality_score
                self.best_iteration = iteration
                self.best_configuration = configuration
                self.logger.info(f"New best equality score: {equality_score:.4f} at iteration {iteration}")
        
        # Figure out number of unique seed cell configs used
        self.unique_seeds = set(self.all_seeds)
        self.logger.info(f'Number of unique seed cell configurations = {len(self.unique_seeds)}')
        
        # After all iterations, use the best configuration
        self.grid = self.best_configuration.grid
        self.folds = self.best_configuration.folds


################################################################################################################
def maximal_rectangle(matrix):
    if not matrix.any():
        return None
    nrows, ncols = matrix.shape
    max_area = 0
    max_rect = None
    heights = [0] * ncols
    left = [0] * ncols
    right = [ncols] * ncols
    for i in range(nrows):
        cur_left, cur_right = 0, ncols
        # Update heights
        for j in range(ncols):
            if matrix[i][j]:
                heights[j] += 1
            else:
                heights[j] = 0
        # Update left
        for j in range(ncols):
            if matrix[i][j]:
                left[j] = max(left[j], cur_left)
            else:
                left[j] = 0
                cur_left = j + 1
        # Update right
        for j in reversed(range(ncols)):
            if matrix[i][j]:
                right[j] = min(right[j], cur_right)
            else:
                right[j] = ncols
                cur_right = j
        # Compute areas
        for j in range(ncols):
            area = (right[j] - left[j]) * heights[j]
            if area > max_area:
                max_area = area
                min_row = i - heights[j] + 1
                max_row = i
                min_col = left[j]
                max_col = right[j] - 1
                max_rect = {'min_row': min_row, 'max_row': max_row, 'min_col': min_col, 'max_col': max_col}
    return max_rect

def find_rectangles_by_maximal_rectangle(binary_grid):
    rectangles = []
    grid = binary_grid.copy()
    while grid.any():
        rect = maximal_rectangle(grid)
        if rect is None:
            break
        rectangles.append(rect)
        # Remove the rectangle from the grid
        min_row, max_row = rect['min_row'], rect['max_row']
        min_col, max_col = rect['min_col'], rect['max_col']
        grid[min_row:max_row+1, min_col:max_col+1] = 0
    return rectangles

def plot_fold_rectangles(binary_grid, rectangles, title):
    rows, cols = binary_grid.shape
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(binary_grid, cmap='gray', origin='upper')
    # Plot rectangles
    for rect in rectangles:
        min_row, max_row = rect['min_row'], rect['max_row']
        min_col, max_col = rect['min_col'], rect['max_col']
        rect_patch = plt.Rectangle(
            (min_col - 0.5, min_row - 0.5),
            max_col - min_col + 1,
            max_row - min_row + 1,
            edgecolor='red',
            facecolor='none',
            linewidth=1
        )
        ax.add_patch(rect_patch)
    ax.set_title(title)
    ax.set_xlabel('Column Index')
    ax.set_ylabel('Row Index')
    plt.gca().invert_yaxis()
    plt.show()

# def process_folds(best_grid):
#     fold_ids = np.unique(best_grid)
#     fold_ids = fold_ids[fold_ids != 0]  # Exclude 0 if present
#     fold_rectangles = {}
#     for fold_id in fold_ids:
#         print(f"Processing Fold {fold_id}")
#         binary_grid = (best_grid == fold_id).astype(int)
#         rectangles = find_rectangles_by_maximal_rectangle(binary_grid)
#         fold_rectangles[fold_id] = rectangles
#         total_perimeter = sum(2 * ((rect['max_row'] - rect['min_row'] + 1) + (rect['max_col'] - rect['min_col'] + 1))
#                               for rect in rectangles)
#         print(f"Fold {fold_id} has {len(rectangles)} rectangles with total perimeter {total_perimeter}.")
#         plot_fold_rectangles(binary_grid, rectangles, fold_id)
#     return fold_rectangles

# color_names = [
#     "red", "green", "blue", "purple", "cyan", "orange", "yellow",
#     "pink", "gold", "teal", "lightblue", "darkblue", "lightgreen", "crimson",
#     "coral", "limegreen", "peru", "magenta"
# ]


# def get_color_for_fold(fold_id):
#     """
#     Returns a PyVista Color for a given fold ID, cycling through predefined colors.
#     """
#     color_cycle = itertools.cycle(color_names)  # Cycle through color list
#     for _ in range(fold_id):
#         color = next(color_cycle)
#     return pv.Color(color)


def plot_mesh_folds(fold_meshes, backend="static", cmap='turbo'):
    """
    Plots the given mesh folds with unique colors assigned via a colormap.
    All regions within the same fold share the same color. Labels and legends are removed to reduce complexity.

    Args:
        fold_meshes (dict): Dictionary of meshes by fold.
            - If combine_subregions=True:
                {fold_id: {category: cropped_mesh, ...}, ...}
            - If combine_subregions=False:
                {fold_id: {region_id: {category: cropped_mesh, ...}, ...}, ...}
        backend (str): The backend to use for plotting. Either 'trame' or 'static'.
    """
    # Set the PyVista backend
    if backend == 'trame':
        pv.set_jupyter_backend('trame')
        disable_trame_logger()  # Ensure this function is defined elsewhere in your code
    else:
        pv.set_jupyter_backend('static')

    p = pv.Plotter()

    # Collect unique fold_ids
    fold_ids = list(fold_meshes.keys())
    num_folds = len(fold_ids)

    if num_folds == 0:
        logging.warning("No meshes to plot.")
        return

    # Create a colormap with enough distinct colors
    cmap = plt.get_cmap(cmap, num_folds*2)  # 'tab20' provides 20 distinct colors

    # Generate color mapping: fold_id -> color
    color_mapping = {}
    for idx, fold_id in enumerate(fold_ids):
        color = cmap(idx % cmap.N)[:3]  # Extract RGB, ignore alpha
        color_mapping[fold_id] = color

    # Plot each mesh with its assigned color
    for fold_id, data in fold_meshes.items():
        color = color_mapping.get(fold_id, (1.0, 1.0, 1.0))  # Default to white if not found

        if isinstance(data, dict):
            # Determine if data contains categories or regions
            first_inner_value = next(iter(data.values()), None)
            if isinstance(first_inner_value, pv.PolyData):
                # Combined subregions: {category: mesh, ...}
                for category, mesh in data.items():
                    p.add_mesh(mesh, color=color, show_edges=True, opacity=0.55)
            elif isinstance(first_inner_value, dict):
                # Separate subregions: {region_id: {category: mesh, ...}, ...}
                for region_id, categories_dict in data.items():
                    for category, mesh in categories_dict.items():
                        p.add_mesh(mesh, color=color, show_edges=True, opacity=0.55)
            else:
                logging.error(f"Unexpected data format for fold {fold_id}: {data}")
        else:
            logging.error(f"Unexpected data format for fold {fold_id}: {data}")

    # Show axes and grid for reference
    p.show_axes_all()
    p.show_grid(font_size=14)
    if backend == 'static':
        p.window_size = [1200, 800]

    # Render the plot without labels or legends
    p.show(title="Fold Meshes")


def map_grid_to_spatial(min_col, max_col, min_row, max_row, x_edges, y_edges):
    """
    Map grid indices to spatial coordinates using x_edges and y_edges.
    
    Parameters:
        min_col, max_col, min_row, max_row (int): Grid indices.
        x_edges, y_edges (array-like): Bin edges for x and y axes.
    
    Returns:
        tuple: xmin, xmax, ymin, ymax mapped to spatial coordinates.
    """
    xmin = x_edges[min_col]
    xmax = x_edges[max_col + 1]  # +1 because x_edges has length grid_size_x + 1
    ymin = y_edges[min_row]
    ymax = y_edges[max_row + 1]  # +1 because y_edges has length grid_size_y + 1
    return xmin, xmax, ymin, ymax


def crop_meshes_per_fold(category_meshes, fold_rectangles_info, x_edges, y_edges):
    """
    Crop meshes for each fold or subregion based on rectangular bounding boxes using sequential plane clipping.

    Parameters:
        category_meshes (dict): Dictionary of category names and corresponding PyVista meshes.
        fold_rectangles_info (dict): Dictionary containing:
            - 'combine_subregions': bool
            - 'fold_rectangles': dict
        x_edges (array-like): Bin edges for x-axis.
        y_edges (array-like): Bin edges for y-axis.

    Returns:
        dict: A dictionary containing cropped meshes.
              - If combine_subregions=True:
                  {fold_id: {category: cropped_mesh, ...}, ...}
              - If combine_subregions=False:
                  {fold_id: {region_id: {category: cropped_mesh, ...}, ...}, ...}
    """
    combine_subregions = fold_rectangles_info.get('combine_subregions', True)
    fold_rectangles = fold_rectangles_info.get('fold_rectangles', {})
    fold_meshes = {}

    for fold_id, rectangles_or_subregions in fold_rectangles.items():
        if combine_subregions:
            rectangles = rectangles_or_subregions  # List of rectangles for the entire fold
            logging.info(f"Processing Fold {fold_id} with combined subregions")
            fold_category_meshes = {}

            for category, mesh in category_meshes.items():
                mesh = pv.wrap(mesh)  # Ensure the mesh is a PyVista object

                cropped_meshes = []
                for rect in rectangles:
                    # Validate rectangle structure
                    if not isinstance(rect, dict):
                        logging.error(f"Invalid rectangle format in Fold {fold_id}: {rect}")
                        continue

                    # Extract spatial coordinates
                    try:
                        xmin, xmax, ymin, ymax = map_grid_to_spatial(
                            rect['min_col'], rect['max_col'], rect['min_row'], rect['max_row'], x_edges, y_edges)
                    except KeyError as e:
                        logging.error(f"Missing key in rectangle {rect}: {e}")
                        continue

                    # Define clipping planes
                    planes = [
                        ('x', xmin, False),  # Left
                        ('x', xmax, True),   # Right
                        ('y', ymin, False),  # Bottom
                        ('y', ymax, True),   # Top
                    ]

                    clipped_mesh = deepcopy(mesh)  # Start with a fresh copy
                    for axis, origin, invert in planes:
                        normal = {'x': (1, 0, 0), 'y': (0, 1, 0)}[axis]
                        point = [origin, 0, 0] if axis == 'x' else [0, origin, 0]
                        try:
                            clipped_mesh = clipped_mesh.clip(normal=normal, origin=point, invert=invert)
                        except Exception as e:
                            logging.error(f"Clipping error for Fold {fold_id}, Category '{category}', Rectangle {rect}: {e}")
                            clipped_mesh = pv.PolyData()  # Empty mesh

                        if clipped_mesh.n_points == 0:
                            break  # Exit early if no points remain
                    if clipped_mesh.n_points > 0:
                        cropped_meshes.append(clipped_mesh)
                    else:
                        logging.debug(f"Empty mesh after clipping rectangle {rect} for category '{category}' in Fold {fold_id}")

                if cropped_meshes:
                    # Merge all cropped meshes for this category within the fold
                    combined_mesh = cropped_meshes[0]
                    for cm in cropped_meshes[1:]:
                        try:
                            combined_mesh = combined_mesh.merge(cm, merge_points=True, main_has_priority=False)
                        except Exception as e:
                            logging.error(f"Merging error for Fold {fold_id}, Category '{category}': {e}")
                    fold_category_meshes[category] = combined_mesh
                else:
                    logging.warning(f"No meshes found for category '{category}' in Fold {fold_id}")

            fold_meshes[fold_id] = fold_category_meshes
        else:
            # Process each subregion separately
            fold_meshes[fold_id] = {}
            for region_id, rectangles in rectangles_or_subregions.items():
                logging.info(f"Processing Region {region_id} in Fold {fold_id}")
                fold_meshes[fold_id][region_id] = {}

                for category, mesh in category_meshes.items():
                    mesh = pv.wrap(mesh)  # Ensure the mesh is a PyVista object

                    cropped_meshes = []
                    for rect in rectangles:
                        # Validate rectangle structure
                        if not isinstance(rect, dict):
                            logging.error(f"Invalid rectangle format in Fold {fold_id}, Region {region_id}: {rect}")
                            continue

                        # Extract spatial coordinates
                        try:
                            xmin, xmax, ymin, ymax = map_grid_to_spatial(
                                rect['min_col'], rect['max_col'], rect['min_row'], rect['max_row'], x_edges, y_edges)
                        except KeyError as e:
                            logging.error(f"Missing key in rectangle {rect}: {e}")
                            continue

                        # Define clipping planes
                        planes = [
                            ('x', xmin, False),  # Left
                            ('x', xmax, True),   # Right
                            ('y', ymin, False),  # Bottom
                            ('y', ymax, True),   # Top
                        ]

                        clipped_mesh = deepcopy(mesh)  # Start with a fresh copy
                        for axis, origin, invert in planes:
                            normal = {'x': (1, 0, 0), 'y': (0, 1, 0)}[axis]
                            point = [origin, 0, 0] if axis == 'x' else [0, origin, 0]
                            try:
                                clipped_mesh = clipped_mesh.clip(normal=normal, origin=point, invert=invert)
                            except Exception as e:
                                logging.error(f"Clipping error for Fold {fold_id}, Region {region_id}, Category '{category}', Rectangle {rect}: {e}")
                                clipped_mesh = pv.PolyData()  # Empty mesh

                            if clipped_mesh.n_points == 0:
                                break  # Exit early if no points remain
                        if clipped_mesh.n_points > 0:
                            cropped_meshes.append(clipped_mesh)
                        else:
                            logging.debug(f"Empty mesh after clipping rectangle {rect} for category '{category}' in Region {region_id}, Fold {fold_id}")

                    if cropped_meshes:
                        # Merge all cropped meshes for this category within the subregion
                        combined_mesh = cropped_meshes[0]
                        for cm in cropped_meshes[1:]:
                            try:
                                combined_mesh = combined_mesh.merge(cm, merge_points=True, main_has_priority=False)
                            except Exception as e:
                                logging.error(f"Merging error for Fold {fold_id}, Region {region_id}, Category '{category}': {e}")
                        fold_meshes[fold_id][region_id][category] = combined_mesh
                    else:
                        logging.warning(f"No meshes found for category '{category}' in Region {region_id}, Fold {fold_id}")

    return fold_meshes


def save_fold_meshes(dh, splits, fold_to_split=None):
    """
    Save meshes for each fold and its subregions into corresponding split directories with sceneID tags.

    Args:
        dh (DataHandler): 
            An object with a `split_dirs` attribute, which is a dictionary mapping split names 
            (e.g., 'train', 'test', 'eval') to their respective directories (as Path objects).
        splits (dict): 
            A dictionary containing fold-region-category mappings of meshes.
            Format:
            {
                fold_id: {
                    region_id: {
                        category: PolyData, 
                        ...
                    },
                    ...
                },
                ...
            }
        fold_to_split (dict, optional): 
            A dictionary mapping fold IDs to split names.
            Example: {1: 'train', 2: 'test', 3: 'eval'}
            If not provided, defaults to {1: 'train', 2: 'test', 3: 'eval'}.

    Returns:
        None: 
            The function saves the merged meshes directly to disk.
    """
    if fold_to_split is None:
        fold_to_split = {1: 'train', 2: 'test', 3: 'eval'}

    # Ensure that all split directories exist
    dh._ensure_split_dirs()

    for fold_id, regions in splits.items():
        # Map fold_id to split name using the provided mapping
        split_name = fold_to_split.get(fold_id)
        if not split_name:
            logging.warning(f"Fold ID {fold_id} not found in `fold_to_split` mapping. Skipping this fold.")
            continue

        # Retrieve the directory for the current split
        split_dir = dh.split_dirs.get(split_name)
        if not split_dir:
            logging.warning(f"Split directory for '{split_name}' not found in DataHandler. Skipping this split.")
            continue

        # Ensure that split_dir is a Path object
        if not isinstance(split_dir, Path):
            logging.error(f"Split directory for '{split_name}' is not a Path object. Please check DataHandler.")
            continue

        # Iterate over each region within the current fold
        for region_id, categories in regions.items():
            # Iterate over each category within the current region
            for category, mesh in categories.items():
                # Check if the mesh is empty
                if mesh.n_points == 0 and mesh.n_cells == 0:
                    logging.warning(f"Fold: {fold_id}, Region: {region_id}, Category: '{category}' has an empty mesh. Skipping.")
                    continue

                # Optionally set active normals if they exist
                normals = mesh.GetPointData().GetNormals()
                if normals:
                    mesh.GetPointData().SetActiveNormals('Normals')

                # Define the output filename with sceneID tag
                output_filename = f"{category.lower()}_sceneid{region_id}.ply"
                output_file = split_dir / output_filename

                # Initialize the PLY writer
                writer = vtk.vtkPLYWriter()
                writer.SetFileName(str(output_file))
                writer.SetInputData(mesh)
                writer.SetColorModeToDefault()  # Ensure colors are written from Scalars
                writer.SetArrayName('RGB')
                writer.Write()

                logging.info(f"Saved Fold {fold_id}, Region {region_id}, Category '{category}' to {output_file}.")
