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




class GridSplitter:
    def __init__(self, counts, x_edges, y_edges, weights, iterations=100, min_percentage_threshold=5, verbose=True, logger=None):
        # Initialize parameters
        self.counts = counts
        self.x_edges = x_edges
        self.y_edges = y_edges
        self.weights = weights
        self.iterations = iterations
        self.min_percentage_threshold = min_percentage_threshold
        self.verbose = verbose

        # Set up logging
        self.logger = logger or logging.getLogger(__name__)

        # Extract categories and map to indices
        self.categories = list(counts.keys())
        self.num_categories = len(self.categories)
        self.category_to_index = {cat: idx for idx, cat in enumerate(self.categories)}

        # Compute total counts per category
        self.total_category_counts = {cat: np.sum(counts[cat]) for cat in self.categories}

        # Get grid size from counts arrays
        self.grid_size_x, self.grid_size_y = counts[self.categories[0]].shape  # Swap the order

        # Create category_counts_grid of shape (grid_size_y, grid_size_x, num_categories)
        self.category_counts_grid = np.zeros((self.grid_size_y, self.grid_size_x, self.num_categories))

        # Populate category_counts_grid
        for cat in self.categories:
            idx = self.category_to_index[cat]
            self.category_counts_grid[:, :, idx] = counts[cat].T  # Transpose to match dimensions

        # Total number of grid cells
        self.total_cells = self.grid_size_x * self.grid_size_y

        # Desired counts per area
        self.num_areas = len(weights)
        self.desired_category_counts_list = []
        for area_id in range(1, self.num_areas + 1):
            area_weight = weights[area_id - 1]
            desired_category_counts = {cat: area_weight * self.total_category_counts[cat] for cat in self.categories}
            # For last area, adjust desired counts to ensure total counts are correct
            if area_id == self.num_areas:
                for cat in self.categories:
                    desired_category_counts[cat] = self.total_category_counts[cat] - sum(
                        self.desired_category_counts_list[i][cat] for i in range(area_id - 1)
                    )
            self.desired_category_counts_list.append(desired_category_counts)

        # Intended number of cells per area based on weights
        self.intended_area_sizes = [weight * self.total_cells for weight in weights]

        # Variables to keep track of the best assignment
        self.best_equality_score = float('inf')
        self.best_iteration = None
        self.best_grid = None
        self.best_area_category_counts = None
        self.best_area_sizes = None
        self.best_area_bounding_boxes = None
        self.best_category_percentages = None

        # List to store performance metrics per iteration
        self.iteration_metrics = []

        # Optionally store all iterations' data
        self.all_iterations_data = []

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

    def compute_priority(self, cell, area_id, area_category_counts, area_size, area_bounding_box):
        i, j = cell
        cell_counts = self.category_counts_grid[i, j, :]
        priority = 0
        # Compute area size deficit
        area_size_deficit = self.intended_area_sizes[area_id - 1] - area_size
        for idx, cat in enumerate(self.categories):
            total_cat_count = self.total_category_counts[cat]
            area_count = area_category_counts[cat]
            desired_count = self.desired_category_counts_list[area_id - 1][cat]
            cell_count = cell_counts[idx]
            remaining = desired_count - area_count
            if remaining > 0 and cell_count > 0:
                # Compute the deficit percentage
                area_percentage = (area_count / total_cat_count) * 100 if total_cat_count > 0 else 0
                deficit = self.min_percentage_threshold - area_percentage
                if deficit > 0:
                    # Increase priority if the category is underrepresented
                    priority += cell_count * (deficit ** 2)  # Square to emphasize underrepresentation
                else:
                    priority += cell_count
        # Adjust priority based on area size
        if area_size_deficit <= 0:
            # Area has reached or exceeded intended size; reduce priority
            priority -= abs(area_size_deficit) * 1000  # Penalize exceeding size
        # Compute new bounding box if the cell is added
        min_i = min(area_bounding_box['min_i'], i)
        max_i = max(area_bounding_box['max_i'], i)
        min_j = min(area_bounding_box['min_j'], j)
        max_j = max(area_bounding_box['max_j'], j)
        width = max_j - min_j + 1
        height = max_i - min_i + 1
        # Compute aspect ratio before and after adding the cell
        old_width = area_bounding_box['max_j'] - area_bounding_box['min_j'] + 1
        old_height = area_bounding_box['max_i'] - area_bounding_box['min_i'] + 1
        old_aspect_ratio = max(old_width / old_height, old_height / old_width)
        new_aspect_ratio = max(width / height, height / width)
        # Penalize if aspect ratio worsens
        if new_aspect_ratio > old_aspect_ratio:
            aspect_ratio_increase = new_aspect_ratio - old_aspect_ratio
            priority -= aspect_ratio_increase * 1000  # Adjust multiplier as needed
        # Negative priority because heapq is a min-heap and we want higher priority to be popped first
        return -priority

    def assign_unallocated_cells(self, grid):
        unallocated_cells = np.argwhere(grid == 0)
        for cell in unallocated_cells:
            i, j = cell
            # Find neighboring cells that are allocated
            neighbors = self.get_neighbors(i, j)
            neighbor_areas = [grid[n] for n in neighbors if grid[n] > 0]
            if neighbor_areas:
                # Assign to the area with the most neighboring cells
                area_id = max(set(neighbor_areas), key=neighbor_areas.count)
                grid[i, j] = area_id
            else:
                # Assign to a random area
                grid[i, j] = random.randint(1, self.num_areas)

    def compute_equality_score(self, area_category_counts, area_sizes, area_bounding_boxes):
        category_scores = []
        penalty = 0
        for cat in self.categories:
            total_cat_count = self.total_category_counts[cat]
            if total_cat_count > 0:
                percentages = [
                    (area_category_counts[area_id][cat] / total_cat_count) * 100
                    for area_id in area_category_counts
                ]
                # Check for underrepresented categories
                for p in percentages:
                    if p < self.min_percentage_threshold:
                        # Exponential penalty for underrepresentation
                        deficit = self.min_percentage_threshold - p
                        penalty += np.exp(deficit)
                        if p == 0:
                            penalty += 1e6  # Large penalty if category is missing
                variance = np.var(percentages)
                category_scores.append(variance)
        # Penalty for deviation from intended area sizes
        for area_size, intended_size in zip(area_sizes, self.intended_area_sizes):
            size_diff = abs(area_size - intended_size)
            size_penalty = (size_diff / self.total_cells) * 1e5  # Adjust the multiplier as needed
            penalty += size_penalty
        # Penalty for aspect ratios
        for area_bb in area_bounding_boxes:
            width = area_bb['max_j'] - area_bb['min_j'] + 1
            height = area_bb['max_i'] - area_bb['min_i'] + 1
            if width > 0 and height > 0:
                aspect_ratio = max(width / height, height / width)
                aspect_ratio_penalty = (aspect_ratio - 1) ** 2 * 1000  # Adjust multiplier as needed
                penalty += aspect_ratio_penalty
            else:
                penalty += 1e6  # Large penalty if width or height is zero (shouldn't happen)
        # The overall equality score is the mean variance across all categories plus penalties
        equality_score = np.mean(category_scores) + penalty
        return equality_score

    def compute_category_percentages_with_uncertainties(self, area_category_counts):
        category_percentages = {}
        category_uncertainties = {}
        for area_id in area_category_counts:
            category_percentages[area_id] = {}
            category_uncertainties[area_id] = {}
            for cat in self.categories:
                count = area_category_counts[area_id][cat]
                total_cat_count = self.total_category_counts[cat]
                if total_cat_count > 0:
                    percentage = (count / total_cat_count) * 100
                    # Compute binomial uncertainty
                    p = count / total_cat_count
                    n = total_cat_count
                    std_error = np.sqrt(p * (1 - p) / n) * 100  # Convert to percentage
                else:
                    percentage = 0
                    std_error = 0
                category_percentages[area_id][cat] = percentage
                category_uncertainties[area_id][cat] = std_error
        return category_percentages, category_uncertainties

    def run(self, random_seed=4125214):
        random.seed(random_seed)
        for iteration in range(1, self.iterations + 1):
            self.logger.debug(f"Iteration {iteration}")

            # Initialize grid
            grid = np.zeros((self.grid_size_y, self.grid_size_x), dtype=int)

            # Randomly select seed cells for each area
            seed_cells = []
            used_cells = set()
            for area_id in range(1, self.num_areas + 1):
                while True:
                    i = random.randint(0, self.grid_size_y - 1)
                    j = random.randint(0, self.grid_size_x - 1)
                    if (i, j) not in used_cells:
                        seed_cells.append((i, j))
                        used_cells.add((i, j))
                        break

            # Initialize area data
            area_category_counts_list = []
            area_sizes = []
            area_bounding_boxes = []
            for area_id, seed in zip(range(1, self.num_areas + 1), seed_cells):
                area_category_counts_list.append({cat: 0 for cat in self.categories})
                area_sizes.append(0)
                area_bounding_boxes.append({'min_i': seed[0], 'max_i': seed[0], 'min_j': seed[1], 'max_j': seed[1]})
            assigned_cells = set()
            queued_cells = set()

            # Initialize combined priority queue
            heap = []

            # Add seed cells to the heap
            for area_id, seed in zip(range(1, self.num_areas + 1), seed_cells):
                if grid[seed] == 0:
                    grid[seed] = area_id
                    assigned_cells.add(seed)
                    area_sizes[area_id - 1] += 1
                    cell_counts = self.category_counts_grid[seed[0], seed[1], :]
                    for idx, cat in enumerate(self.categories):
                        area_category_counts_list[area_id - 1][cat] += cell_counts[idx]
                    neighbors = self.get_neighbors(seed[0], seed[1])
                    for n in neighbors:
                        if grid[n] == 0 and n not in queued_cells:
                            priority = self.compute_priority(
                                n,
                                area_id,
                                area_category_counts_list[area_id - 1],
                                area_sizes[area_id - 1],
                                area_bounding_boxes[area_id - 1]
                            )
                            heapq.heappush(heap, (priority, n, area_id))
                            queued_cells.add(n)

            # Grow areas simultaneously
            while heap:
                _, current_cell, area_id = heapq.heappop(heap)
                if grid[current_cell] != 0:
                    continue  # Skip if already assigned
                grid[current_cell] = area_id
                assigned_cells.add(current_cell)
                area_sizes[area_id - 1] += 1
                cell_counts = self.category_counts_grid[current_cell[0], current_cell[1], :]
                for idx, cat in enumerate(self.categories):
                    area_category_counts_list[area_id - 1][cat] += cell_counts[idx]
                # Update area bounding box
                area_bb = area_bounding_boxes[area_id - 1]
                area_bb['min_i'] = min(area_bb['min_i'], current_cell[0])
                area_bb['max_i'] = max(area_bb['max_i'], current_cell[0])
                area_bb['min_j'] = min(area_bb['min_j'], current_cell[1])
                area_bb['max_j'] = max(area_bb['max_j'], current_cell[1])
                # Add neighbors to the heap
                neighbors = self.get_neighbors(current_cell[0], current_cell[1])
                for n in neighbors:
                    if grid[n] == 0 and n not in queued_cells:
                        priority = self.compute_priority(
                            n,
                            area_id,
                            area_category_counts_list[area_id - 1],
                            area_sizes[area_id - 1],
                            area_bounding_boxes[area_id - 1]
                        )
                        heapq.heappush(heap, (priority, n, area_id))
                        queued_cells.add(n)

            # Assign any remaining unallocated cells
            self.assign_unallocated_cells(grid)

            # Recalculate area_category_counts and area_sizes
            area_category_counts = {area_id: area_category_counts_list[area_id - 1] for area_id in range(1, self.num_areas + 1)}
            area_sizes = [np.sum(grid == area_id) for area_id in range(1, self.num_areas + 1)]

            # Compute equality score with area size and shape penalty
            equality_score = self.compute_equality_score(
                area_category_counts,
                area_sizes,
                area_bounding_boxes
            )

            # Compute category percentages and uncertainties
            category_percentages, category_uncertainties = self.compute_category_percentages_with_uncertainties(area_category_counts)

            # Save metrics for this iteration
            iteration_data = {
                'iteration': iteration,
                'equality_score': equality_score,
                'grid': grid.copy(),
                'area_category_counts': {k: v.copy() for k, v in area_category_counts.items()},
                'area_sizes': area_sizes.copy(),
                'area_bounding_boxes': [bb.copy() for bb in area_bounding_boxes],
                'category_percentages': {k: v.copy() for k, v in category_percentages.items()},
                'category_uncertainties': {k: v.copy() for k, v in category_uncertainties.items()}
            }
            self.iteration_metrics.append(iteration_data)
            self.all_iterations_data.append(iteration_data)

            # Update best assignment if current one is better
            if equality_score < self.best_equality_score:
                self.best_equality_score = equality_score
                self.best_iteration = iteration
                self.best_grid = grid.copy()
                self.best_area_category_counts = {k: v.copy() for k, v in area_category_counts.items()}
                self.best_area_sizes = area_sizes.copy()
                self.best_area_bounding_boxes = [bb.copy() for bb in area_bounding_boxes]
                self.best_category_percentages = {k: v.copy() for k, v in category_percentages.items()}
                self.best_category_uncertainties = {k: v.copy() for k, v in category_uncertainties.items()}
                self.logger.info(f"New best equality score: {equality_score:.4f} at iteration {iteration}")

            # Log counts and percentages per area at DEBUG level
            if self.verbose:
                log_msg = f"Iteration {iteration} Equality Score: {equality_score:.4f}\n"
                log_msg += "Category counts and percentages per area:\n"
                for area_id in range(1, self.num_areas + 1):
                    log_msg += f"Area {area_id}:\n"
                    total_area_counts = sum(area_category_counts[area_id][cat] for cat in self.categories)
                    area_size = area_sizes[area_id - 1]
                    intended_size = self.intended_area_sizes[area_id - 1]
                    size_percentage = (area_size / self.total_cells) * 100
                    intended_percentage = (intended_size / self.total_cells) * 100
                    log_msg += f"  Area size: {area_size} cells ({size_percentage:.2f}% of total, intended {intended_percentage:.2f}%)\n"
                    for cat in self.categories:
                        count = area_category_counts[area_id][cat]
                        total_cat_count = self.total_category_counts[cat]
                        percentage = category_percentages[area_id][cat]
                        uncertainty = category_uncertainties[area_id][cat]
                        log_msg += f"  {cat}: {int(count)} points ({percentage:.2f}% ± {uncertainty:.2f}% of total {cat})\n"
                    log_msg += f"  Total points in area: {int(total_area_counts)}\n"
                self.logger.debug(log_msg)

        # After iterations, use the best assignment
        self.grid = self.best_grid
        self.area_category_counts = self.best_area_category_counts
        self.area_sizes = self.best_area_sizes
        self.area_bounding_boxes = self.best_area_bounding_boxes
        self.category_percentages = self.best_category_percentages
        self.category_uncertainties = self.best_category_uncertainties

        self.logger.info(f"\nBest equality score after {self.iterations} iterations: {self.best_equality_score:.4f}")
        self.logger.info(f"Best iteration: {self.best_iteration}")

        # Log the final counts and percentages
        log_msg = "\nFinal category counts and percentages per area:\n"
        for area_id in range(1, self.num_areas + 1):
            log_msg += f"\nArea {area_id}:\n"
            total_area_counts = sum(self.area_category_counts[area_id][cat] for cat in self.categories)
            area_size = self.area_sizes[area_id - 1]
            intended_size = self.intended_area_sizes[area_id - 1]
            size_percentage = (area_size / self.total_cells) * 100
            intended_percentage = (intended_size / self.total_cells) * 100
            log_msg += f"  Area size: {area_size} cells ({size_percentage:.2f}% of total, intended {intended_percentage:.2f}%)\n"
            for cat in self.categories:
                count = self.area_category_counts[area_id][cat]
                total_cat_count = self.total_category_counts[cat]
                percentage = self.category_percentages[area_id][cat]
                uncertainty = self.category_uncertainties[area_id][cat]
                log_msg += f"  {cat}: {int(count)} points ({percentage:.2f}% ± {uncertainty:.2f}% of total {cat})\n"
            log_msg += f"  Total points in area: {int(total_area_counts)}\n"
        self.logger.info(log_msg)

    def plot(self):
        # Plotting the grid and the sub-areas
        fig, ax = plt.subplots()
        ax.set_xlim(self.x_edges[0], self.x_edges[-1])
        ax.set_ylim(self.y_edges[0], self.y_edges[-1])
        ax.set_aspect('equal')

        cell_widths = self.x_edges[1:] - self.x_edges[:-1]  # Length: grid_size_x
        cell_heights = self.y_edges[1:] - self.y_edges[:-1]  # Length: grid_size_y

        # For plotting, we need to know the x and y positions of the grid cells
        x_positions = self.x_edges[:-1]  # Length: grid_size_x
        y_positions = self.y_edges[:-1]  # Length: grid_size_y

        for i in range(self.grid_size_y):      # i from 0 to grid_size_y - 1
            for j in range(self.grid_size_x):  # j from 0 to grid_size_x - 1
                area_id = self.grid[i, j]
                if area_id > 0:
                    x = x_positions[j]
                    y = y_positions[i]
                    width = cell_widths[j]
                    height = cell_heights[i]
                    ax.add_patch(plt.Rectangle((x, y), width, height, facecolor=f'C{area_id}',
                                               edgecolor='black', linewidth=0.5))

        plt.title('Contiguous Sub-Areas with Balanced Category Representation and Compact Shapes')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.show()


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

def plot_fold_rectangles(binary_grid, rectangles, fold_id):
    rows, cols = binary_grid.shape
    fig, ax = plt.subplots(figsize=(3, 3))
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
    ax.set_title(f'Fold {fold_id} with Rectangles')
    ax.set_xlabel('Column Index')
    ax.set_ylabel('Row Index')
    plt.gca().invert_yaxis()
    plt.show()

def process_folds(best_grid):
    fold_ids = np.unique(best_grid)
    fold_ids = fold_ids[fold_ids != 0]  # Exclude 0 if present
    fold_rectangles = {}
    for fold_id in fold_ids:
        print(f"Processing Fold {fold_id}")
        binary_grid = (best_grid == fold_id).astype(int)
        rectangles = find_rectangles_by_maximal_rectangle(binary_grid)
        fold_rectangles[fold_id] = rectangles
        total_perimeter = sum(2 * ((rect['max_row'] - rect['min_row'] + 1) + (rect['max_col'] - rect['min_col'] + 1))
                              for rect in rectangles)
        print(f"Fold {fold_id} has {len(rectangles)} rectangles with total perimeter {total_perimeter}.")
        plot_fold_rectangles(binary_grid, rectangles, fold_id)
    return fold_rectangles

color_names = [
    "red", "green", "blue", "purple", "cyan", "orange", "yellow",
    "pink", "gold", "teal", "lightblue", "darkblue", "lightgreen", "crimson",
    "coral", "limegreen", "peru", "magenta"
]


# Cycle through colors to handle arbitrary fold IDs
def get_color_for_fold(fold_id):
    """
    Returns a PyVista Color for a given fold ID, cycling through predefined colors.
    """
    color_cycle = itertools.cycle(color_names)  # Cycle through color list
    for _ in range(fold_id):
        color = next(color_cycle)
    return pv.Color(color)


# Function to plot the mesh folds using either 'trame' or 'static' backends
def plot_mesh_folds(fold_meshes, backend="static"):
    """
    Plots the given mesh folds with the option to use the 'trame' or 'static' PyVista backend.

    Args:
        fold_meshes (dict): Dictionary of meshes by fold.
        backend (str): The backend to use for plotting. Either 'trame' or 'static'.
    """
    if backend == 'trame':
        pv.set_jupyter_backend('trame')
        disable_trame_logger()
        notebook = True
    else:
        pv.set_jupyter_backend('static')
        notebook = True

    p = pv.Plotter() #notebook=notebook)
    
    for fold_id, category_meshes in fold_meshes.items():
        color = get_color_for_fold(fold_id)
        for category, mesh in category_meshes.items():
            p.add_mesh(mesh, color=color, show_edges=True, opacity=0.55)
    
    # Show axes and grid
    p.show_axes_all()
    p.show_grid(font_size=14)
    if backend == 'static':
        p.window_size = [1200, 800]

    # Show the plot
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

def crop_meshes_per_fold(category_meshes, fold_rectangles, x_edges, y_edges):
    """
    Crop meshes for each fold based on rectangular bounding boxes using sequential plane clipping.

    Parameters:
        category_meshes (dict): Dictionary of category names and corresponding PyVista meshes.
        fold_rectangles (dict): Dictionary of fold IDs and bounding boxes.
        x_edges (array-like): Bin edges for x-axis.
        y_edges (array-like): Bin edges for y-axis.

    Returns:
        dict: A dictionary containing cropped meshes for each fold and category.
    """
    fold_meshes = {}
    for fold_id, rectangles in fold_rectangles.items():
        logging.info(f"Processing Fold {fold_id}")
        
        # For each category, we'll create a list to store cropped meshes
        fold_category_meshes = {}
        
        for category, mesh in category_meshes.items():
            mesh = pv.wrap(mesh)  # Ensure the mesh is a PyVista object
            
            # Initialize an empty list to collect cropped meshes for this category and fold
            cropped_meshes = []
            
            for rect in rectangles:
                # Map grid indices to spatial coordinates
                xmin, xmax, ymin, ymax = map_grid_to_spatial(
                    rect['min_col'], rect['max_col'], rect['min_row'], rect['max_row'], x_edges, y_edges)
                
                # Create planes for clipping
                planes = [
                    ('x', xmin, False),  # Left plane
                    ('x', xmax, True),   # Right plane
                    ('y', ymin, False),  # Bottom plane
                    ('y', ymax, True),   # Top plane
                ]
                
                # Start with the original mesh and sequentially clip with each plane
                clipped_mesh = mesh
                for axis, origin, invert in planes:
                    normal = {'x': (1, 0, 0), 'y': (0, 1, 0)}[axis]
                    point = [origin, 0, 0] if axis == 'x' else [0, origin, 0]
                    clipped_mesh = clipped_mesh.clip(normal=normal, origin=point, invert=invert)
                    if clipped_mesh.n_points == 0:
                        break  # No points left, exit early
                if clipped_mesh.n_points > 0:
                    cropped_meshes.append(clipped_mesh)
                else:
                    logging.debug(f"Cropping resulted in empty mesh for rectangle {rect} in category {category}")
                
            # Merge the cropped meshes for this category and fold
            if cropped_meshes:
                combined_mesh = cropped_meshes[0]
                for cm in cropped_meshes[1:]:
                    combined_mesh = combined_mesh.merge(cm, merge_points=True, main_has_priority=False)
                fold_category_meshes[category] = combined_mesh
            else:
                logging.warning(f"No mesh data in Fold {fold_id} for category {category}")
        
        # Store the combined meshes per fold
        fold_meshes[fold_id] = fold_category_meshes

    return fold_meshes


def save_splits(dh, splits):
    """
    Merge cells for each fold-category permutation and save them to disk.

    Args:
        dh (DataHandler): An object with a split_dirs attribute, containing directories for 'train', 'test', 'eval'.
        splits (dict): A dictionary containing fold-category mappings of meshes. 
    Returns:
        None: The function saves the merged meshes directly to disk.
    """
    dh._ensure_split_dirs()
    # Iterate over the splits dictionary (e.g., 'train', 'test', 'eval')
    for fold, categories in splits.items():
        # Get the appropriate directory for the fold from DataHandler
        fold_dir = dh.split_dirs.get(fold)
        cell_counter = 0
        # Iterate over each category in the fold
        for category, mesh in categories.items():
            if mesh.n_points == 0 and combined_mesh.n_cells == 0:
                logger.warn(f"Fold: {fold}, Category: {category} has an empty mesh!")
            mesh.GetPointData().SetActiveNormals('Normals')

            output_file = fold_dir / f"{category.lower()}.ply"           
            # Save the merged mesh to disk using vtk for control over color storage
            writer = vtk.vtkPLYWriter()
            writer.SetFileName(output_file.as_posix())
            writer.SetInputData(mesh)
            writer.SetColorModeToDefault()  # Ensure colors are written from the Scalars
            writer.SetArrayName('RGB')
            writer.Write()
            logger.info(f"Fold {fold}, category {category} saved.")
