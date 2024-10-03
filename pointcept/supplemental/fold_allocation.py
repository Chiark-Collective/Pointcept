import numpy as np
import random
import heapq
import matplotlib.pyplot as plt
import logging


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

    def run(self):
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

            # Save metrics for this iteration
            iteration_data = {
                'iteration': iteration,
                'equality_score': equality_score,
                'grid': grid.copy(),
                'area_category_counts': {k: v.copy() for k, v in area_category_counts.items()},
                'area_sizes': area_sizes.copy(),
                'area_bounding_boxes': [bb.copy() for bb in area_bounding_boxes]
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
                self.logger.info(f"New best equality score: {equality_score:.4f} at iteration {iteration}")

            # Log counts and percentages per area at DEBUG level
            if self.verbose:
                log_msg = f"Iteration {iteration} Equality Score: {equality_score:.4f}\n"
                log_msg += "Category counts per area:\n"
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
                        percentage = (count / total_cat_count * 100) if total_cat_count > 0 else 0
                        log_msg += f"  {cat}: {int(count)} points ({percentage:.2f}% of total {cat})\n"
                    log_msg += f"  Total points in area: {int(total_area_counts)}\n"
                self.logger.debug(log_msg)

        # After iterations, use the best assignment
        self.grid = self.best_grid
        self.area_category_counts = self.best_area_category_counts
        self.area_sizes = self.best_area_sizes
        self.area_bounding_boxes = self.best_area_bounding_boxes

        self.logger.info(f"\nBest equality score after {self.iterations} iterations: {self.best_equality_score:.4f}")
        self.logger.info(f"Best iteration: {self.best_iteration}")

        # Log the final counts and percentages
        log_msg = "\nFinal category counts per area:\n"
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
                percentage = (count / total_cat_count * 100) if total_cat_count > 0 else 0
                log_msg += f"  {cat}: {int(count)} points ({percentage:.2f}% of total {cat})\n"
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
