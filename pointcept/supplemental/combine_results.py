import os
from pathlib import Path
import numpy as np
import torch
import laspy
import logging
import shutil

# Set up logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', datefmt='%H:%M:%S')
logger = logging.getLogger(__name__)


def create_las_with_results(scenes_dir, results_dir, output_dir=None):
    """
    Create LAS and .pth files by combining scene data and predictions.

    Args:
        scenes_dir (str or Path): Directory containing scene .pth files.
        results_dir (str or Path): Directory containing prediction .npy files.
        output_dir (str or Path, optional): Directory to save output files.
            Defaults to "full_data" subdirectory within results_dir.

    The function reads scene data and corresponding predictions, merges them, and writes out .las and .pth files.
    If the output directory exists, it will be cleaned (all contents deleted). Otherwise, it will be created.
    """

    scenes_dir = Path(scenes_dir)
    results_dir = Path(results_dir)

    # Default output directory to "full_data" within results_dir
    if output_dir is None:
        output_dir = results_dir / 'full_data'
    else:
        output_dir = Path(output_dir)

    # Clean the output directory if it exists, or create it if it doesn't
    clean_and_create_output_dir(output_dir)

    logger.info(f"Processing scenes from: {scenes_dir}")
    logger.info(f"Using predictions from: {results_dir}")
    logger.info(f"Output directory: {output_dir}")

    # Load predictions
    predictions = load_predictions(results_dir)
    logger.info(f"Loaded predictions for {len(predictions)} scenes.")

    # Load scenes
    scenes = load_scenes(scenes_dir)
    logger.info(f"Loaded {len(scenes)} scenes.")

    # Process each scene
    for scene_id, pred in predictions.items():
        if scene_id in scenes:
            scene_data = scenes[scene_id]
            logger.info(f"Processing scene: {scene_id}")
            process_scene(scene_id, scene_data, pred, output_dir)
        else:
            logger.warning(f"Scene data for {scene_id} not found in scenes directory.")


def clean_and_create_output_dir(output_dir):
    """
    Cleans the output directory if it exists (deletes all contents),
    or creates it if it does not exist.

    Args:
        output_dir (Path): The directory to clean and/or create.
    """
    if output_dir.exists():
        # Confirm that the directory is safe to delete
        if output_dir.is_dir():
            logger.info(f"Cleaning output directory: {output_dir}")
            # Delete all contents inside the directory
            for item in output_dir.iterdir():
                try:
                    if item.is_file() or item.is_symlink():
                        item.unlink()
                    elif item.is_dir():
                        shutil.rmtree(item)
                    logger.debug(f"Deleted {item}")
                except Exception as e:
                    logger.error(f"Failed to delete {item}. Reason: {e}")
        else:
            raise ValueError(f"Output path {output_dir} exists and is not a directory.")
    else:
        output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Created output directory: {output_dir}")


def load_predictions(results_dir):
    """
    Load prediction files from the results directory.

    Args:
        results_dir (Path): Directory containing prediction .npy files.

    Returns:
        dict: A dictionary mapping scene IDs to prediction arrays.
    """
    predictions = {}
    for pred_file in results_dir.glob("*.npy"):
        filename = pred_file.stem  # Filename without extension
        if filename.endswith('_pred'):
            scene_id = filename[:-5]  # Remove '_pred' suffix
            predictions[scene_id] = np.load(pred_file)
            logger.debug(f"Loaded predictions for scene: {scene_id}")
    return predictions


def load_scenes(scenes_dir):
    """
    Load scene files from the scenes directory.

    Args:
        scenes_dir (Path): Directory containing scene .pth files.

    Returns:
        dict: A dictionary mapping scene IDs to scene data.
    """
    scenes = {}
    for scene_file in scenes_dir.glob("*.pth"):
        scene_id = scene_file.stem  # Filename without extension
        scenes[scene_id] = torch.load(scene_file)
        logger.debug(f"Loaded scene data for scene: {scene_id}")
    return scenes


def process_scene(scene_id, scene_data, pred, output_dir):
    """
    Process a single scene and write out the LAS and .pth files.

    Args:
        scene_id (str): Scene identifier.
        scene_data (dict): Scene data containing point coordinates, colors, labels, etc.
        pred (np.array): Prediction array for the scene.
        output_dir (Path): Directory to save output files.
    """
    logger.info(f"Processing scene: {scene_id}")
    # Extract data from scene_data
    points = scene_data.get('coord')  # Assuming scene_data has 'coord' key for point coordinates
    colors = scene_data.get('color')  # Assuming scene_data has 'color' key
    normals = scene_data.get('normal')  # Assuming scene_data has 'normal' key
    gt_labels = scene_data.get('gt')  # Ground truth labels

    # Ensure data is in numpy arrays
    points = np.asarray(points)
    colors = np.asarray(colors)
    normals = np.asarray(normals)
    gt_labels = np.asarray(gt_labels)
    pred_labels = np.asarray(pred)

    # Check that the number of points matches
    num_points = points.shape[0]
    if pred_labels.shape[0] != num_points:
        logger.warning(f"Prediction length {pred_labels.shape[0]} does not match number of points {num_points} for scene {scene_id}.")
        return

    # Prepare data dictionary for .pth file
    data = {
        'coord': points.astype(np.float32),
        'color': colors.astype(np.float32),
        'scene_id': scene_id
    }

    if normals is not None:
        data['normal'] = normals.astype(np.float32)

    if gt_labels is not None:
        data['gt'] = gt_labels.astype(np.int64)

    # Add predicted labels
    data['pred'] = pred_labels.astype(np.int64)

    # Save .pth file
    pth_out = output_dir / f"{scene_id}.pth"
    torch.save(data, pth_out.as_posix())
    logger.info(f"Saved .pth file: {pth_out}")

    # Prepare LAS header
    header = laspy.LasHeader(version="1.4", point_format=8)
    header.x_scale = 0.001  # Adjust the scale to match the units of your points
    header.y_scale = 0.001
    header.z_scale = 0.001
    header.x_offset = np.min(points[:, 0])
    header.y_offset = np.min(points[:, 1])
    header.z_offset = np.min(points[:, 2])

    las = laspy.LasData(header)

    # Rescale colors to LAS format
    las_colors = np.round(colors * (65535 / 255)).astype(np.uint16)

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
    las['NormalX'] = normals[:, 0]
    las['NormalY'] = normals[:, 1]
    las['NormalZ'] = normals[:, 2]

    # Add ground truth labels as extra dimension
    las.add_extra_dim(laspy.ExtraBytesParams(name="gt", type=np.uint8, description="Ground truth labels"))
    las['gt'] = gt_labels.astype(np.uint8)

    # Add predicted labels as extra dimension
    las.add_extra_dim(laspy.ExtraBytesParams(name="pred", type=np.uint8, description="Predicted labels"))
    las['pred'] = pred_labels.astype(np.uint8)

    # Write LAS file
    las_out = output_dir / f"{scene_id}.las"
    las.write(las_out.as_posix())
    logger.info(f"Saved LAS file: {las_out}")
