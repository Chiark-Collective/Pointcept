import argparse
import glob
import os
import sys
import logging
from pathlib import Path
from collections import namedtuple
from typing import List, Optional, Tuple, Dict, Callable

import laspy
import torch
import numpy as np

from pdal_runner import PDALPipelineRunner, PDALException

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants
VALID_LABELS = [
    'park_row', 'maritime_museum', 'rog_north', 'rog_south',
    'library', 'brass_foundry', 'site',
]
EXTRA_DIMS = "NormalX=float64, NormalY=float64, NormalZ=float64"
CATEGORIES = [
    '1_wall', '2_floor', '3_roof', '4_ceiling', '5_footpath', '6_grass',
    '7_column', '8_door', '9_window', '10_stair', '11_railing', '12_rwp', '13_other',
]

Config = namedtuple("Config", [
    "label", "resolution", "scene_capacity", "file_capacity",
    "root_dir", "raw_dir", "base_dir", "merged_filename", "scene_dir", "scene_template", "split_dir", "split_template",
    "pth_dir"
])

class TemplateError(Exception):
    """Custom exception for template-related errors."""
    pass

def run_pdal_pipeline(template_name: str, config: Config, parameters: dict) -> Optional[Path]:
    """
    Run a PDAL pipeline using a specified template.
    Args:
        template_name (str): Name of the template file.
        config (Config): Configuration object containing file paths and settings.
        parameters (dict): Parameters to pass to the PDAL pipeline.
    Returns:
        Optional[Path]: Path to the output file if successful, None otherwise.
    Raises:
        TemplateError: If the template file does not exist.
    """
    pipeline_template_path = Path(__file__).parent / "pipeline_templates" / template_name
    if not pipeline_template_path.exists():
        raise TemplateError(f"Template file does not exist: {pipeline_template_path}")
    
    runner = PDALPipelineRunner(pipeline_template_path, config.root_dir)
    
    try:
        output_file = runner.run(parameters)
        logger.info(f"Ran pipeline with template {template_name}, output file: {output_file}")
        return Path(output_file)
    except PDALException as e:
        logger.error(f"Error running pipeline with template {template_name}: {e}")
        return None

def run_all_categories(config: Config) -> None:
    """
    Process all categories of point cloud data.
    Args:
        config (Config): Configuration object containing file paths and settings.
    """
    for category in CATEGORIES:
        fname = config.raw_dir / f"{category}.las"
        if not fname.exists():
            continue
        
        gt_index = int(category.split('_')[0])
        parameters = {
            "input_file": str(fname),
            "output_file": str(config.base_dir / f"{category}.las"),
            "resolution": config.resolution * 100,
            "gt_index": gt_index,
            "extra_dims": EXTRA_DIMS
        }
        run_pdal_pipeline("process_category.json", config, parameters)

def run_merger_pipeline(config: Config) -> None:
    """
    Merge processed category files into a single file.
    Args:
        config (Config): Configuration object containing file paths and settings.
    """
    input_files = [str(config.base_dir / f"{category}.las") for category in CATEGORIES if (config.base_dir / f"{category}.las").exists()]
    
    parameters = {
        "input_files": input_files,
        "output_file": str(config.merged_filename),
        "extra_dims": EXTRA_DIMS
    }
    
    output_file = run_pdal_pipeline("merge_categories.json", config, parameters)
    if output_file:
        # Remove intermediate files
        for f in input_files:
            os.remove(f)
        logger.info("Removed intermediate split category .las files")

def run_chipper_pipeline(config: Config, input_file: Path, output_template: str, capacity: int) -> Optional[Path]:
    """
    Run the chipper pipeline to split point cloud data.
    Args:
        config (Config): Configuration object containing file paths and settings.
        input_file (Path): Input file path.
        output_template (str): Output file template.
        capacity (int): Capacity for each output file.
    Returns:
        Optional[Path]: Path to the output file if successful, None otherwise.
    """
    parameters = {
        "input_file": str(input_file),
        "output_template": output_template,
        "capacity": capacity,
        "extra_dims": EXTRA_DIMS
    }
    
    return run_pdal_pipeline("chipper.json", config, parameters)

def run_scene_chipper_pipeline(config: Config) -> None:
    """
    Run the scene chipper pipeline to split the merged file into scenes.
    Args:
        config (Config): Configuration object containing file paths and settings.
    """
    output_file = run_chipper_pipeline(config, config.merged_filename, config.scene_template, config.scene_capacity)
    if output_file:
        file_count = len(list(config.scene_dir.glob('*.las')))
        logger.info(f"Scene chipper created {file_count} splits.")

def run_file_chipper_pipeline(config: Config) -> None:
    """
    Run the file chipper pipeline to further split scene files.
    Args:
        config (Config): Configuration object containing file paths and settings.
    """
    scene_files = list(config.scene_dir.glob('scene*.las'))
    
    for f in scene_files:
        output_template = str(config.split_dir / f"{f.stem}_split#.las")
        run_chipper_pipeline(config, f, output_template, config.file_capacity)
    
    file_count = len(list(config.split_dir.glob('*.las')))
    logger.info(f"File chipper created {file_count} splits across all scenes.")

def scene_files_exist(config: Config) -> bool:
    """
    Check if scene files already exist.
    Args:
        config (Config): Configuration object containing file paths and settings.
    Returns:
        bool: True if scene files exist, False otherwise.
    """
    scene_files = list(config.scene_dir.glob('scene*.las'))
    return len(scene_files) > 0


def chunked_las_to_pth(in_f: str, chunk_size: int = 1000000, num_points: Optional[int] = None) -> None:
    """
    Convert a single LAS file to PTH format using chunked reading for memory efficiency.
    Args:
        in_f (str): Input LAS file path.
        chunk_size (int): Number of points to process in each chunk.
        num_points (Optional[int]): Total number of points to process. If None, process all points.
    """
    in_f = Path(in_f)
    output_pth_path = in_f.with_suffix('.pth')
    output_pth_path = output_pth_path.parent / 'pth' / output_pth_path.name
    if num_points is not None:
        output_pth_path = output_pth_path.with_stem(f'{output_pth_path.stem}_n{num_points}')

    logger.info(f"Processing file: {in_f}")
    logger.info(f"Output will be saved to: {output_pth_path}")

    with laspy.open(str(in_f)) as file:
        total_points = file.header.point_count
        max_points = min(num_points or total_points, total_points)
        logger.info(f"Total points in file: {total_points}")
        logger.info(f"Points to be processed: {max_points}")

        # Initialize file objects for each attribute
        coord_file = open(output_pth_path.with_name(f"{output_pth_path.stem}_coord.tmp"), 'wb')
        color_file = open(output_pth_path.with_name(f"{output_pth_path.stem}_color.tmp"), 'wb')
        normal_file = open(output_pth_path.with_name(f"{output_pth_path.stem}_normal.tmp"), 'wb')
        gt_file = open(output_pth_path.with_name(f"{output_pth_path.stem}_gt.tmp"), 'wb')

        processed_points = 0
        for i, points in enumerate(file.chunk_iterator(chunk_size)):
            if processed_points >= max_points:
                break

            logger.info(f"Processing chunk {i+1}")
            
            chunk_size = min(len(points), max_points - processed_points)
            
            coord = np.stack((
                points.x[:chunk_size] - np.min(points.x[:chunk_size]),
                points.y[:chunk_size] - np.min(points.y[:chunk_size]),
                points.z[:chunk_size] - np.min(points.z[:chunk_size])
            ), axis=-1).astype(np.float32)

            color = np.stack((points.red[:chunk_size], points.green[:chunk_size], points.blue[:chunk_size]), axis=-1).astype(np.float32) / 256.0

            normal = np.stack((points.NormalX[:chunk_size], points.NormalY[:chunk_size], points.NormalZ[:chunk_size]), axis=-1)
            norm = np.linalg.norm(normal, axis=1, keepdims=True)
            normal = (normal / norm).astype(np.float32)

            gt = points.gt[:chunk_size].astype(np.int64)

            # Write chunk data to temporary files
            coord_file.write(coord.tobytes())
            color_file.write(color.tobytes())
            normal_file.write(normal.tobytes())
            gt_file.write(gt.tobytes())

            processed_points += chunk_size
            logger.info(f"Processed {processed_points} points so far")

        # Close temporary files
        coord_file.close()
        color_file.close()
        normal_file.close()
        gt_file.close()

        logger.info("Creating final PTH file...")
        
        # Read data back from temporary files and create tensors
        data = {
            'coord': torch.from_numpy(np.fromfile(coord_file.name, dtype=np.float32).reshape(-1, 3)),
            'color': torch.from_numpy(np.fromfile(color_file.name, dtype=np.float32).reshape(-1, 3)),
            'normal': torch.from_numpy(np.fromfile(normal_file.name, dtype=np.float32).reshape(-1, 3)),
            'gt': torch.from_numpy(np.fromfile(gt_file.name, dtype=np.int64)),
            'scene_id': output_pth_path.with_suffix('')
        }

        # Save the final PTH file
        torch.save(data, output_pth_path)
        logger.info(f"Saved {processed_points} points to {output_pth_path}")

        # Clean up temporary files
        for tmp_file in [coord_file.name, color_file.name, normal_file.name, gt_file.name]:
            Path(tmp_file).unlink()
        logger.info("Cleaned up temporary files")

def convert_all_las_to_pth(config: Config) -> None:
    """
    Convert all LAS files to PTH format.
    Args:
        config (Config): Configuration object containing file paths and settings.
    """
    split_files = glob.glob(str(config.split_template))
    for f in split_files:
        chunked_las_to_pth(f, chunk_size=config.file_capacity)

def parse_arguments() -> argparse.Namespace:
    """
    Parse command-line arguments.
    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Process point cloud data using PDAL.")
    parser.add_argument("dataset", type=str, help="The dataset tag to process, e.g., 'park_row'")
    parser.add_argument("resolution", type=float, help="The resolution to use in subsampling.")
    parser.add_argument(
        "--root-dir", type=Path, default=Path(__file__).parent.parent / "data",
        help="The root directory containing the 'raw_clouds' folder. Default is two directories up from the script location, in a 'data' folder."
    )
    parser.add_argument(
        "--file-capacity", type=int, default=500000,
        help="The point capacity for each file used as input to Pointcept, default is 500k."
    )
    parser.add_argument(
        "--scene-capacity", type=int, default=2400000,
        help="The point capacity for each file train/test/val scene, default is 2.4 million."
    )
    parser.add_argument(
        '--merger-only', action='store_true', default=False,
        help='Enable merger only mode (default: %(default)s)'
    )
    return parser.parse_args()

def create_config(args: argparse.Namespace) -> Config:
    """
    Create a Config object from parsed arguments.
    Args:
        args (argparse.Namespace): Parsed command-line arguments.
    Returns:
        Config: Configuration object.
    Raises:
        ValueError: If invalid arguments are provided.
    """
    if args.dataset not in VALID_LABELS:
        raise ValueError(f"Unknown dataset label: '{args.dataset}'.")

    if not isinstance(args.resolution, float) or args.resolution < 0.005:
        raise ValueError(f"Invalid resolution: {args.resolution}. Must be a float >= 0.005.")

    if not isinstance(args.scene_capacity, int) or not isinstance(args.file_capacity, int):
        raise ValueError(f"Invalid capacity values. Both must be integers.")

    root_dir = args.root_dir.resolve()
    raw_dir = root_dir / "raw_clouds" / args.dataset
    base_dir = root_dir / f"processed_clouds/resolution_{args.resolution}/{args.dataset}"
    scene_dir = base_dir / f"scene_capacity_{args.scene_capacity}"
    
    return Config(
        label=args.dataset,
        resolution=args.resolution,
        scene_capacity=args.scene_capacity,
        file_capacity=args.file_capacity,
        root_dir=root_dir,
        raw_dir=raw_dir,
        base_dir=base_dir,
        merged_filename=base_dir / f"{args.dataset}_merged.las",
        scene_dir=scene_dir,
        scene_template=str(scene_dir / "scene#.las"),
        split_dir=scene_dir / f'file_capacity_{args.file_capacity}',
        split_template=str(scene_dir / f'file_capacity_{args.file_capacity}/scene*_split*.las'),
        pth_dir=scene_dir / f'file_capacity_{args.file_capacity}/pth',
    )

def main() -> None:
    """
    Main function to run the point cloud processing pipeline.
    """
    args = parse_arguments()
    config = create_config(args)

    logger.info(f"Processing dataset: {config.label}")
    logger.info(f"Resolution: {config.resolution}")
    logger.info(f"Scene capacity: {config.scene_capacity}")
    logger.info(f"File capacity: {config.file_capacity}")

    os.makedirs(config.base_dir, exist_ok=True)

    try:
        if not config.merged_filename.exists():
            logger.info("Subsampling individual category clouds.")
            run_all_categories(config)
            run_merger_pipeline(config)
        else:
            logger.info(f"Merged .las output already exists for {config.label} at this resolution.")
        
        if args.merger_only:
            logger.info("Merger only mode enabled, exiting now.")
            return

        os.makedirs(config.scene_dir, exist_ok=True)
        os.makedirs(config.pth_dir, exist_ok=True)
        if not scene_files_exist(config):
            run_scene_chipper_pipeline(config)
        else:
            logger.info("Scene files exist for this scene capacity. Skipping to file chipper...")

        run_file_chipper_pipeline(config)

        logger.info("Converting split LAS files to PTH format using chunked processing.")
        convert_all_las_to_pth(config)

        logger.info("Data pipeline complete!")
    except TemplateError as e:
        logger.error(f"Template error: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
