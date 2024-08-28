import subprocess
import os
import glob
import argparse
import sys
import multiprocessing

def run_cloudcompare_command(command, cloudcompare_path):
    try:
        full_command = ["flatpak", "run", cloudcompare_path, "-SILENT"] + command
        print(f"Running command: {' '.join(full_command)}")
        result = subprocess.run(full_command, check=True, capture_output=True, text=True)
        print(f"CloudCompare output:\n{result.stdout}")
        if result.stderr:
            print(f"CloudCompare error output:\n{result.stderr}")
        return result.returncode
    except subprocess.CalledProcessError as e:
        print(f"Error running CloudCompare command: {e}")
        print(f"CloudCompare output:\n{e.stdout}")
        print(f"CloudCompare error output:\n{e.stderr}")
        return e.returncode

def process_mesh_element(mesh_file: str, element: str, output_dir: str, density: float, cloudcompare_path: str, use_gpu: bool, max_threads: int):
    base_name = os.path.splitext(os.path.basename(mesh_file))[0]
    
    # Prepare acceleration options
    acceleration_options = []
    if use_gpu:
        acceleration_options.append("-USE_GPU")
    if max_threads > 0:
        acceleration_options.extend(["-C_MAX_THREADS", str(max_threads)])
    
    # Sample points on mesh element
    sampled_file = f"{output_dir}/{base_name}_{element}_sampled.bin"
    result = run_cloudcompare_command([
        "-O", mesh_file,
        "-EXTRACT_VERTICES", element,
        "-SAMPLE_MESH", "DENSITY", str(density),
        "-C_EXPORT_FMT", "BIN",
        "-SAVE_CLOUDS", "FILE", sampled_file
    ] + acceleration_options, cloudcompare_path)
    
    if result != 0:
        print(f"Error sampling points from {base_name} - {element}. Skipping LAS conversion.")
        return
    
    # Convert to LAS
    las_file = f"{output_dir}/{base_name}_{element}.las"
    result = run_cloudcompare_command([
        "-O", sampled_file,
        "-C_EXPORT_FMT", "LAS",
        "-LAS_EXPORT_FORMAT", "1.4",
        "-LAS_EXPORT_POINT_FORMAT", "8",
        "-SAVE_CLOUDS", "FILE", las_file
    ] + acceleration_options, cloudcompare_path)
    
    if result == 0:
        # Clean up temporary file
        os.remove(sampled_file)
    else:
        print(f"Error converting {base_name} - {element} to LAS format.")

def process_mesh(mesh_file: str, output_dir: str, density: float, cloudcompare_path: str, use_gpu: bool, max_threads: int):
    elements = ['1_WALL', '2_FLOOR', '3_ROOF', '4_CEILING', '5_FOOTPATH', '6_GRASS', 
                '7_COLUMN', '8_DOOR', '9_WINDOW', '10_STAIR', '11_RAILING', '12_RWP', '13_OTHER']
    
    for element in elements:
        print(f"Processing {element}...")
        process_mesh_element(mesh_file, element, output_dir, density, cloudcompare_path, use_gpu, max_threads)

def main():
    parser = argparse.ArgumentParser(description="Automate CloudCompare workflow: process .bin mesh files to .las point clouds.")
    parser.add_argument("--input-dir", default="./data/meshes", help="Input directory containing .bin mesh files")
    parser.add_argument("--output-dir", default="./data/raw_clouds", help="Output directory for processed .las files")
    parser.add_argument("--density", type=float, default=2.0, help="Density for point sampling")
    parser.add_argument("--project-name", help="Specific project to process (omit to process all)")
    parser.add_argument("--cloudcompare-path", default="org.cloudcompare.CloudCompare", help="Flatpak ID for CloudCompare")
    parser.add_argument("--use-gpu", action="store_true", help="Enable GPU acceleration")
    parser.add_argument("--max-threads", type=int, default=multiprocessing.cpu_count(), help="Maximum number of threads to use (default: number of CPU cores)")

    args = parser.parse_args()

    if not os.path.exists(args.input_dir):
        print(f"Error: Input directory '{args.input_dir}' does not exist.")
        return 1

    os.makedirs(args.output_dir, exist_ok=True)

    if args.project_name:
        mesh_files = [f"{args.input_dir}/{args.project_name}.bin"]
    else:
        mesh_files = glob.glob(f"{args.input_dir}/*.bin")

    for mesh_file in mesh_files:
        if not os.path.exists(mesh_file):
            print(f"Warning: Mesh file '{mesh_file}' not found. Skipping.")
            continue

        project_name = os.path.splitext(os.path.basename(mesh_file))[0]
        project_output_dir = f"{args.output_dir}/{project_name}"
        os.makedirs(project_output_dir, exist_ok=True)
        
        print(f"Processing {project_name}...")
        process_mesh(mesh_file, project_output_dir, args.density, args.cloudcompare_path, args.use_gpu, args.max_threads)
        print(f"Finished processing {project_name}")

    print("All processing complete!")

if __name__ == "__main__":
    sys.exit(main())
