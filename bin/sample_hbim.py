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
        return result.returncode, result.stdout
    except subprocess.CalledProcessError as e:
        print(f"Error running CloudCompare command: {e}")
        print(f"CloudCompare output:\n{e.stdout}")
        print(f"CloudCompare error output:\n{e.stderr}")
        return e.returncode, e.stdout

def process_mesh(mesh_file, output_dir, density, cloudcompare_path, use_gpu, max_threads):
    base_name = os.path.splitext(os.path.basename(mesh_file))[0]
    output_file = f"{output_dir}/{base_name}_sampled.las"
    
    # Prepare acceleration options
    acceleration_options = []
    if use_gpu:
        acceleration_options.append("-USE_GPU")
    if max_threads > 0:
        acceleration_options.extend(["-C_MAX_THREADS", str(max_threads)])
    
    sample_command = [
        "-O", mesh_file,
        "-SAMPLE_MESH", "DENSITY", str(density),
        "-C_EXPORT_FMT", "LAS",
        "-LAS_EXPORT_FORMAT", "1.4",
        "-LAS_EXPORT_POINT_FORMAT", "8",
        "-SAVE_CLOUDS", "FILE", output_file
    ] + acceleration_options
    
    result, _ = run_cloudcompare_command(sample_command, cloudcompare_path)
    
    if result == 0:
        print(f"Successfully processed {mesh_file}")
    else:
        print(f"Error processing {mesh_file}")

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
