import argparse
from pathlib import Path
from pointcept.supplemental.mesh_processing import DataHandler, MeshAnalyser, MeshSampler, set_data_root

def main(data_root, resolution, library_bin_file=None):
    # Set the root directory for data
    set_data_root(data_root)
    
    # Initialize DataHandler with "library" hardcoded as label
    d = DataHandler("library")
    if library_bin_file:
        d.set_bin_file(library_bin_file)  # Set the custom .bin file if provided

    # First we'll extract the raw meshes from the .bin file
    # and run some preprocessing.
    d.ensure_meshes()

    # Now we'll split and shuffle the meshes randomly
    # then save them under the data root
    analyser = MeshAnalyser(d)
    splits, full_meshes = analyser.generate_library_splits()
    d.save_splits(splits)

    # Generate and save point clouds based on the specified resolution
    d.generate_and_save_fold_clouds(resolution=resolution)

if __name__ == "__main__":
    # Setup argument parser
    parser = argparse.ArgumentParser(description="Process mesh data and generate outputs.")

    # Adding arguments
    parser.add_argument("--data-root", type=str, required=True,
                        help="The root directory where the data should be stored.")
    parser.add_argument("--resolution", type=float, required=True,
                        help="The resolution for generating point clouds.")
    parser.add_argument("--library-bin-file", type=str, default=None,
                        help="Optional path to the .bin file for the library.")

    # Parse arguments
    args = parser.parse_args()

    # Call the main function with parsed arguments
    main(args.data_root, args.resolution, args.library_bin_file)
