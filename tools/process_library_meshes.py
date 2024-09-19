import argparse
from pathlib import Path
from pointcept.supplemental.mesh_processing import DataHandler, MeshAnalyser, set_data_root, merge_and_save_cells

def main(data_root, library_bin_file=None, cell_width=2.0):
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
    splits = analyser.generate_library_splits(cell_width=cell_width)
    merge_and_save_cells(d, splits)


if __name__ == "__main__":
    # Setup argument parser
    parser = argparse.ArgumentParser(description="Process mesh data and generate outputs.")

    # Adding arguments
    parser.add_argument("--data-root", type=str, required=True,
                        help="The root directory where the data should be stored.")
    parser.add_argument("--library-bin-file", type=str, default=None,
                        help="Optional path to the .bin file for the library.")
    parser.add_argument("--cell-width", type=float, required=False,
                        help="The cell width to use when splitting and jumbling the library data.")
    # Parse arguments
    args = parser.parse_args()

    # Call the main function with parsed arguments
    main(args.data_root, args.library_bin_file, args.cell_width)
