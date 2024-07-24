import argparse
from pointcept.supplemental.converters import partition_pth_file

def main():
    parser = argparse.ArgumentParser(description="Process .pth files into partitioned voxel directories.")
    parser.add_argument("input_pth_filename", type=str, help="Filename of the input .pth file.")
    parser.add_argument("category", type=str, help="Category of the input .pth file.")
    parser.add_argument("--name_tag", type=str, default=None, help="Optional name tag for the output directory.")
    parser.add_argument("--num_voxels_x", type=int, default=None, help="Number of voxels along the x-axis (optional).")
    parser.add_argument("--num_voxels_y", type=int, default=None, help="Number of voxels along the y-axis (optional).")
    parser.add_argument("--num_voxels_z", type=int, default=None, help="Number of voxels along the z-axis (optional).")
    parser.add_argument("--print_contents", action='store_true', default=True, help="Whether to print the contents of the operation (default: True).")
    parser.add_argument("--no_print_contents", action='store_false', dest='print_contents', help="Do not print the operation contents.")

    args = parser.parse_args()

    # Pass the parsed arguments to the function
    partition_pth_file(
        input_pth_filename=args.input_pth_filename,
        category=args.category,
        name_tag=args.name_tag,
        num_voxels_x=args.num_voxels_x,
        num_voxels_y=args.num_voxels_y,
        num_voxels_z=args.num_voxels_z,
        print_contents=args.print_contents
    )

if __name__ == "__main__":
    main()
