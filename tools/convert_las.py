import argparse
from pointcept.supplemental.converters import las_to_np_pth

def main():
    parser = argparse.ArgumentParser(description="Process LAS files into numpy .pth format.")
    parser.add_argument("input_las_path", type=str, help="Path to the input LAS file.")
    parser.add_argument("category", type=str, help="Category of the input LAS file.")
    parser.add_argument("scene_id", type=str, help="Scene ID of the input LAS file.")
    parser.add_argument("--num_points", type=int, default=None, help="Number of points to process (optional).")
    parser.add_argument("--spoof_normal", action='store_true', default=True, help="Whether to spoof normals (default: True).")
    parser.add_argument("--no_spoof_normal", action='store_false', dest='spoof_normal', help="Do not spoof normals.")
    parser.add_argument("--spoof_gt", action='store_true', default=True, help="Whether to spoof ground truth labels (default: True).")
    parser.add_argument("--no_spoof_gt", action='store_false', dest='spoof_gt', help="Do not spoof ground truth labels.")
    parser.add_argument("--print_contents", action='store_true', default=True, help="Whether to print contents (default: True).")
    parser.add_argument("--no_print_contents", action='store_false', dest='print_contents', help="Do not print contents.")

    args = parser.parse_args()

    # Pass the parsed arguments to the function
    las_to_np_pth(
        input_las_path=args.input_las_path,
        category=args.category,
        scene_id=args.scene_id,
        num_points=args.num_points,
        spoof_normal=args.spoof_normal,
        spoof_gt=args.spoof_gt,
        print_contents=args.print_contents
    )

if __name__ == "__main__":
    main()
