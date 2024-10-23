import argparse
from pointcept.supplemental.combine_results import create_las_with_results

def main():
    parser = argparse.ArgumentParser(description="Create LAS files from scenes and results directories and optionally store them in a specified output directory.")
    parser.add_argument("scenes_dir", type=str, help="Directory containing the scene files.")
    parser.add_argument("results_dir", type=str, help="Directory where the results should be stored.")
    parser.add_argument("--output_dir", type=str, default=None, help="Optional directory where the LAS files should be output.")

    args = parser.parse_args()

    # Pass the parsed arguments to the function
    create_las_with_results(
        scenes_dir=args.scenes_dir,
        results_dir=args.results_dir,
        output_dir=args.output_dir
    )

if __name__ == "__main__":
    main()

