import argparse

import vtk

from pathlib import Path
from pointcept.supplemental.mesh_processing import DataHandler, set_data_root


def main(data_root, resolution, poisson_radius, output_format, save_all_formats, labels=None, ):

    vtk.vtkSMPTools.SetBackend('STDThread')
    print("Current SMP Backend:", vtk.vtkSMPTools.GetBackend())
    set_data_root(data_root)
    if labels == None:
        labels = ['maritime_museum', 'park_row', 'rog_south', 'brass_foundry', 'library']
    print(f"Processing labels: {labels}")

    for label in labels:
        dh = DataHandler(label)
        dh.generate_and_save_fold_clouds(
            resolution=resolution,
            poisson_radius=poisson_radius,
            output_format=output_format,
            save_all_formats=save_all_formats
        )


if __name__ == "__main__":
    # Setup argument parser
    parser = argparse.ArgumentParser(description="Process mesh data and generate outputs.")

    # Adding arguments
    parser.add_argument("--labels", nargs='+', required=False,
                        help="The site labels for which to generate clouds.")
    parser.add_argument("--data-root", type=str, required=True,
                        help="The root directory where the data should be stored.")
    parser.add_argument("--resolution", type=float, required=True,
                        help="The resolution for initial point cloud sampling.")
    parser.add_argument("--poisson-radius", type=float, required=True,
                        help="The radius of the poisson disk sampler in the second sampling.")
    parser.add_argument("--output-format", type=str, default='.las', choices=['.las', '.pth'],
                        help="The format for saving the point clouds (default is .las).")
    parser.add_argument("--save-all-formats", action='store_true',
                        help="Flag to save the point clouds in both .las and .pth formats.")

    # Parse arguments
    args = parser.parse_args()

    # Call the main function with parsed arguments
    main(args.data_root, args.resolution, args.poisson_radius, args.output_format, args.save_all_formats, args.labels)
