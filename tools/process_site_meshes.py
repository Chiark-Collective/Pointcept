import argparse
import vtk
from pathlib import Path
from pointcept.supplemental.mesh_processing import DataHandler, set_data_root
from pointcept.supplemental.fold_allocation import FoldConfiguration, crop_meshes_per_fold, save_fold_meshes

def main(data_root, template_dir, labels=None):

    vtk.vtkSMPTools.SetBackend('STDThread')

    set_data_root(data_root)
    if labels == None:
        labels = ['maritime_museum', 'park_row', 'rog_south', 'brass_foundry']
    print(f"Processing labels: {labels}")
    template_dir = Path(template_dir).absolute()

    for label in labels:
    
        dh = DataHandler(label)
        dh.ensure_meshes()

        # Load the config and generate meshes from it
        config_path = template_dir / f"{label}_v1.pkl"
        fold_config = FoldConfiguration.load(config_path.as_posix())

        fold_rectangles = fold_config.generate_fold_rectangles(combine_subregions=False, plot=False)
        fold_meshes = crop_meshes_per_fold(
            dh.extracted_meshes,
            fold_rectangles,
            fold_config.x_edges,
            fold_config.y_edges,
        )
        save_fold_meshes(dh, fold_meshes)


if __name__ == "__main__":
    # Setup argument parser
    parser = argparse.ArgumentParser(description="Process mesh data and generate outputs.")

    # Adding arguments
    parser.add_argument("--labels", nargs='+', required=False,
                        help="The site labels for which to generate clouds.")
    parser.add_argument("--data-root", type=str, required=True,
                        help="The root directory where the data should be stored.")
    parser.add_argument("--template-dir", type=str, required=True,
                        help="The directory containing the persisted Fold Configuration information.")

    args = parser.parse_args()
    main(args.data_root, args.template_dir, args.labels)