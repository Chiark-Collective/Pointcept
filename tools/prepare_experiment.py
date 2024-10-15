import argparse
import os
import shutil
import sys

def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Combine *combined* .pth files from specified labels into a new directory."
    )
    parser.add_argument(
        "-s", "--source_dir",
        required=True,
        help="Path to the source directory containing labels."
    )
    parser.add_argument(
        "-l", "--labels",
        required=True,
        nargs="+",
        help="List of labels to include (e.g., park_row brass_foundry)."
    )
    parser.add_argument(
        "-o", "--output_dir",
        required=True,
        help="Name of the new output directory to create."
    )
    parser.add_argument(
        "-f", "--force",
        action='store_true',
        help="Force scrubbing the output directory without confirmation."
    )
    return parser.parse_args()

def scrub_output_directory(output_dir, force=False):
    """
    Deletes the output directory if it exists.
    If force is False, prompts the user for confirmation.
    """
    if os.path.exists(output_dir):
        if not force:
            response = input(
                f"Output directory '{output_dir}' already exists. Do you want to delete it and continue? [y/N]: "
            ).strip().lower()
            if response != 'y':
                print("Operation aborted by the user.")
                sys.exit(0)
        try:
            shutil.rmtree(output_dir)
            print(f"Existing output directory '{output_dir}' has been removed.")
        except Exception as e:
            print(f"Error removing existing output directory '{output_dir}': {e}")
            sys.exit(1)

def create_directory_structure(output_dir):
    """
    Creates the output directory structure: train, test, eval.
    Assumes that the output directory has already been scrubbed if necessary.
    """
    try:
        os.makedirs(output_dir, exist_ok=True)
        for fold in ['train', 'test', 'eval']:
            fold_path = os.path.join(output_dir, fold)
            os.makedirs(fold_path, exist_ok=True)
        print(f"Created directory structure in '{output_dir}'.")
    except Exception as e:
        print(f"Error creating directory structure: {e}")
        sys.exit(1)

def copy_combined_pth_files(source_dir, labels, output_dir):
    """
    Copies combined *.pth files from specified labels and folds to the output directory.
    """
    folds = ['train', 'test', 'eval']
    for label in labels:
        label_path = os.path.join(source_dir, label)
        if not os.path.isdir(label_path):
            raise FileNotFoundError(f"Label directory '{label_path}' does not exist.")
        
        print(f"Processing label: '{label}'")
        
        for fold in folds:
            fold_path = os.path.join(label_path, fold)
            if not os.path.isdir(fold_path):
                raise FileNotFoundError(f"Fold directory '{fold_path}' does not exist for label '{label}'.")
            
            # Find combined .pth files
            combined_files = [f for f in os.listdir(fold_path) if f.startswith("combined") and f.endswith(".pth")]
            
            if not combined_files:
                raise FileNotFoundError(f"No 'combined*.pth' files found in '{fold_path}' for label '{label}'.")
            
            for file_name in combined_files:
                src_file = os.path.join(fold_path, file_name)
                dst_file = os.path.join(output_dir, fold, file_name)
                try:
                    shutil.copy2(src_file, dst_file)
                    print(f"Copied '{src_file}' to '{dst_file}'.")
                except Exception as e:
                    raise IOError(f"Error copying '{src_file}' to '{dst_file}': {e}")

def rename_eval_to_val(output_dir):
    """
    Renames the 'eval' directory within the output directory to 'val'.
    """
    eval_path = os.path.join(output_dir, 'eval')
    val_path = os.path.join(output_dir, 'val')
    if os.path.isdir(eval_path):
        try:
            os.rename(eval_path, val_path)
            print(f"Renamed 'eval' to 'val' in '{output_dir}'.")
        except Exception as e:
            raise OSError(f"Error renaming 'eval' to 'val': {e}")
    else:
        raise FileNotFoundError(f"'eval' directory does not exist in '{output_dir}'.")

def main():
    args = parse_arguments()
    
    source_dir = args.source_dir
    labels = args.labels
    output_dir = args.output_dir
    force_scrub = args.force

    # Validate source directory
    if not os.path.isdir(source_dir):
        print(f"Error: Source directory '{source_dir}' does not exist.")
        sys.exit(1)
    
    # Scrub the output directory if it exists
    scrub_output_directory(output_dir, force=force_scrub)
    
    # Create output directory structure
    create_directory_structure(output_dir)
    
    try:
        # Copy combined .pth files
        copy_combined_pth_files(source_dir, labels, output_dir)
        
        # Rename eval to val
        rename_eval_to_val(output_dir)
    except (FileNotFoundError, IOError, OSError) as e:
        print(f"Error: {e}")
        print("Scrubbing the output directory due to the error.")
        scrub_output_directory(output_dir, force=True)
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {e}")
        print("Scrubbing the output directory due to the error.")
        scrub_output_directory(output_dir, force=True)
        sys.exit(1)
    
    print("Operation completed successfully.")

if __name__ == "__main__":
    main()
