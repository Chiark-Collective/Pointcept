#!/usr/bin/env python3
"""
Semantic Segmentation Metrics Calculator

This script processes `.pth` files containing both ground truth and prediction data.
It computes various metrics, generates visualizations, and saves the results in a structured `results` directory.

Usage:
    python semantic_segmentation_metrics.py -i /path/to/input_dir -o /path/to/output_dir
    python semantic_segmentation_metrics.py -i /path/to/input_dir -f file1.pth file2.pth --no-interactive
    python semantic_segmentation_metrics.py -i /path/to/input_dir -v
"""
import argparse
import logging
import sys
from pathlib import Path
from typing import List, Optional, Dict, Any

import inquirer
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import torch
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

# Configure logger
logger = logging.getLogger("SemanticSegmentationMetrics")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler(sys.stdout)
formatter = logging.Formatter('[%(levelname)s] %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

# Define class labels starting at 1
CLASS_LABELS = pd.Series({
    1: "wall",
    2: "floor",
    3: "roof",
    4: "ceiling",
    5: "footpath",
    6: "grass",
    7: "column",
    8: "door",
    9: "window",
    10: "stair",
    11: "railing",
    12: "rainwater_pipe",
    13: "other"
})

def parse_arguments():
    parser = argparse.ArgumentParser(description="Semantic Segmentation Metrics Calculator")
    parser.add_argument(
        '-i', '--input_dir',
        type=Path,
        required=True,
        help="Path to the input directory containing .pth files."
    )
    parser.add_argument(
        '-o', '--output_dir',
        type=Path,
        default=None,
        help="Path to the output directory. Defaults to 'results' within the input directory."
    )
    parser.add_argument(
        '-f', '--files',
        nargs='+',
        type=str,
        help="Specific .pth files to process."
    )
    parser.add_argument(
        '--no-interactive',
        action='store_true',
        help="Disable interactive file selection and process all or specified files."
    )
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help="Enable verbose logging."
    )
    return parser.parse_args()

def setup_logging(verbose: bool, output_dir: Path):
    if verbose:
        logger.setLevel(logging.DEBUG)
        logger.debug("Verbose logging enabled.")
    else:
        logger.setLevel(logging.INFO)
    
    # File handler for log file
    file_handler = logging.FileHandler(output_dir / 'processing.log')
    file_handler.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(message)s'))
    logger.addHandler(file_handler)

def abbreviate_number(num: int) -> str:
    """Convert number to abbreviated string (e.g., 1234567 to '1.23M')"""
    if num >= 1e6:
        return f'{num/1e6:.2f}M'
    elif num >= 1e3:
        return f'{num/1e3:.1f}k'
    else:
        return str(int(num))

def interactive_file_selection(pth_files: List[str]) -> List[str]:
    questions = [
        inquirer.Checkbox(
            'selected_files',
            message="Select the .pth files to process",
            choices=pth_files
        )
    ]
    answers = inquirer.prompt(questions)
    return answers['selected_files'] if answers else []

def load_pth_data(pth_path: Path) -> Optional[Dict[str, np.ndarray]]:
    """
    Load ground truth and predictions from a .pth file.
    Handles both PyTorch tensors and NumPy arrays.

    :param pth_path: Path to the .pth file.
    :return: Dictionary with 'y_true' and 'y_pred' NumPy arrays or None if loading fails.
    """
    try:
        data = torch.load(pth_path, map_location='cpu')
        y_true = data.get('gt')
        y_pred = data.get('pred')
        if y_true is None or y_pred is None:
            logger.error(f"Missing 'gt' or 'pred' in {pth_path.name}")
            return None
        # Convert to NumPy arrays if they are PyTorch tensors
        if isinstance(y_true, torch.Tensor):
            y_true = y_true.numpy()
        if isinstance(y_pred, torch.Tensor):
            y_pred = y_pred.numpy()
        return {
            'y_true': y_true,
            'y_pred': y_pred
        }
    except Exception as e:
        logger.error(f"Error loading {pth_path.name}: {e}")
        return None

def clf_report_func(y_true: np.ndarray, y_pred: np.ndarray) -> pd.DataFrame:
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    return pd.DataFrame(report).transpose()

def confusion_matrix_dataframe(y_true: np.ndarray, y_pred: np.ndarray, classes: pd.Series) -> pd.DataFrame:
    cm = confusion_matrix(y_true, y_pred, labels=classes.index)
    df_cm = pd.DataFrame(cm, index=classes.values, columns=classes.values)
    df_cm['Total'] = df_cm.sum(axis=1)
    df_cm.loc['Total'] = df_cm.sum()
    
    # Calculate precision and recall
    for cls in classes.values:
        if df_cm.loc[cls, cls] + df_cm.loc['Total', cls] > 0:
            df_cm.loc['Recall', cls] = df_cm.loc[cls, cls] / df_cm.loc['Total', cls]
        else:
            df_cm.loc['Recall', cls] = 0.0
        if df_cm.loc[cls, cls] + df_cm.loc[cls, 'Total'] > 0:
            df_cm.loc['Precision', cls] = df_cm.loc[cls, cls] / df_cm.loc[cls, 'Total']
        else:
            df_cm.loc['Precision', cls] = 0.0
    return df_cm

def evaluate_hard_label_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    metrics = {
        'overall_recall': recall_score(y_true, y_pred, average='micro', zero_division=0),
        'mean_recall': recall_score(y_true, y_pred, average='macro', zero_division=0),
        'f1_weighted': f1_score(y_true, y_pred, average='weighted', zero_division=0),
        'precision_weighted': precision_score(y_true, y_pred, average='weighted', zero_division=0),
    }
    return metrics

def calculate_iou(y_true: np.ndarray, y_pred: np.ndarray, classes: pd.Series) -> Dict[str, float]:
    """Calculate Intersection over Union (IoU) for each class."""
    iou_dict = {}
    for cls in classes.index:
        intersection = np.logical_and(y_true == cls, y_pred == cls).sum()
        union = np.logical_or(y_true == cls, y_pred == cls).sum()
        if union == 0:
            iou = np.nan  # Undefined IoU
        else:
            iou = intersection / union
        class_name = classes[cls]
        iou_dict[class_name] = iou
    return iou_dict

def calculate_recall_per_class(y_true: np.ndarray, y_pred: np.ndarray, classes: pd.Series) -> Dict[str, float]:
    """Calculate recall for each class."""
    recall_dict = {}
    for cls in classes.index:
        tp = np.logical_and(y_pred == cls, y_true == cls).sum()
        fn = np.logical_and(y_pred != cls, y_true == cls).sum()
        denominator = tp + fn
        if denominator == 0:
            recall = np.nan  # Undefined recall
        else:
            recall = tp / denominator
        class_name = classes[cls]
        recall_dict[class_name] = recall
    return recall_dict

def calculate_iou_and_recall(y_true: np.ndarray, y_pred: np.ndarray, classes: pd.Series) -> pd.DataFrame:
    """Calculate IoU and Recall for each class."""
    iou_dict = calculate_iou(y_true, y_pred, classes)
    recall_dict = calculate_recall_per_class(y_true, y_pred, classes)
    
    df = pd.DataFrame({
        'Class': list(iou_dict.keys()),
        'IoU': list(iou_dict.values()),
        'Recall': list(recall_dict.values())
    })

    df['IoU'] = df['IoU'].round(4)
    df['Recall'] = df['Recall'].round(4)

    # Calculate mean IoU and mean Recall
    mean_iou = df['IoU'].mean(skipna=True)
    mean_recall = df['Recall'].mean(skipna=True)

    # Calculate overall recall
    overall_recall = recall_score(y_true, y_pred, average='micro', zero_division=0)

    # Create a new row for mean values
    mean_row = pd.DataFrame({
        'Class': ['Mean'],
        'IoU': [round(mean_iou, 4)],
        'Recall': [round(mean_recall, 4)]
    })

    # Create a row for overall recall
    overall_row = pd.DataFrame({
        'Class': ['Overall'],
        'IoU': [np.nan],
        'Recall': [round(overall_recall, 4)]
    })

    # Concatenate the mean and overall rows
    df = pd.concat([df, mean_row, overall_row], ignore_index=True)

    return df

def create_confusion_matrix_fig(df_cm: pd.DataFrame, dataset_name: str) -> go.Figure:
    cm = df_cm.iloc[:-2, :-1].values  # Exclude 'Total', 'Recall', 'Precision'
    classes = df_cm.columns[:-1]

    cm_normalized = cm / cm.sum(axis=1, keepdims=True)

    text_array = [
        [f'{percent*100:.1f}%<br>{abbreviate_number(count)}' for count, percent in zip(row, cm_normalized[i])]
        for i, row in enumerate(cm)
    ]

    fig_cm = go.Figure(data=go.Heatmap(
        z=cm_normalized,
        x=classes,
        y=classes,
        colorscale='RdBu_r',
        text=text_array,
        hoverinfo='text',
        colorbar=dict(title='Normalized'),
    ))

    # Add annotations
    for i in range(len(classes)):
        for j in range(len(classes)):
            fig_cm.add_annotation(
                x=classes[j],
                y=classes[i],
                text=f'{cm_normalized[i][j]*100:.1f}%',
                showarrow=False,
                font=dict(color='white' if cm_normalized[i][j] > 0.5 else 'black', size=12)
            )

    fig_cm.update_layout(
        title=f'Normalized Confusion Matrix - {dataset_name}',
        xaxis_title='Predicted Label',
        yaxis_title='True Label',
        width=800,
        height=800,
        template='plotly_white'
    )
    return fig_cm

def create_per_class_metrics_fig(df_iou_recall: pd.DataFrame, dataset_name: str) -> go.Figure:
    classes = df_iou_recall['Class'][:-2]  # Exclude 'Mean' and 'Overall' rows
    iou = df_iou_recall['IoU'][:-2]
    recall = df_iou_recall['Recall'][:-2]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        name='IoU',
        x=classes,
        y=iou,
        marker_color='rgb(59, 130, 246)'
    ))
    fig.add_trace(go.Bar(
        name='Recall',
        x=classes,
        y=recall,
        marker_color='rgb(16, 185, 129)'
    ))

    fig.update_layout(
        title=f'Per-Class IoU and Recall - {dataset_name}',
        xaxis_title='Class',
        yaxis_title='Score',
        barmode='group',
        width=1000,
        height=600,
        template='plotly_white'
    )
    return fig

def create_overall_metrics_fig(metrics: Dict[str, float], dataset_name: str) -> go.Figure:
    selected_metrics = {
        'Overall Recall': metrics['overall_recall'],
        'Mean Recall': metrics['mean_recall'],
        'F1 Score': metrics['f1_weighted'],
        'Precision': metrics['precision_weighted']
    }

    labels = list(selected_metrics.keys())
    values = list(selected_metrics.values())
    values += [values[0]]  # Repeat the first value to close the radar chart
    labels += [labels[0]]

    fig = go.Figure()

    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=labels,
        fill='toself',
        name=dataset_name
    ))

    fig.update_layout(
        title=f'Overall Model Performance - {dataset_name}',
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            )
        ),
        showlegend=True,
        width=700,
        height=700,
        template='plotly_white'
    )
    return fig

def create_combined_radar_plot(visualization_results: List[tuple]) -> go.Figure:
    colors = [
        'rgba(79, 70, 229, 0.6)',  # Indigo
        'rgba(16, 185, 129, 0.6)',  # Emerald
        'rgba(239, 68, 68, 0.6)',   # Red
        'rgba(245, 158, 11, 0.6)',  # Amber
        'rgba(99, 102, 241, 0.6)',  # Blue
        'rgba(236, 72, 153, 0.6)',  # Pink
        'rgba(34, 197, 94, 0.6)',   # Green
        'rgba(168, 85, 247, 0.6)',  # Purple
        'rgba(234, 88, 12, 0.6)',   # Orange
        'rgba(59, 130, 246, 0.6)'   # Light Blue
    ]

    fig = go.Figure()

    for i, (dataset_name, (_, _, fig_overall)) in enumerate(visualization_results):
        trace = fig_overall.data[0]
        fig.add_trace(go.Scatterpolar(
            r=trace.r[:-1],  # Exclude the repeated first value
            theta=trace.theta[:-1],
            fill='toself',
            name=dataset_name,
            line=dict(color=colors[i % len(colors)], width=2)
        ))

    fig.update_layout(
        title='Combined Model Performance Comparison',
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            )
        ),
        showlegend=True,
        width=800,
        height=800,
        template='plotly_white'
    )
    return fig

def export_metrics_html(df: pd.DataFrame, save_path: Path, title: str):
    """Export DataFrame as an HTML table with improved styling."""
    try:
        html_table = df.to_html(index=False, classes='table table-striped table-bordered', border=0)
        html_content = f"""
        <html>
        <head>
            <title>{title}</title>
            <link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/4.0.0/css/bootstrap.min.css">
            <style>
                body {{
                    padding: 20px;
                    background-color: #f8f9fa;
                }}
                h1 {{
                    text-align: center;
                    margin-bottom: 40px;
                }}
                table {{
                    margin: auto;
                    width: 80%;
                }}
                .table-striped tbody tr:nth-of-type(odd) {{
                    background-color: rgba(0,0,0,.05);
                }}
                th {{
                    background-color: #343a40;
                    color: white;
                }}
                td, th {{
                    text-align: center;
                }}
            </style>
        </head>
        <body>
            <h1>{title}</h1>
            {html_table}
        </body>
        </html>
        """
        with open(save_path, 'w') as f:
            f.write(html_content)
        logger.debug(f"Exported HTML table to {save_path}")
    except Exception as e:
        logger.error(f"Failed to export HTML table to {save_path}: {e}")

def process_file(pth_file: Path) -> Optional[Dict[str, Any]]:
    scene_id = pth_file.stem
    logger.debug(f"Loading data for scene: {scene_id}")
    data = load_pth_data(pth_file)
    if data is None:
        return None

    y_true = data['y_true']
    y_pred = data['y_pred']

    try:
        report_df = clf_report_func(y_true, y_pred)
        # Determine unique labels present in both y_true and y_pred
        unique_labels = np.unique(np.concatenate((y_true, y_pred)))
        selected_classes = CLASS_LABELS.loc[unique_labels].dropna()
        cm_df = confusion_matrix_dataframe(y_true, y_pred, selected_classes)
        iou_recall_df = calculate_iou_and_recall(y_true, y_pred, selected_classes)
        metrics = evaluate_hard_label_metrics(y_true, y_pred)
        logger.debug(f"Metrics computed for scene: {scene_id}")
        return {
            'scene_id': scene_id,
            'y_true': y_true,
            'y_pred': y_pred,
            'report_df': report_df,
            'cm_df': cm_df,
            'metrics': metrics,
            'iou_recall_df': iou_recall_df
        }
    except Exception as e:
        logger.error(f"Error computing metrics for {scene_id}: {e}")
        return None

def main():
    args = parse_arguments()

    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve() if args.output_dir else input_dir / "results"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Setup logging after ensuring output_dir exists
    setup_logging(args.verbose, output_dir)

    logger.info(f"Input Directory: {input_dir}")
    logger.info(f"Output Directory: {output_dir}")

    # Gather all .pth files
    all_pth_files = [f for f in input_dir.glob("*.pth") if f.is_file()]
    all_pth_filenames = [f.name for f in all_pth_files]

    # Determine files to process
    if not args.no_interactive:
        selected_filenames = interactive_file_selection(all_pth_filenames)
    elif args.files:
        selected_filenames = args.files
    else:
        selected_filenames = all_pth_filenames

    # Validate selected files
    selected_files = [input_dir / fname for fname in selected_filenames if (input_dir / fname).exists()]

    if not selected_files:
        logger.error("No valid .pth files selected for processing.")
        sys.exit(1)

    visualization_results = []
    combined_y_true = []
    combined_y_pred = []

    for pth_file in selected_files:
        logger.info(f"Processing: {pth_file.name}")
        result = process_file(pth_file)
        if not result:
            continue

        scene_id = result['scene_id']
        save_dir = output_dir / scene_id
        save_dir.mkdir(parents=True, exist_ok=True)

        # Generate and save confusion matrix
        fig_cm = create_confusion_matrix_fig(result['cm_df'], scene_id)
        fig_cm.write_html(save_dir / "confusion_matrix.html")
        logger.debug(f"Saved confusion matrix for {scene_id}")

        # Generate and save per-class metrics
        fig_metrics = create_per_class_metrics_fig(result['iou_recall_df'], scene_id)
        fig_metrics.write_html(save_dir / "per_class_metrics.html")
        logger.debug(f"Saved per-class metrics for {scene_id}")

        # Generate and save overall metrics radar
        fig_overall = create_overall_metrics_fig(result['metrics'], scene_id)
        fig_overall.write_html(save_dir / "overall_metrics.html")
        logger.debug(f"Saved overall metrics for {scene_id}")

        # Save per-category IoU and Recall tables
        iou_recall_df = result['iou_recall_df']
        csv_path = save_dir / "iou_recall_table.csv"
        iou_recall_df.to_csv(csv_path, index=False)
        logger.debug(f"Saved IoU and Recall table for {scene_id}")

        # Export HTML table
        html_path = save_dir / "iou_recall_table.html"
        export_metrics_html(iou_recall_df, html_path, f"IoU and Recall Table - {scene_id}")

        visualization_results.append((scene_id, (fig_cm, fig_metrics, fig_overall)))

        # Aggregate for combined metrics
        combined_y_true.append(result['y_true'])
        combined_y_pred.append(result['y_pred'])

    if visualization_results:
        # Create and save combined radar plot
        combined_fig = create_combined_radar_plot(visualization_results)
        combined_fig.write_html(output_dir / "radar_combined_metrics.html")
        logger.info("Saved combined radar metrics.")

        # Compute combined metrics across all scenes
        logger.info("Computing combined metrics across all scenes...")
        combined_y_true_all = np.concatenate(combined_y_true)
        combined_y_pred_all = np.concatenate(combined_y_pred)
        combined_unique_labels = np.unique(np.concatenate((combined_y_true_all, combined_y_pred_all)))
        combined_selected_classes = CLASS_LABELS.loc[combined_unique_labels].dropna()
        combined_cm_df = confusion_matrix_dataframe(combined_y_true_all, combined_y_pred_all, combined_selected_classes)
        combined_iou_recall_df = calculate_iou_and_recall(combined_y_true_all, combined_y_pred_all, combined_selected_classes)
        combined_metrics = evaluate_hard_label_metrics(combined_y_true_all, combined_y_pred_all)

        # Save combined metrics
        combined_save_dir = output_dir / "combined_metrics"
        combined_save_dir.mkdir(parents=True, exist_ok=True)

        # Generate and save confusion matrix for combined metrics
        combined_fig_cm = create_confusion_matrix_fig(combined_cm_df, "Combined")
        combined_fig_cm.write_html(combined_save_dir / "confusion_matrix.html")
        logger.debug("Saved combined confusion matrix.")

        # Generate and save per-class metrics for combined metrics
        combined_fig_metrics = create_per_class_metrics_fig(combined_iou_recall_df, "Combined")
        combined_fig_metrics.write_html(combined_save_dir / "per_class_metrics.html")
        logger.debug("Saved combined per-class metrics.")

        # Generate and save overall metrics radar for combined metrics
        combined_fig_overall = create_overall_metrics_fig(combined_metrics, "Combined")
        combined_fig_overall.write_html(combined_save_dir / "overall_metrics.html")
        logger.debug("Saved combined overall metrics.")

        # Save per-category IoU and Recall tables for combined metrics
        combined_csv_path = combined_save_dir / "iou_recall_table.csv"
        combined_iou_recall_df.to_csv(combined_csv_path, index=False)
        logger.debug("Saved combined IoU and Recall table.")

        # Export combined HTML table
        combined_html_path = combined_save_dir / "iou_recall_table.html"
        export_metrics_html(combined_iou_recall_df, combined_html_path, "Combined IoU and Recall Table")

        # Optionally, append combined visualization to radar plot
        visualization_results.append(("Combined", (combined_fig_cm, combined_fig_metrics, combined_fig_overall)))

        # Recreate combined radar plot including the combined metrics
        final_combined_fig = create_combined_radar_plot(visualization_results)
        final_combined_fig.write_html(output_dir / "radar_combined_metrics_final.html")
        logger.info("Saved final combined radar metrics including individual and combined scenes.")
    else:
        logger.warning("No visualizations were generated.")

    logger.info("Processing completed successfully.")

if __name__ == "__main__":
    main()
