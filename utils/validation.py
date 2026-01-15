"""
This script is for validating segmentation models against ground truth.
Provides both numerical metrics and visualizations for 2D segmentation comparison.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
from pathlib import Path
from typing import Tuple, Dict, Optional
import tifffile
from scipy.ndimage import label, distance_transform_edt


def load_segmentation(filepath: str) -> np.ndarray:
    """
    Load a segmentation mask from TIFF file.
    
    Args:
        filepath: Path to segmentation TIFF image
    
    Returns:
        Binary segmentation mask (0 = background, >0 = foreground)
    """
    seg = tifffile.imread(filepath)
    
    # Convert to binary if needed
    if seg.dtype == bool:
        return seg.astype(np.uint8)
    elif seg.max() > 1:
        # Assume any non-zero value is foreground
        return (seg > 0).astype(np.uint8)
    else:
        return seg.astype(np.uint8)


def compute_metrics(ground_truth: np.ndarray, prediction: np.ndarray) -> Dict[str, float]:
    """
    Compute comprehensive segmentation metrics.
    
    Args:
        ground_truth: Binary ground truth mask (0/1)
        prediction: Binary prediction mask (0/1)
    
    Returns:
        Dictionary containing various segmentation metrics
    """
    # Ensure binary
    gt = ground_truth.astype(bool)
    pred = prediction.astype(bool)
    
    # True Positives, False Positives, False Negatives, True Negatives
    tp = np.logical_and(gt, pred).sum()
    fp = np.logical_and(~gt, pred).sum()
    fn = np.logical_and(gt, ~pred).sum()
    tn = np.logical_and(~gt, ~pred).sum()
    
    # Basic metrics
    metrics = {}
    
    # Intersection over Union (IoU) / Jaccard Index
    intersection = tp
    union = tp + fp + fn
    metrics['IoU'] = intersection / union if union > 0 else 0.0
    
    # Dice Coefficient / F1 Score
    metrics['Dice'] = (2 * tp) / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.0
    
    # Precision (Positive Predictive Value)
    metrics['Precision'] = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    
    # Recall (Sensitivity, True Positive Rate)
    metrics['Recall'] = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    
    # Specificity (True Negative Rate)
    metrics['Specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    
    # Accuracy
    metrics['Accuracy'] = (tp + tn) / (tp + tn + fp + fn)
    
    # F1 Score (same as Dice for binary segmentation)
    metrics['F1_Score'] = metrics['Dice']
    
    # Pixel counts
    metrics['TP'] = int(tp)
    metrics['FP'] = int(fp)
    metrics['FN'] = int(fn)
    metrics['TN'] = int(tn)
    
    # Area metrics
    gt_area = gt.sum()
    pred_area = pred.sum()
    metrics['GT_Area'] = int(gt_area)
    metrics['Pred_Area'] = int(pred_area)
    metrics['Area_Diff'] = int(pred_area - gt_area)
    metrics['Area_Ratio'] = pred_area / gt_area if gt_area > 0 else 0.0
    
    # Hausdorff Distance (if there are objects)
    if gt.sum() > 0 and pred.sum() > 0:
        metrics['Hausdorff'] = compute_hausdorff_distance(gt, pred)
    else:
        metrics['Hausdorff'] = np.inf
    
    return metrics


def compute_hausdorff_distance(mask1: np.ndarray, mask2: np.ndarray) -> float:
    """
    Compute symmetric Hausdorff distance between two binary masks.
    
    Args:
        mask1: First binary mask
        mask2: Second binary mask
    
    Returns:
        Hausdorff distance in pixels
    """
    # Get boundary points
    boundary1 = mask1 ^ distance_transform_edt(mask1) > 1
    boundary2 = mask2 ^ distance_transform_edt(mask2) > 1
    
    if boundary1.sum() == 0 or boundary2.sum() == 0:
        return np.inf
    
    # Distance transform
    dt1 = distance_transform_edt(~boundary1)
    dt2 = distance_transform_edt(~boundary2)
    
    # Hausdorff distance (max of min distances)
    hd1 = dt1[boundary2].max()
    hd2 = dt2[boundary1].max()
    
    return max(hd1, hd2)


def create_visualization(ground_truth: np.ndarray, 
                         prediction: np.ndarray,
                         metrics: Dict[str, float],
                         save_path: Optional[str] = None,
                         title: str = "Segmentation Comparison") -> plt.Figure:
    """
    Create comprehensive visualization comparing segmentations.
    
    Args:
        ground_truth: Binary ground truth mask
        prediction: Binary prediction mask
        metrics: Dictionary of computed metrics
        save_path: Optional path to save the figure
        title: Main title for the figure
    
    Returns:
        Matplotlib figure object
    """
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)
    
    # 1. Ground Truth
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(ground_truth, cmap='gray', interpolation='nearest')
    ax1.set_title('Ground Truth', fontsize=14, fontweight='bold')
    ax1.axis('off')
    ax1.text(0.02, 0.98, f'Area: {metrics["GT_Area"]:,} px', 
             transform=ax1.transAxes, fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # 2. Prediction
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.imshow(prediction, cmap='gray', interpolation='nearest')
    ax2.set_title('Prediction', fontsize=14, fontweight='bold')
    ax2.axis('off')
    ax2.text(0.02, 0.98, f'Area: {metrics["Pred_Area"]:,} px', 
             transform=ax2.transAxes, fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # 3. Overlay (GT in green, Pred in magenta, overlap in white)
    ax3 = fig.add_subplot(gs[0, 2])
    overlay = np.zeros((*ground_truth.shape, 3))
    overlay[ground_truth > 0] = [0, 1, 0]  # Green for GT
    overlay[prediction > 0] += [1, 0, 1]   # Magenta for Pred (overlap becomes white)
    ax3.imshow(overlay, interpolation='nearest')
    ax3.set_title('Overlay (GT=Green, Pred=Magenta)', fontsize=14, fontweight='bold')
    ax3.axis('off')
    
    # 4. Difference Map (TP, FP, FN)
    ax4 = fig.add_subplot(gs[0, 3])
    diff_map = np.zeros(ground_truth.shape, dtype=np.uint8)
    diff_map[np.logical_and(ground_truth, prediction)] = 3  # True Positive (white)
    diff_map[np.logical_and(~ground_truth, prediction)] = 2  # False Positive (red)
    diff_map[np.logical_and(ground_truth, ~prediction)] = 1  # False Negative (blue)
    
    cmap_diff = matplotlib.colors.ListedColormap(['black', 'blue', 'red', 'white'])
    ax4.imshow(diff_map, cmap=cmap_diff, interpolation='nearest', vmin=0, vmax=3)
    ax4.set_title('Difference Map', fontsize=14, fontweight='bold')
    ax4.axis('off')
    
    # Legend for difference map
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='white', edgecolor='black', label=f'TP: {metrics["TP"]:,}'),
                      Patch(facecolor='red', edgecolor='black', label=f'FP: {metrics["FP"]:,}'),
                      Patch(facecolor='blue', edgecolor='black', label=f'FN: {metrics["FN"]:,}')]
    ax4.legend(handles=legend_elements, loc='upper right', fontsize=10)
    
    # 5. Metrics Table
    ax5 = fig.add_subplot(gs[1, :2])
    ax5.axis('off')
    
    metrics_text = f"""SEGMENTATION METRICS
{'='*50}

Primary Metrics:
  IoU (Jaccard):        {metrics['IoU']:.4f}
  Dice Coefficient:     {metrics['Dice']:.4f}
  F1 Score:             {metrics['F1_Score']:.4f}
  
Classification Metrics:
  Precision (PPV):      {metrics['Precision']:.4f}
  Recall (Sensitivity): {metrics['Recall']:.4f}
  Specificity:          {metrics['Specificity']:.4f}
  Accuracy:             {metrics['Accuracy']:.4f}
  
Pixel Counts:
  True Positives:       {metrics['TP']:,}
  False Positives:      {metrics['FP']:,}
  False Negatives:      {metrics['FN']:,}
  True Negatives:       {metrics['TN']:,}
  
Area Comparison:
  Ground Truth Area:    {metrics['GT_Area']:,} pixels
  Prediction Area:      {metrics['Pred_Area']:,} pixels
  Area Difference:      {metrics['Area_Diff']:+,} pixels
  Area Ratio (P/GT):    {metrics['Area_Ratio']:.4f}
  
Distance Metric:
  Hausdorff Distance:   {metrics['Hausdorff']:.2f} pixels
"""
    
    ax5.text(0.05, 0.95, metrics_text, transform=ax5.transAxes,
             fontsize=11, verticalalignment='top', family='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # 6. Confusion Matrix
    ax6 = fig.add_subplot(gs[1, 2:])
    confusion = np.array([[metrics['TN'], metrics['FP']],
                          [metrics['FN'], metrics['TP']]])
    
    im = ax6.imshow(confusion, cmap='Blues', interpolation='nearest')
    ax6.set_xticks([0, 1])
    ax6.set_yticks([0, 1])
    ax6.set_xticklabels(['Negative (BG)', 'Positive (FG)'], fontsize=11)
    ax6.set_yticklabels(['Negative (BG)', 'Positive (FG)'], fontsize=11)
    ax6.set_xlabel('Predicted', fontsize=12, fontweight='bold')
    ax6.set_ylabel('Ground Truth', fontsize=12, fontweight='bold')
    ax6.set_title('Confusion Matrix', fontsize=14, fontweight='bold')
    
    # Add text annotations
    for i in range(2):
        for j in range(2):
            text = ax6.text(j, i, f'{confusion[i, j]:,}',
                          ha="center", va="center", color="black", fontsize=14, fontweight='bold')
    
    plt.colorbar(im, ax=ax6, fraction=0.046, pad=0.04)
    
    # 7. Metric bars
    ax7 = fig.add_subplot(gs[2, :])
    metric_names = ['IoU', 'Dice', 'Precision', 'Recall', 'Specificity', 'Accuracy']
    metric_values = [metrics[m] for m in metric_names]
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    
    bars = ax7.barh(metric_names, metric_values, color=colors, alpha=0.7, edgecolor='black')
    ax7.set_xlim(0, 1)
    ax7.set_xlabel('Score', fontsize=12, fontweight='bold')
    ax7.set_title('Metric Comparison', fontsize=14, fontweight='bold')
    ax7.grid(axis='x', alpha=0.3)
    
    # Add value labels on bars
    for bar, value in zip(bars, metric_values):
        width = bar.get_width()
        ax7.text(width + 0.01, bar.get_y() + bar.get_height()/2,
                f'{value:.3f}', ha='left', va='center', fontsize=11, fontweight='bold')
    
    # Overall title
    fig.suptitle(title, fontsize=18, fontweight='bold', y=0.98)
    
    # Save if path provided
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Visualization saved to: {save_path}")
    
    return fig


def compare_segmentations(ground_truth_path: str,
                          prediction_path: str,
                          output_dir: Optional[str] = None,
                          show_plot: bool = False) -> Dict[str, float]:
    """
    Main function to compare two segmentations and generate report.
    
    Args:
        ground_truth_path: Path to ground truth segmentation
        prediction_path: Path to prediction segmentation
        output_dir: Directory to save visualization (optional)
        show_plot: Whether to display the plot (default: False for HPC)
    
    Returns:
        Dictionary of computed metrics
    """
    # Load segmentations
    print(f"Loading ground truth: {ground_truth_path}")
    gt = load_segmentation(ground_truth_path)
    
    print(f"Loading prediction: {prediction_path}")
    pred = load_segmentation(prediction_path)
    
    # Check dimensions match
    if gt.shape != pred.shape:
        raise ValueError(f"Shape mismatch! GT: {gt.shape}, Pred: {pred.shape}")
    
    print(f"Image size: {gt.shape}")
    
    # Compute metrics
    print("\nComputing metrics...")
    metrics = compute_metrics(gt, pred)
    
    # Print summary
    print("\n" + "="*60)
    print("SEGMENTATION COMPARISON RESULTS")
    print("="*60)
    print(f"IoU (Jaccard Index):  {metrics['IoU']:.4f}")
    print(f"Dice Coefficient:     {metrics['Dice']:.4f}")
    print(f"Precision:            {metrics['Precision']:.4f}")
    print(f"Recall:               {metrics['Recall']:.4f}")
    print(f"F1 Score:             {metrics['F1_Score']:.4f}")
    print(f"Accuracy:             {metrics['Accuracy']:.4f}")
    print(f"Hausdorff Distance:   {metrics['Hausdorff']:.2f} pixels")
    print("="*60)
    
    # Create visualization
    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Generate filename
        gt_name = Path(ground_truth_path).stem
        pred_name = Path(prediction_path).stem
        save_path = output_path / f"comparison_{pred_name}_vs_{gt_name}.png"
        
        print(f"\nCreating visualization...")
        create_visualization(gt, pred, metrics, save_path=str(save_path))
    elif show_plot:
        create_visualization(gt, pred, metrics)
        plt.show()
    
    return metrics


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Compare two 2D segmentations')
    parser.add_argument('--ground_truth', '-gt', required=True, help='Path to ground truth segmentation')
    parser.add_argument('--prediction', '-pred', required=True, help='Path to prediction segmentation')
    parser.add_argument('--output_dir', '-o', default='./validation_results', help='Output directory for results')
    parser.add_argument('--show', action='store_true', help='Show plot (not recommended for HPC)')
    
    args = parser.parse_args()
    
    # Run comparison
    metrics = compare_segmentations(
        ground_truth_path=args.ground_truth,
        prediction_path=args.prediction,
        output_dir=args.output_dir,
        show_plot=args.show
    )

