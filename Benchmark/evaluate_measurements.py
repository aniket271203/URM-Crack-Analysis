#!/usr/bin/env python3
"""
Measurement Validation Script
Calculates error metrics for crack and brick dimensions across different test conditions.
Used to validate the accuracy and robustness of the crack analysis pipeline.
"""

import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def calculate_errors(pred, gt):
    """Calculate absolute and percentage errors"""
    abs_error = abs(pred - gt)
    rel_error = (abs_error / gt) * 100 if gt != 0 else 0
    return abs_error, rel_error

def run_evaluation(gt_csv_path, results_dir):
    """
    Run evaluation by matching ground truth CSV with JSON results.
    """
    if not os.path.exists(gt_csv_path):
        print(f"Error: Ground truth file {gt_csv_path} not found.")
        return
    
    gt_df = pd.read_csv(gt_csv_path)
    results_dir = Path(results_dir)
    
    evaluation_results = []
    
    for _, row in gt_df.iterrows():
        filename = row['filename']
        # Remove extension to find corresponding json/sub dir
        base_name = os.path.splitext(filename)[0]
        
        # Look for measurement.json in expected locations
        # Structure could be results_dir/base_name/measurements.json
        json_path = results_dir / base_name / "measurements.json"
        
        if not json_path.exists():
            # Try alternate: results_dir/base_name_measurements.json
            json_path = results_dir / f"{base_name}_measurements.json"
            
        if not json_path.exists():
            print(f"Warning: Results for {filename} not found at {json_path}")
            continue
            
        with open(json_path, 'r') as f:
            pred_data = json.load(f)
            
        if not pred_data.get('success', False):
            print(f"Warning: Measurement failed for {filename}")
            continue
            
        # Crack Length Metrics
        l_pred = pred_data.get('length_mm', 0)
        l_gt = row['true_crack_length_mm']
        l_abs, l_rel = calculate_errors(l_pred, l_gt)
        
        # Max Width Metrics
        w_pred = pred_data.get('width_stats', {}).get('max_mm', 0)
        w_gt = row['true_max_width_mm']
        w_abs, w_rel = calculate_errors(w_pred, w_gt)
        
        # Brick Height Metrics (Calibration Accuracy)
        # The tool uses input brick height/length to set scale. 
        # We check how consistent the detected px dimensions are.
        h_pred_px = pred_data.get('calibration', {}).get('brick_height_px', 0)
        scale = pred_data.get('scale_mm_per_px', 0)
        h_calc_mm = h_pred_px * scale
        h_gt_mm = row['true_brick_height_mm']
        h_abs, h_rel = calculate_errors(h_calc_mm, h_gt_mm)

        evaluation_results.append({
            'filename': filename,
            'angle': row.get('angle_deg', 90),
            'lighting': row.get('lighting_cond', 'Normal'),
            'distance': row.get('distance_m', 2),
            # Predicted
            'l_pred': l_pred,
            'w_pred': w_pred,
            'h_pred': h_calc_mm,
            # GT
            'l_gt': l_gt,
            'w_gt': w_gt,
            'h_gt': h_gt_mm,
            # Errors
            'l_error_pct': l_rel,
            'w_error_pct': w_rel,
            'h_error_pct': h_rel,
            'l_error_abs': l_abs,
            'w_error_abs': w_abs
        })

    eval_df = pd.DataFrame(evaluation_results)
    if eval_df.empty:
        print("No matches found between GT and results.")
        return
    
    # Calculate Aggregate Metrics
    summary = {
        'MAPE_Length': eval_df['l_error_pct'].mean(),
        'MAPE_Width': eval_df['w_error_pct'].mean(),
        'MAPE_Calibration': eval_df['h_error_pct'].mean(),
        'RMSE_Length': np.sqrt((eval_df['l_error_abs']**2).mean()),
        'RMSE_Width': np.sqrt((eval_df['w_error_abs']**2).mean()),
        'Max_Error_Length_Pct': eval_df['l_error_pct'].max(),
        'Max_Error_Width_Pct': eval_df['w_error_pct'].max()
    }
    
    print("\n" + "="*40)
    print("MEASUREMENT VALIDATION SUMMARY")
    print("="*40)
    for k, v in summary.items():
        print(f"{k:25}: {v:.3f}")
    
    # Save Detailed CSV
    output_csv = results_dir / "validation_report.csv"
    eval_df.to_csv(output_csv, index=False)
    print(f"\nDetailed report saved to: {output_csv}")
    
    # Generate Visualizations if we have enough data
    generate_plots(eval_df, results_dir)

def generate_plots(df, output_dir):
    """Generate plots for paper figures"""
    plt.style.use('seaborn-v0_8-whitegrid')
    output_dir = Path(output_dir)
    
    # 1. Error vs Angle
    if 'angle' in df.columns and len(df['angle'].unique()) > 1:
        plt.figure(figsize=(10, 6))
        sns.boxplot(x='angle', y='l_error_pct', data=df)
        plt.title('Crack Length Measurement Error vs. Camera Angle')
        plt.ylabel('Percentage Error (%)')
        plt.xlabel('Camera Angle (degrees)')
        plt.savefig(output_dir / "error_vs_angle.png")
        plt.close()

    # 2. Predicted vs Ground Truth (Parity Plot)
    plt.figure(figsize=(8, 8))
    max_val = max(df['l_gt'].max(), df['l_pred'].max())
    plt.scatter(df['l_gt'], df['l_pred'], alpha=0.7, c='blue', label='Length')
    plt.plot([0, max_val], [0, max_val], 'r--', label='Ideal')
    plt.title('Predicted vs. Measured Crack Length')
    plt.xlabel('Ground Truth (mm)')
    plt.ylabel('Predicted (mm)')
    plt.legend()
    plt.savefig(output_dir / "length_parity_plot.png")
    plt.close()

    # 3. Width Accuracy
    plt.figure(figsize=(8, 8))
    max_w = max(df['w_gt'].max(), df['w_pred'].max())
    plt.scatter(df['w_gt'], df['w_pred'], alpha=0.7, c='green', label='Max Width')
    plt.plot([0, max_w], [0, max_w], 'r--', label='Ideal')
    plt.title('Predicted vs. Measured Max Crack Width')
    plt.xlabel('Ground Truth (mm)')
    plt.ylabel('Predicted (mm)')
    plt.legend()
    plt.savefig(output_dir / "width_parity_plot.png")
    plt.close()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate measurement accuracy")
    parser.add_argument("--gt", required=True, help="Path to ground truth CSV")
    parser.add_argument("--results", required=True, help="Directory containing measurements.json files")
    args = parser.parse_args()
    
    run_evaluation(args.gt, args.results)
