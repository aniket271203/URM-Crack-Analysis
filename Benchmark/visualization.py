"""
Visualization utilities for crack analysis benchmarking
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import cv2
from matplotlib.patches import Rectangle
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from sklearn.metrics import confusion_matrix
import io
import base64


class BenchmarkVisualizer:
    """Create comprehensive visualizations for benchmark results"""
    
    def __init__(self, output_dir: Path, style: str = 'seaborn-v0_8', dpi: int = 300):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set style
        plt.style.use(style)
        self.dpi = dpi
        
        # Color palette
        self.colors = {
            'primary': '#2E86AB',
            'secondary': '#A23B72',
            'accent': '#F18F01', 
            'success': '#C73E1D',
            'info': '#5D737E',
            'crack_types': ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'],
            'severity': ['#90EE90', '#FFD700', '#FF4500']  # light, medium, severe
        }
    
    def plot_detection_metrics(self, yolo_metrics: Dict, save_name: str = "detection_metrics.png") -> Path:
        """Plot YOLO detection performance metrics"""
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('YOLO Crack Detection Performance', fontsize=16, fontweight='bold')
        
        # Precision, Recall, F1-Score bar plot
        ax1 = axes[0, 0]
        metrics = ['Precision', 'Recall', 'F1-Score']
        values = [yolo_metrics['precision'], yolo_metrics['recall'], yolo_metrics['f1_score']]
        bars = ax1.bar(metrics, values, color=self.colors['crack_types'][:3])
        ax1.set_ylabel('Score')
        ax1.set_title('Detection Performance Metrics')
        ax1.set_ylim(0, 1)
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # Detection counts pie chart
        ax2 = axes[0, 1]
        labels = ['True Positives', 'False Positives', 'False Negatives']
        sizes = [yolo_metrics['tp_count'], yolo_metrics['fp_count'], yolo_metrics['fn_count']]
        colors = [self.colors['success'], self.colors['accent'], self.colors['secondary']]
        
        if sum(sizes) > 0:
            wedges, texts, autotexts = ax2.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%')
            ax2.set_title('Detection Count Distribution')
        
        # mAP visualization
        if 'mAP_per_iou' in yolo_metrics:
            ax3 = axes[1, 0]
            iou_thresholds = list(yolo_metrics['mAP_per_iou'].keys())
            map_values = list(yolo_metrics['mAP_per_iou'].values())
            ax3.plot(iou_thresholds, map_values, marker='o', linewidth=2, markersize=6,
                    color=self.colors['primary'])
            ax3.set_xlabel('IoU Threshold')
            ax3.set_ylabel('Average Precision')
            ax3.set_title('mAP vs IoU Threshold')
            ax3.grid(True, alpha=0.3)
        
        # Summary statistics
        ax4 = axes[1, 1]
        ax4.axis('off')
        summary_text = f"""
        Detection Summary
        ─────────────────
        Total GT Objects: {yolo_metrics.get('total_gt', 'N/A')}
        Total Predictions: {yolo_metrics.get('total_pred', 'N/A')}
        
        mAP@0.5: {yolo_metrics.get('mAP@0.5', 0):.3f}
        mAP@0.5:0.95: {yolo_metrics.get('mAP@0.5:0.95', 0):.3f}
        
        True Positives: {yolo_metrics['tp_count']}
        False Positives: {yolo_metrics['fp_count']}
        False Negatives: {yolo_metrics['fn_count']}
        """
        ax4.text(0.1, 0.5, summary_text, fontsize=12, verticalalignment='center',
                bbox=dict(boxstyle="round,pad=0.3", facecolor=self.colors['info'], alpha=0.1))
        
        plt.tight_layout()
        save_path = self.output_dir / save_name
        plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        
        return save_path
    
    def plot_segmentation_metrics(self, seg_metrics: Dict, save_name: str = "segmentation_metrics.png") -> Path:
        """Plot segmentation performance metrics"""
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Segmentation Performance Analysis', fontsize=16, fontweight='bold')
        
        # Main metrics bar plot
        ax1 = axes[0, 0]
        metrics = ['Mean IoU', 'Mean Dice', 'Mean Pixel Accuracy']
        values = [seg_metrics['mean_iou'], seg_metrics['mean_dice'], seg_metrics['mean_pixel_accuracy']]
        stds = [seg_metrics['std_iou'], seg_metrics['std_dice'], seg_metrics['std_pixel_accuracy']]
        
        bars = ax1.bar(metrics, values, yerr=stds, capsize=5, color=self.colors['crack_types'][:3])
        ax1.set_ylabel('Score')
        ax1.set_title('Segmentation Metrics with Standard Deviation')
        ax1.set_ylim(0, 1)
        
        # Add value labels
        for bar, value, std in zip(bars, values, stds):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + std + 0.01,
                    f'{value:.3f}±{std:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # IoU distribution histogram
        ax2 = axes[0, 1]
        if 'per_image_metrics' in seg_metrics:
            ious = seg_metrics['per_image_metrics']['ious']
            ax2.hist(ious, bins=20, alpha=0.7, color=self.colors['primary'], edgecolor='black')
            ax2.axvline(seg_metrics['mean_iou'], color=self.colors['accent'], linestyle='--', 
                       linewidth=2, label=f'Mean: {seg_metrics["mean_iou"]:.3f}')
            ax2.set_xlabel('IoU Score')
            ax2.set_ylabel('Frequency')
            ax2.set_title('IoU Score Distribution')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        
        # Dice vs IoU scatter plot
        ax3 = axes[1, 0]
        if 'per_image_metrics' in seg_metrics:
            ious = seg_metrics['per_image_metrics']['ious']
            dices = seg_metrics['per_image_metrics']['dice_scores']
            ax3.scatter(ious, dices, alpha=0.6, color=self.colors['primary'])
            ax3.plot([0, 1], [0, 1], 'r--', alpha=0.8, label='Perfect Correlation')
            ax3.set_xlabel('IoU Score')
            ax3.set_ylabel('Dice Score')
            ax3.set_title('IoU vs Dice Score Correlation')
            ax3.grid(True, alpha=0.3)
            ax3.legend()
        
        # Performance summary
        ax4 = axes[1, 1]
        ax4.axis('off')
        summary_text = f"""
        Segmentation Summary
        ─────────────────────
        Mean IoU: {seg_metrics['mean_iou']:.3f} ± {seg_metrics['std_iou']:.3f}
        Mean Dice: {seg_metrics['mean_dice']:.3f} ± {seg_metrics['std_dice']:.3f}
        Mean Pixel Acc: {seg_metrics['mean_pixel_accuracy']:.3f} ± {seg_metrics['std_pixel_accuracy']:.3f}
        
        Best IoU: {max(seg_metrics['per_image_metrics']['ious']) if 'per_image_metrics' in seg_metrics else 'N/A'}
        Worst IoU: {min(seg_metrics['per_image_metrics']['ious']) if 'per_image_metrics' in seg_metrics else 'N/A'}
        """
        ax4.text(0.1, 0.5, summary_text, fontsize=12, verticalalignment='center',
                bbox=dict(boxstyle="round,pad=0.3", facecolor=self.colors['info'], alpha=0.1))
        
        plt.tight_layout()
        save_path = self.output_dir / save_name
        plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        
        return save_path
    
    def plot_classification_metrics(self, class_metrics: Dict, save_name: str = "classification_metrics.png") -> Path:
        """Plot classification performance metrics"""
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Crack Classification Performance', fontsize=16, fontweight='bold')
        
        # Per-class metrics heatmap
        ax1 = axes[0, 0]
        class_names = class_metrics['class_names']
        per_class = class_metrics['per_class_metrics']
        
        # Create data matrix for heatmap
        metrics_matrix = []
        metric_names = ['Precision', 'Recall', 'F1-Score']
        
        for metric in ['precision', 'recall', 'f1_score']:
            row = [per_class[class_name][metric] for class_name in class_names]
            metrics_matrix.append(row)
        
        im = ax1.imshow(metrics_matrix, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
        ax1.set_xticks(range(len(class_names)))
        ax1.set_yticks(range(len(metric_names)))
        ax1.set_xticklabels(class_names)
        ax1.set_yticklabels(metric_names)
        ax1.set_title('Per-Class Performance Heatmap')
        
        # Add text annotations
        for i in range(len(metric_names)):
            for j in range(len(class_names)):
                text = ax1.text(j, i, f'{metrics_matrix[i][j]:.3f}',
                               ha="center", va="center", color="black", fontweight='bold')
        
        plt.colorbar(im, ax=ax1)
        
        # Confusion matrix
        ax2 = axes[0, 1]
        cm = class_metrics['confusion_matrix']
        im2 = ax2.imshow(cm, cmap='Blues', aspect='auto')
        ax2.set_xticks(range(len(class_names)))
        ax2.set_yticks(range(len(class_names)))
        ax2.set_xticklabels(class_names, rotation=45)
        ax2.set_yticklabels(class_names)
        ax2.set_xlabel('Predicted')
        ax2.set_ylabel('True')
        ax2.set_title('Confusion Matrix')
        
        # Add text annotations
        for i in range(len(class_names)):
            for j in range(len(class_names)):
                text = ax2.text(j, i, cm[i, j], ha="center", va="center",
                               color="white" if cm[i, j] > cm.max()/2 else "black",
                               fontweight='bold')
        
        plt.colorbar(im2, ax=ax2)
        
        # Overall metrics bar plot
        ax3 = axes[1, 0]
        overall_metrics = ['Accuracy', 'Precision (Macro)', 'Recall (Macro)', 'F1 (Macro)']
        overall_values = [
            class_metrics['accuracy'],
            class_metrics['precision_macro'],
            class_metrics['recall_macro'],
            class_metrics['f1_macro']
        ]
        
        bars = ax3.bar(overall_metrics, overall_values, color=self.colors['crack_types'])
        ax3.set_ylabel('Score')
        ax3.set_title('Overall Classification Metrics')
        ax3.set_ylim(0, 1)
        plt.setp(ax3.get_xticklabels(), rotation=45, ha='right')
        
        # Add value labels
        for bar, value in zip(bars, overall_values):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # ROC curves (if available)
        ax4 = axes[1, 1]
        if 'roc_auc' in class_metrics:
            for i, class_name in enumerate(class_names):
                if i in class_metrics['fpr'] and i in class_metrics['tpr']:
                    ax4.plot(class_metrics['fpr'][i], class_metrics['tpr'][i],
                            label=f'{class_name} (AUC = {class_metrics["roc_auc"][i]:.3f})',
                            color=self.colors['crack_types'][i % len(self.colors['crack_types'])])
            
            ax4.plot([0, 1], [0, 1], 'k--', alpha=0.8, label='Random Classifier')
            ax4.set_xlabel('False Positive Rate')
            ax4.set_ylabel('True Positive Rate')
            ax4.set_title('ROC Curves')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
        else:
            ax4.axis('off')
            summary_text = f"""
            Classification Summary
            ───────────────────────
            Overall Accuracy: {class_metrics['accuracy']:.3f}
            Macro Precision: {class_metrics['precision_macro']:.3f}
            Macro Recall: {class_metrics['recall_macro']:.3f}
            Macro F1-Score: {class_metrics['f1_macro']:.3f}
            
            Weighted Precision: {class_metrics['precision_weighted']:.3f}
            Weighted Recall: {class_metrics['recall_weighted']:.3f}
            Weighted F1-Score: {class_metrics['f1_weighted']:.3f}
            """
            ax4.text(0.1, 0.5, summary_text, fontsize=10, verticalalignment='center',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor=self.colors['info'], alpha=0.1))
        
        plt.tight_layout()
        save_path = self.output_dir / save_name
        plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        
        return save_path
    
    def plot_pipeline_overview(self, pipeline_metrics: Dict, timing_metrics: Dict,
                              save_name: str = "pipeline_overview.png") -> Path:
        """Plot overall pipeline performance overview"""
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('End-to-End Pipeline Performance Overview', fontsize=16, fontweight='bold')
        
        # Overall accuracy breakdown
        ax1 = axes[0, 0]
        accuracy_types = ['Overall', 'Detection', 'Classification']
        accuracy_values = [
            pipeline_metrics['overall_accuracy'],
            pipeline_metrics['detection_accuracy'],
            pipeline_metrics['classification_accuracy']
        ]
        
        bars = ax1.bar(accuracy_types, accuracy_values, color=self.colors['crack_types'][:3])
        ax1.set_ylabel('Accuracy')
        ax1.set_title('Pipeline Accuracy Breakdown')
        ax1.set_ylim(0, 1)
        
        for bar, value in zip(bars, accuracy_values):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # Error distribution pie chart
        ax2 = axes[0, 1]
        error_labels = ['Correct', 'Detection Errors', 'Classification Errors']
        error_counts = [
            pipeline_metrics['correct_predictions'],
            pipeline_metrics['detection_errors'],
            pipeline_metrics['classification_errors']
        ]
        colors = [self.colors['success'], self.colors['accent'], self.colors['secondary']]
        
        wedges, texts, autotexts = ax2.pie(error_counts, labels=error_labels, colors=colors,
                                          autopct='%1.1f%%')
        ax2.set_title('Error Distribution')
        
        # Processing time breakdown
        ax3 = axes[1, 0]
        if timing_metrics:
            time_components = ['Detection', 'Segmentation', 'Classification']
            time_values = [
                timing_metrics.get('mean_detection_time', 0),
                timing_metrics.get('mean_segmentation_time', 0),
                timing_metrics.get('mean_classification_time', 0)
            ]
            
            bars = ax3.bar(time_components, time_values, color=self.colors['crack_types'][:3])
            ax3.set_ylabel('Time (seconds)')
            ax3.set_title('Average Processing Time per Component')
            
            for bar, value in zip(bars, time_values):
                ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                        f'{value:.3f}s', ha='center', va='bottom', fontweight='bold')
        
        # Performance summary
        ax4 = axes[1, 1]
        ax4.axis('off')
        summary_text = f"""
        Pipeline Performance Summary
        ─────────────────────────────
        Total Images Processed: {pipeline_metrics['total_images']}
        Overall Accuracy: {pipeline_metrics['overall_accuracy']:.3f}
        
        Correct Predictions: {pipeline_metrics['correct_predictions']}
        Detection Errors: {pipeline_metrics['detection_errors']}
        Classification Errors: {pipeline_metrics['classification_errors']}
        
        Average Total Time: {timing_metrics.get('mean_total_time', 0):.3f}s
        Throughput: {timing_metrics.get('images_per_second', 0):.2f} images/sec
        """
        ax4.text(0.1, 0.5, summary_text, fontsize=11, verticalalignment='center',
                bbox=dict(boxstyle="round,pad=0.3", facecolor=self.colors['info'], alpha=0.1))
        
        plt.tight_layout()
        save_path = self.output_dir / save_name
        plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        
        return save_path
    
    def plot_failure_analysis(self, failure_cases: List[Dict], save_name: str = "failure_analysis.png") -> Path:
        """Plot analysis of failure cases"""
        
        if not failure_cases:
            # Create empty plot
            fig, ax = plt.subplots(1, 1, figsize=(10, 6))
            ax.text(0.5, 0.5, 'No Failure Cases Found\n🎉 Perfect Performance!',
                   ha='center', va='center', fontsize=16, 
                   bbox=dict(boxstyle="round,pad=0.5", facecolor=self.colors['success'], alpha=0.2))
            ax.set_title('Failure Case Analysis')
            ax.axis('off')
            save_path = self.output_dir / save_name
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            plt.close()
            return save_path
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Failure Case Analysis', fontsize=16, fontweight='bold')
        
        # Failure type distribution
        ax1 = axes[0, 0]
        failure_types = [case.get('failure_type', 'Unknown') for case in failure_cases]
        unique_types, counts = np.unique(failure_types, return_counts=True)
        
        bars = ax1.bar(unique_types, counts, color=self.colors['crack_types'][:len(unique_types)])
        ax1.set_ylabel('Count')
        ax1.set_title('Failure Type Distribution')
        plt.setp(ax1.get_xticklabels(), rotation=45, ha='right')
        
        # Confidence distribution for false positives
        ax2 = axes[0, 1]
        fp_confidences = [case.get('confidence', 0) for case in failure_cases 
                         if case.get('failure_type') == 'false_positive']
        
        if fp_confidences:
            ax2.hist(fp_confidences, bins=10, alpha=0.7, color=self.colors['secondary'])
            ax2.set_xlabel('Confidence Score')
            ax2.set_ylabel('Frequency')
            ax2.set_title('False Positive Confidence Distribution')
        else:
            ax2.text(0.5, 0.5, 'No False Positives', ha='center', va='center')
            ax2.set_title('False Positive Analysis')
        
        # Severity vs failure rate
        ax3 = axes[1, 0]
        severities = [case.get('severity', 'Unknown') for case in failure_cases]
        unique_severities, sev_counts = np.unique(severities, return_counts=True)
        
        colors_sev = self.colors['severity'][:len(unique_severities)]
        bars = ax3.bar(unique_severities, sev_counts, color=colors_sev)
        ax3.set_ylabel('Failure Count')
        ax3.set_title('Failures by Crack Severity')
        
        # Summary statistics
        ax4 = axes[1, 1]
        ax4.axis('off')
        
        total_failures = len(failure_cases)
        detection_failures = len([c for c in failure_cases if 'detection' in c.get('failure_type', '')])
        classification_failures = len([c for c in failure_cases if 'classification' in c.get('failure_type', '')])
        
        summary_text = f"""
        Failure Analysis Summary
        ─────────────────────────
        Total Failures: {total_failures}
        Detection Failures: {detection_failures}
        Classification Failures: {classification_failures}
        
        Most Common Failure: {max(set(failure_types), key=failure_types.count) if failure_types else 'N/A'}
        
        Avg Confidence (FP): {np.mean(fp_confidences):.3f} if fp_confidences else 'N/A'
        """
        ax4.text(0.1, 0.5, summary_text, fontsize=11, verticalalignment='center',
                bbox=dict(boxstyle="round,pad=0.3", facecolor=self.colors['secondary'], alpha=0.1))
        
        plt.tight_layout()
        save_path = self.output_dir / save_name
        plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        
        return save_path
    
    def create_interactive_dashboard(self, all_metrics: Dict) -> str:
        """Create an interactive HTML dashboard with Plotly"""
        
        # This would create a comprehensive interactive dashboard
        # For now, returning a placeholder
        dashboard_html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Crack Analysis Benchmark Results</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                .metric {{ margin: 20px 0; }}
                .highlight {{ color: #2E86AB; font-weight: bold; }}
            </style>
        </head>
        <body>
            <h1>🔍 Crack Analysis Pipeline Benchmark Results</h1>
            
            <div class="metric">
                <h2>📊 Overall Performance</h2>
                <p>Pipeline Accuracy: <span class="highlight">{all_metrics.get('pipeline', {}).get('overall_accuracy', 0):.3f}</span></p>
                <p>Processing Speed: <span class="highlight">{all_metrics.get('timing', {}).get('images_per_second', 0):.2f} images/sec</span></p>
            </div>
            
            <div class="metric">
                <h2>🎯 Detection Performance</h2>
                <p>Precision: <span class="highlight">{all_metrics.get('detection', {}).get('precision', 0):.3f}</span></p>
                <p>Recall: <span class="highlight">{all_metrics.get('detection', {}).get('recall', 0):.3f}</span></p>
                <p>F1-Score: <span class="highlight">{all_metrics.get('detection', {}).get('f1_score', 0):.3f}</span></p>
            </div>
            
            <div class="metric">
                <h2>🎨 Segmentation Performance</h2>
                <p>Mean IoU: <span class="highlight">{all_metrics.get('segmentation', {}).get('mean_iou', 0):.3f}</span></p>
                <p>Mean Dice: <span class="highlight">{all_metrics.get('segmentation', {}).get('mean_dice', 0):.3f}</span></p>
            </div>
            
            <div class="metric">
                <h2>🏷️ Classification Performance</h2>
                <p>Accuracy: <span class="highlight">{all_metrics.get('classification', {}).get('accuracy', 0):.3f}</span></p>
                <p>Macro F1: <span class="highlight">{all_metrics.get('classification', {}).get('f1_macro', 0):.3f}</span></p>
            </div>
            
            <p><em>Generated on {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}</em></p>
        </body>
        </html>
        """
        
        dashboard_path = self.output_dir / "benchmark_report.html"
        with open(dashboard_path, 'w') as f:
            f.write(dashboard_html)
        
        return str(dashboard_path)
