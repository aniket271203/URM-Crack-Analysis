"""
Evaluate batch results from pipeline evaluation
Analyzes intermediate results and generates comprehensive metrics
"""

import os
import sys
import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Any

# Set matplotlib backend before importing pyplot
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)

# Add paths
sys.path.append(str(Path(__file__).parent))

try:
    from evaluation_metrics import PipelineEvaluationMetrics
    from visualization import BenchmarkVisualizer
except ImportError:
    print("⚠️ Some modules not available, using simplified evaluation")


class BatchResultsEvaluator:
    """Evaluate batch results from pipeline evaluation"""
    
    def __init__(self, results_file: Path, output_dir: Path):
        self.results_file = Path(results_file)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load batch results
        self.batch_data = self._load_batch_results()
        
        # Initialize visualizer
        try:
            self.visualizer = BenchmarkVisualizer(self.output_dir / "plots")
        except:
            self.visualizer = None
    
    def _load_batch_results(self) -> Dict[str, Any]:
        """Load batch results from JSON file"""
        try:
            with open(self.results_file, 'r') as f:
                data = json.load(f)
            print(f"📊 Loaded batch results: {len(data['predictions'])} predictions")
            return data
        except Exception as e:
            print(f"❌ Error loading batch results: {e}")
            return {}
    
    def calculate_detection_metrics(self) -> Dict[str, Any]:
        """Calculate crack detection metrics (has_crack vs no_crack)"""
        
        predictions = self.batch_data.get('predictions', [])
        ground_truths = self.batch_data.get('ground_truths', [])
        
        if not predictions or not ground_truths:
            return {}
        
        # Extract detection predictions and ground truth
        y_pred_detection = [pred['has_crack'] for pred in predictions]
        y_true_detection = [gt['has_crack'] for gt in ground_truths]
        
        # Calculate metrics
        detection_metrics = {
            'accuracy': accuracy_score(y_true_detection, y_pred_detection),
            'precision': precision_score(y_true_detection, y_pred_detection, average='binary'),
            'recall': recall_score(y_true_detection, y_pred_detection, average='binary'),
            'f1': f1_score(y_true_detection, y_pred_detection, average='binary'),
            'total_samples': len(y_true_detection),
            'true_positives': sum(1 for i in range(len(y_true_detection)) if y_true_detection[i] and y_pred_detection[i]),
            'false_positives': sum(1 for i in range(len(y_true_detection)) if not y_true_detection[i] and y_pred_detection[i]),
            'true_negatives': sum(1 for i in range(len(y_true_detection)) if not y_true_detection[i] and not y_pred_detection[i]),
            'false_negatives': sum(1 for i in range(len(y_true_detection)) if y_true_detection[i] and not y_pred_detection[i])
        }
        
        return detection_metrics
    
    def calculate_classification_metrics(self) -> Dict[str, Any]:
        """Calculate crack type classification metrics"""
        
        predictions = self.batch_data.get('predictions', [])
        ground_truths = self.batch_data.get('ground_truths', [])
        
        if not predictions or not ground_truths:
            return {}
        
        # Extract classification predictions and ground truth (only for images with cracks)
        y_pred_class = []
        y_true_class = []
        confidences = []
        
        for pred, gt in zip(predictions, ground_truths):
            if gt['has_crack'] and pred['has_crack']:  # Only evaluate classification for detected cracks
                y_pred_class.append(pred['crack_type'])
                y_true_class.append(gt['crack_type'])
                confidences.append(pred.get('confidence', 0.0))
        
        if not y_pred_class:
            return {}
        
        # Get unique classes
        unique_classes = sorted(set(y_true_class + y_pred_class))
        
        # Calculate metrics
        classification_metrics = {
            'accuracy': accuracy_score(y_true_class, y_pred_class),
            'macro_f1': f1_score(y_true_class, y_pred_class, average='macro'),
            'weighted_f1': f1_score(y_true_class, y_pred_class, average='weighted'),
            'macro_precision': precision_score(y_true_class, y_pred_class, average='macro'),
            'macro_recall': recall_score(y_true_class, y_pred_class, average='macro'),
            'total_classified': len(y_pred_class),
            'average_confidence': np.mean(confidences),
            'classes': unique_classes,
            'confusion_matrix': confusion_matrix(y_true_class, y_pred_class, labels=unique_classes).tolist(),
            'classification_report': classification_report(y_true_class, y_pred_class, output_dict=True)
        }
        
        return classification_metrics
    
    def calculate_end_to_end_metrics(self) -> Dict[str, Any]:
        """Calculate end-to-end pipeline metrics"""
        
        predictions = self.batch_data.get('predictions', [])
        ground_truths = self.batch_data.get('ground_truths', [])
        
        if not predictions or not ground_truths:
            return {}
        
        # Overall accuracy: correct detection AND classification
        correct_overall = 0
        correct_detection = 0
        correct_classification = 0
        total_with_cracks = 0
        
        for pred, gt in zip(predictions, ground_truths):
            # Detection accuracy
            if pred['has_crack'] == gt['has_crack']:
                correct_detection += 1
                
                # If both agree there's a crack, check classification
                if gt['has_crack']:
                    total_with_cracks += 1
                    if pred['crack_type'] == gt['crack_type']:
                        correct_classification += 1
                        correct_overall += 1
                else:
                    # Correct no-crack prediction
                    correct_overall += 1
        
        total_samples = len(predictions)
        
        end_to_end_metrics = {
            'overall_accuracy': correct_overall / total_samples if total_samples > 0 else 0,
            'detection_accuracy': correct_detection / total_samples if total_samples > 0 else 0,
            'classification_accuracy': correct_classification / total_with_cracks if total_with_cracks > 0 else 0,
            'total_samples': total_samples,
            'samples_with_cracks': total_with_cracks,
            'correct_overall': correct_overall,
            'correct_detection': correct_detection,
            'correct_classification': correct_classification
        }
        
        return end_to_end_metrics
    
    def analyze_failure_cases(self) -> Dict[str, Any]:
        """Analyze failure cases"""
        
        failure_cases = self.batch_data.get('failure_cases', [])
        
        if not failure_cases:
            return {'total_failures': 0}
        
        # Analyze failure types
        failure_analysis = {
            'total_failures': len(failure_cases),
            'failure_types': {},
            'crack_type_confusion': {},
            'confidence_stats': {
                'mean': 0,
                'std': 0,
                'min': 0,
                'max': 0
            }
        }
        
        # Count failure types
        for failure in failure_cases:
            failure_type = failure.get('failure_type', 'unknown')
            failure_analysis['failure_types'][failure_type] = \
                failure_analysis['failure_types'].get(failure_type, 0) + 1
        
        # Analyze crack type confusions
        for failure in failure_cases:
            if failure.get('failure_type') == 'misclassification':
                gt_type = failure.get('gt_crack_type', 'unknown')
                pred_type = failure.get('pred_crack_type', 'unknown')
                confusion_key = f"{gt_type} → {pred_type}"
                failure_analysis['crack_type_confusion'][confusion_key] = \
                    failure_analysis['crack_type_confusion'].get(confusion_key, 0) + 1
        
        # Confidence statistics for failures
        confidences = [f.get('confidence', 0) for f in failure_cases if f.get('confidence') is not None]
        if confidences:
            failure_analysis['confidence_stats'] = {
                'mean': np.mean(confidences),
                'std': np.std(confidences),
                'min': np.min(confidences),
                'max': np.max(confidences)
            }
        
        return failure_analysis
    
    def create_visualizations(self, all_metrics: Dict[str, Any]):
        """Create comprehensive visualizations"""
        
        if not self.visualizer:
            print("⚠️ Visualizer not available, skipping plots")
            return []
        
        plot_files = []
        
        # 1. Detection metrics visualization
        if 'detection' in all_metrics:
            try:
                fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
                fig.suptitle('Crack Detection Performance', fontsize=16, fontweight='bold')
                
                # Metrics bar plot
                metrics = all_metrics['detection']
                metric_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
                metric_values = [metrics['accuracy'], metrics['precision'], metrics['recall'], metrics['f1']]
                
                bars = ax1.bar(metric_names, metric_values, color=['#2E86AB', '#A23B72', '#F18F01', '#C73E1D'])
                ax1.set_ylim(0, 1)
                ax1.set_ylabel('Score')
                ax1.set_title('Detection Metrics')
                
                # Add value labels on bars
                for bar, value in zip(bars, metric_values):
                    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                           f'{value:.3f}', ha='center', va='bottom')
                
                # Confusion matrix (simplified for detection)
                tp = metrics['true_positives']
                fp = metrics['false_positives']
                tn = metrics['true_negatives']
                fn = metrics['false_negatives']
                
                cm = np.array([[tn, fp], [fn, tp]])
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax2,
                           xticklabels=['No Crack', 'Crack'],
                           yticklabels=['No Crack', 'Crack'])
                ax2.set_title('Detection Confusion Matrix')
                ax2.set_ylabel('True')
                ax2.set_xlabel('Predicted')
                
                # Sample distribution
                labels = ['True Positives', 'False Positives', 'True Negatives', 'False Negatives']
                sizes = [tp, fp, tn, fn]
                colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D']
                
                ax3.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
                ax3.set_title('Detection Results Distribution')
                
                # Performance summary
                ax4.axis('off')
                summary_text = f"""
Detection Summary:
Total Samples: {metrics['total_samples']}
Correctly Detected: {tp + tn}
Detection Rate: {(tp + tn) / metrics['total_samples']:.2%}

True Positives: {tp}
False Positives: {fp}
True Negatives: {tn}
False Negatives: {fn}
                """
                ax4.text(0.1, 0.5, summary_text, fontsize=12, verticalalignment='center',
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
                
                plt.tight_layout()
                detection_plot = self.output_dir / "plots" / "detection_performance.png"
                plt.savefig(detection_plot, dpi=300, bbox_inches='tight')
                plt.close()
                plot_files.append(detection_plot)
                print(f"   ✅ Detection plot: {detection_plot.name}")
                
            except Exception as e:
                print(f"   ⚠️ Error creating detection plot: {e}")
        
        # 2. Classification metrics visualization
        if 'classification' in all_metrics and all_metrics['classification']:
            try:
                class_metrics = all_metrics['classification']
                
                fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
                fig.suptitle('Crack Classification Performance', fontsize=16, fontweight='bold')
                
                # Classification metrics
                metric_names = ['Accuracy', 'Macro F1', 'Weighted F1', 'Macro Precision', 'Macro Recall']
                metric_values = [
                    class_metrics['accuracy'],
                    class_metrics['macro_f1'],
                    class_metrics['weighted_f1'],
                    class_metrics['macro_precision'],
                    class_metrics['macro_recall']
                ]
                
                bars = ax1.barh(metric_names, metric_values, color='skyblue')
                ax1.set_xlim(0, 1)
                ax1.set_xlabel('Score')
                ax1.set_title('Classification Metrics')
                
                # Add value labels
                for bar, value in zip(bars, metric_values):
                    ax1.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                           f'{value:.3f}', ha='left', va='center')
                
                # Confusion matrix
                cm = np.array(class_metrics['confusion_matrix'])
                classes = class_metrics['classes']
                
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax2,
                           xticklabels=classes, yticklabels=classes)
                ax2.set_title('Classification Confusion Matrix')
                ax2.set_ylabel('True')
                ax2.set_xlabel('Predicted')
                
                # Per-class F1 scores
                if 'classification_report' in class_metrics:
                    report = class_metrics['classification_report']
                    class_f1s = []
                    class_names = []
                    
                    for class_name in classes:
                        if class_name in report:
                            class_f1s.append(report[class_name]['f1-score'])
                            class_names.append(class_name)
                    
                    if class_f1s:
                        bars = ax3.bar(class_names, class_f1s, color='lightcoral')
                        ax3.set_ylim(0, 1)
                        ax3.set_ylabel('F1-Score')
                        ax3.set_title('Per-Class F1 Scores')
                        ax3.tick_params(axis='x', rotation=45)
                        
                        # Add value labels
                        for bar, value in zip(bars, class_f1s):
                            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                                   f'{value:.3f}', ha='center', va='bottom')
                
                # Classification summary
                ax4.axis('off')
                summary_text = f"""
Classification Summary:
Total Classified: {class_metrics['total_classified']}
Accuracy: {class_metrics['accuracy']:.2%}
Average Confidence: {class_metrics['average_confidence']:.3f}

Classes: {', '.join(classes)}
                """
                ax4.text(0.1, 0.5, summary_text, fontsize=12, verticalalignment='center',
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.7))
                
                plt.tight_layout()
                classification_plot = self.output_dir / "plots" / "classification_performance.png"
                plt.savefig(classification_plot, dpi=300, bbox_inches='tight')
                plt.close()
                plot_files.append(classification_plot)
                print(f"   ✅ Classification plot: {classification_plot.name}")
                
            except Exception as e:
                print(f"   ⚠️ Error creating classification plot: {e}")
        
        # 3. Failure analysis plot
        if 'failure_analysis' in all_metrics and all_metrics['failure_analysis']['total_failures'] > 0:
            try:
                failure_data = all_metrics['failure_analysis']
                
                fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
                fig.suptitle('Failure Case Analysis', fontsize=16, fontweight='bold')
                
                # Failure types distribution
                failure_types = failure_data['failure_types']
                if failure_types:
                    ax1.pie(failure_types.values(), labels=failure_types.keys(), autopct='%1.1f%%')
                    ax1.set_title('Failure Types Distribution')
                
                # Crack type confusion
                confusions = failure_data['crack_type_confusion']
                if confusions:
                    confusion_labels = list(confusions.keys())
                    confusion_counts = list(confusions.values())
                    
                    bars = ax2.barh(confusion_labels, confusion_counts, color='salmon')
                    ax2.set_xlabel('Count')
                    ax2.set_title('Crack Type Confusions')
                    
                    # Add value labels
                    for bar, count in zip(bars, confusion_counts):
                        ax2.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height()/2,
                               str(count), ha='left', va='center')
                
                # Confidence distribution for failures
                if 'confidence_stats' in failure_data:
                    stats = failure_data['confidence_stats']
                    ax3.axis('off')
                    stats_text = f"""
Failure Confidence Statistics:
Mean: {stats['mean']:.3f}
Std: {stats['std']:.3f}
Min: {stats['min']:.3f}
Max: {stats['max']:.3f}
                    """
                    ax3.text(0.1, 0.5, stats_text, fontsize=12, verticalalignment='center',
                            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow"))
                
                # Overall failure summary
                ax4.axis('off')
                total_failures = failure_data['total_failures']
                total_samples = all_metrics.get('end_to_end', {}).get('total_samples', 1)
                failure_rate = total_failures / total_samples
                
                summary_text = f"""
Failure Summary:
Total Failures: {total_failures}
Total Samples: {total_samples}
Failure Rate: {failure_rate:.2%}
                """
                ax4.text(0.1, 0.5, summary_text, fontsize=12, verticalalignment='center',
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightcoral", alpha=0.7))
                
                plt.tight_layout()
                failure_plot = self.output_dir / "plots" / "failure_analysis.png"
                plt.savefig(failure_plot, dpi=300, bbox_inches='tight')
                plt.close()
                plot_files.append(failure_plot)
                print(f"   ✅ Failure analysis plot: {failure_plot.name}")
                
            except Exception as e:
                print(f"   ⚠️ Error creating failure analysis plot: {e}")
        
        return plot_files
    
    def generate_report(self, all_metrics: Dict[str, Any]) -> Path:
        """Generate a comprehensive evaluation report"""
        
        report_path = self.output_dir / "batch_evaluation_report.md"
        
        with open(report_path, 'w') as f:
            f.write("# Crack Analysis Pipeline - Batch Evaluation Report\n\n")
            f.write(f"**Evaluation Date:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"**Batch Results File:** {self.results_file.name}\n")
            f.write(f"**Processed Batches:** {self.batch_data.get('processed_batches', 'Unknown')}\n\n")
            
            # Executive Summary
            f.write("## Executive Summary\n\n")
            if 'end_to_end' in all_metrics:
                e2e = all_metrics['end_to_end']
                f.write(f"- **Overall Accuracy:** {e2e['overall_accuracy']:.1%}\n")
                f.write(f"- **Detection Accuracy:** {e2e['detection_accuracy']:.1%}\n")
                f.write(f"- **Classification Accuracy:** {e2e['classification_accuracy']:.1%}\n")
                f.write(f"- **Total Samples Processed:** {e2e['total_samples']}\n")
                f.write(f"- **Samples with Cracks:** {e2e['samples_with_cracks']}\n\n")
            
            # Detection Performance
            if 'detection' in all_metrics:
                f.write("## Detection Performance\n\n")
                det = all_metrics['detection']
                f.write(f"| Metric | Value |\n")
                f.write(f"|--------|-------|\n")
                f.write(f"| Accuracy | {det['accuracy']:.3f} |\n")
                f.write(f"| Precision | {det['precision']:.3f} |\n")
                f.write(f"| Recall | {det['recall']:.3f} |\n")
                f.write(f"| F1-Score | {det['f1']:.3f} |\n")
                f.write(f"| True Positives | {det['true_positives']} |\n")
                f.write(f"| False Positives | {det['false_positives']} |\n")
                f.write(f"| True Negatives | {det['true_negatives']} |\n")
                f.write(f"| False Negatives | {det['false_negatives']} |\n\n")
            
            # Classification Performance
            if 'classification' in all_metrics and all_metrics['classification']:
                f.write("## Classification Performance\n\n")
                cls = all_metrics['classification']
                f.write(f"| Metric | Value |\n")
                f.write(f"|--------|-------|\n")
                f.write(f"| Accuracy | {cls['accuracy']:.3f} |\n")
                f.write(f"| Macro F1 | {cls['macro_f1']:.3f} |\n")
                f.write(f"| Weighted F1 | {cls['weighted_f1']:.3f} |\n")
                f.write(f"| Macro Precision | {cls['macro_precision']:.3f} |\n")
                f.write(f"| Macro Recall | {cls['macro_recall']:.3f} |\n")
                f.write(f"| Average Confidence | {cls['average_confidence']:.3f} |\n")
                f.write(f"| Total Classified | {cls['total_classified']} |\n\n")
            
            # Failure Analysis
            if 'failure_analysis' in all_metrics:
                f.write("## Failure Analysis\n\n")
                fail = all_metrics['failure_analysis']
                f.write(f"**Total Failures:** {fail['total_failures']}\n\n")
                
                if fail['failure_types']:
                    f.write("### Failure Types:\n")
                    for failure_type, count in fail['failure_types'].items():
                        f.write(f"- **{failure_type.title()}:** {count}\n")
                    f.write("\n")
                
                if fail['crack_type_confusion']:
                    f.write("### Most Common Misclassifications:\n")
                    sorted_confusions = sorted(fail['crack_type_confusion'].items(), 
                                             key=lambda x: x[1], reverse=True)
                    for confusion, count in sorted_confusions[:5]:
                        f.write(f"- **{confusion}:** {count} times\n")
                    f.write("\n")
        
        return report_path
    
    def run_evaluation(self) -> Dict[str, Any]:
        """Run complete batch evaluation"""
        
        print("🔍 Evaluating Batch Results")
        print("=" * 50)
        
        # Calculate all metrics
        all_metrics = {}
        
        # 1. Detection metrics
        print("\n📊 Calculating detection metrics...")
        all_metrics['detection'] = self.calculate_detection_metrics()
        
        # 2. Classification metrics
        print("📊 Calculating classification metrics...")
        all_metrics['classification'] = self.calculate_classification_metrics()
        
        # 3. End-to-end metrics
        print("📊 Calculating end-to-end metrics...")
        all_metrics['end_to_end'] = self.calculate_end_to_end_metrics()
        
        # 4. Failure analysis
        print("📊 Analyzing failure cases...")
        all_metrics['failure_analysis'] = self.analyze_failure_cases()
        
        # 5. Create visualizations
        print("\n📈 Creating visualizations...")
        plot_files = self.create_visualizations(all_metrics)
        
        # 6. Generate report
        print("📄 Generating evaluation report...")
        report_path = self.generate_report(all_metrics)
        
        # 7. Save detailed results
        results_path = self.output_dir / "detailed_metrics.json"
        with open(results_path, 'w') as f:
            json.dump(all_metrics, f, indent=2, default=str)
        
        # Print summary
        print("\n✅ Batch Evaluation Complete!")
        print("=" * 50)
        
        if 'end_to_end' in all_metrics:
            e2e = all_metrics['end_to_end']
            print(f"📊 **Results Summary:**")
            print(f"   Overall Accuracy: {e2e['overall_accuracy']:.1%}")
            print(f"   Detection Accuracy: {e2e['detection_accuracy']:.1%}")
            print(f"   Classification Accuracy: {e2e['classification_accuracy']:.1%}")
            print(f"   Total Samples: {e2e['total_samples']}")
        
        if 'failure_analysis' in all_metrics:
            fail = all_metrics['failure_analysis']
            print(f"   Total Failures: {fail['total_failures']}")
        
        print(f"\n📁 **Output Files:**")
        print(f"   Report: {report_path}")
        print(f"   Detailed Metrics: {results_path}")
        for plot_file in plot_files:
            print(f"   Plot: {plot_file}")
        
        return all_metrics


def main():
    """Main function to evaluate batch results"""
    
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate batch results from pipeline evaluation")
    parser.add_argument("--results", type=str, 
                       default="pipeline_evaluation/intermediate_results_batch_15.json",
                       help="Path to batch results JSON file")
    parser.add_argument("--output", type=str, default="batch_evaluation",
                       help="Output directory for evaluation results")
    
    args = parser.parse_args()
    
    # Setup paths
    benchmark_dir = Path(__file__).parent
    results_file = benchmark_dir / args.results
    output_dir = benchmark_dir / args.output
    
    if not results_file.exists():
        print(f"❌ Results file not found: {results_file}")
        return 1
    
    try:
        # Run evaluation
        evaluator = BatchResultsEvaluator(results_file, output_dir)
        metrics = evaluator.run_evaluation()
        
        print(f"\n🎯 Evaluation completed successfully!")
        return 0
        
    except Exception as e:
        print(f"❌ Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
