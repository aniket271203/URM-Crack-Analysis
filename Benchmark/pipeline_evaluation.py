"""
End-to-End Pipeline Evaluation Script
Runs comprehensive evaluation of the complete crack analysis pipeline
"""

import os
import sys
import json
import time
import argparse
import warnings
from pathlib import Path
from typing import Dict, List, Any, Tuple
import pandas as pd
import numpy as np
import cv2
from tqdm import tqdm
import psutil
import threading
import gc


def json_serializable(obj):
    """Convert numpy types to native Python types for JSON serialization"""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (list, tuple)):
        return [json_serializable(item) for item in obj]
    elif isinstance(obj, dict):
        return {key: json_serializable(value) for key, value in obj.items()}
    else:
        return obj


def safe_json_dump(data, file_path, **kwargs):
    """Safely dump data to JSON file with numpy type conversion"""
    try:
        serializable_data = json_serializable(data)
        with open(file_path, 'w') as f:
            json.dump(serializable_data, f, **kwargs)
        return True
    except Exception as e:
        print(f"⚠️ Error saving JSON to {file_path}: {e}")
        return False

# Import torch for memory management
try:
    import torch
except ImportError:
    torch = None

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Add paths
sys.path.append(str(Path(__file__).parent.parent))
sys.path.append(str(Path(__file__).parent))

try:
    # Import pipeline components
    from pipeline_orchestrator import CrackAnalysisPipeline
    
    # Import benchmark modules
    from config_benchmark import (
        MODEL_PATHS, DETECTION_THRESHOLDS, METRICS_CONFIG, 
        HARDWARE_CONFIG, BENCHMARK_CONFIG, TEST_DATA_DIR, RESULTS_DIR
    )
    from evaluation_metrics import PipelineEvaluationMetrics
    from visualization import BenchmarkVisualizer
    
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure all dependencies are installed and paths are correct")
    sys.exit(1)


class MemoryManager:
    """Memory management utilities to prevent kernel kills"""
    
    @staticmethod
    def get_memory_usage():
        """Get current memory usage percentage"""
        return psutil.virtual_memory().percent
    
    @staticmethod
    def get_available_memory_gb():
        """Get available memory in GB"""
        return psutil.virtual_memory().available / (1024**3)
    
    @staticmethod
    def cleanup_memory():
        """Force memory cleanup"""
        gc.collect()
        if torch and torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    @staticmethod
    def aggressive_cleanup():
        """Perform aggressive memory cleanup"""
        # Multiple rounds of garbage collection
        for _ in range(3):
            gc.collect()
        
        # Clear CUDA cache if available
        if torch and torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    
    @staticmethod
    def check_memory_threshold(threshold=85):
        """Check if memory usage exceeds threshold and cleanup if needed"""
        memory_usage = MemoryManager.get_memory_usage()
        if memory_usage > threshold:
            print(f"⚠️ High memory usage ({memory_usage:.1f}%), performing cleanup...")
            MemoryManager.cleanup_memory()
            return True
        return False
    
    @staticmethod
    def is_memory_critical(threshold=90):
        """Check if memory usage is critically high"""
        return MemoryManager.get_memory_usage() > threshold
    
    @staticmethod
    def safe_memory_check(operation_name="operation", threshold=80):
        """Check if it's safe to proceed with an operation based on memory usage"""
        memory_usage = MemoryManager.get_memory_usage()
        if memory_usage > threshold:
            print(f"⚠️ Memory usage too high ({memory_usage:.1f}%) for {operation_name}")
            return False
        return True


class PerformanceMonitor:
    """Monitor CPU, GPU, and memory usage during benchmarking"""
    
    def __init__(self):
        self.monitoring = False
        self.stats = {
            'cpu_usage': [],
            'memory_usage': [],
            'gpu_usage': [],
            'timestamps': []
        }
        self.monitor_thread = None
    
    def start_monitoring(self):
        """Start performance monitoring in a separate thread"""
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop)
        self.monitor_thread.start()
    
    def stop_monitoring(self):
        """Stop performance monitoring"""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join()
    
    def _monitor_loop(self):
        """Monitor performance metrics with limited history"""
        max_samples = 30  # Limit stored samples to prevent memory growth
        while self.monitoring:
            timestamp = time.time()
            
            # CPU usage
            cpu_percent = psutil.cpu_percent()
            
            # Memory usage
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            
            # GPU usage (if available)
            gpu_percent = 0
            try:
                if torch and torch.cuda.is_available():
                    memory_used = torch.cuda.memory_allocated(0) if torch.cuda.device_count() > 0 else 0
                    memory_total = torch.cuda.max_memory_allocated(0) if torch.cuda.device_count() > 0 else 1
                    gpu_percent = (memory_used / memory_total) * 100 if memory_total > 0 else 0
            except Exception:
                gpu_percent = 0
            
            # Add new data
            self.stats['cpu_usage'].append(cpu_percent)
            self.stats['memory_usage'].append(memory_percent)
            self.stats['gpu_usage'].append(gpu_percent)
            self.stats['timestamps'].append(timestamp)
            
            # Limit history to prevent memory accumulation
            for key in self.stats:
                if len(self.stats[key]) > max_samples:
                    self.stats[key] = self.stats[key][-max_samples:]
            
            time.sleep(1.0)  # Monitor every second
    
    def get_summary(self) -> Dict[str, float]:
        """Get summary of performance metrics"""
        if not self.stats['cpu_usage']:
            return {}
        
        return {
            'avg_cpu_usage': np.mean(self.stats['cpu_usage']),
            'max_cpu_usage': np.max(self.stats['cpu_usage']),
            'avg_memory_usage': np.mean(self.stats['memory_usage']),
            'max_memory_usage': np.max(self.stats['memory_usage']),
            'avg_gpu_usage': np.mean(self.stats['gpu_usage']) if self.stats['gpu_usage'] else 0,
            'max_gpu_usage': np.max(self.stats['gpu_usage']) if self.stats['gpu_usage'] else 0,
        }


class PipelineEvaluator:
    """Evaluator for the complete end-to-end pipeline"""
    
    def __init__(self, test_data_dir: Path, output_dir: Path):
        self.test_data_dir = Path(test_data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.pipeline = None
        self.performance_monitor = PerformanceMonitor()
        
        # Load test data
        self.test_images = []
        self.ground_truth = None
        self._load_test_data()
        
        # Initialize metrics calculators
        self.pipeline_metrics = PipelineEvaluationMetrics()
        
        # Initialize visualizer
        self.visualizer = BenchmarkVisualizer(self.output_dir / "plots")
        
        print(f"🚀 Pipeline evaluator initialized with {len(self.test_images)} test images")
    
    def _load_test_data(self):
        """Load test images and ground truth data with memory management"""
        
        # Load ground truth CSV
        ground_truth_path = self.test_data_dir / "metadata" / "ground_truth.csv"
        if ground_truth_path.exists():
            self.ground_truth = pd.read_csv(ground_truth_path)
            print(f"📊 Loaded ground truth for {len(self.ground_truth)} images")
        else:
            raise FileNotFoundError(f"Ground truth file not found: {ground_truth_path}")
        
        # Check available memory and limit test images heavily for pipeline evaluation
        available_memory_gb = psutil.virtual_memory().available / (1024**3)
        if available_memory_gb < 4:  # Less than 4GB available
            max_images = min(10, len(self.ground_truth))  # Very conservative limit
            print(f"⚠️ Very limited memory detected ({available_memory_gb:.1f}GB), limiting to {max_images} images")
            self.ground_truth = self.ground_truth.head(max_images)
        elif available_memory_gb < 8:  # Less than 8GB available
            max_images = min(20, len(self.ground_truth))  # Very conservative limit
            print(f"⚠️ Limited memory detected ({available_memory_gb:.1f}GB), limiting to {max_images} images")
            self.ground_truth = self.ground_truth.head(max_images)
        elif available_memory_gb < 16:  # Less than 16GB available
            max_images = min(40, len(self.ground_truth))  # Conservative limit
            print(f"⚠️ Moderate memory detected ({available_memory_gb:.1f}GB), limiting to {max_images} images")
            self.ground_truth = self.ground_truth.head(max_images)
        
        # Load test images
        images_dir = self.test_data_dir / "images"
        for filename in self.ground_truth['filename']:
            image_path = images_dir / filename
            if image_path.exists():
                self.test_images.append(image_path)
        
        print(f"📂 Found {len(self.test_images)} test images for pipeline evaluation")
    
    def setup_pipeline(self):
        """Initialize the crack analysis pipeline with conservative memory settings"""
        
        print("🔧 Setting up complete crack analysis pipeline...")
        
        try:
            # Check memory before setup
            if not MemoryManager.safe_memory_check("pipeline setup", threshold=75):
                print("⚠️ Insufficient memory for pipeline setup, performing aggressive cleanup...")
                MemoryManager.aggressive_cleanup()
                
                # Check again after cleanup
                if not MemoryManager.safe_memory_check("pipeline setup", threshold=75):
                    raise MemoryError("Insufficient memory to initialize pipeline even after cleanup")
            
            # Initialize main pipeline with memory optimization
            self.pipeline = CrackAnalysisPipeline(
                yolo_model_path=str(MODEL_PATHS['yolo']),
                segmentation_model_path=str(MODEL_PATHS['segmentation']),
                classification_model_path=str(MODEL_PATHS['classification']),
                rag_data_dir=str(MODEL_PATHS['rag_data']) if MODEL_PATHS['rag_data'].exists() else None,
                device=HARDWARE_CONFIG['device']
            )
            
            # Force initial cleanup
            MemoryManager.cleanup_memory()
            
            print("✅ Complete pipeline setup finished")
            
        except Exception as e:
            print(f"❌ Error setting up pipeline: {e}")
            # Cleanup in case of partial initialization
            if hasattr(self, 'pipeline'):
                self.pipeline = None
            MemoryManager.aggressive_cleanup()
            raise
    
    def evaluate_end_to_end_pipeline(self) -> Tuple[Dict[str, Any], List[Dict]]:
        """Evaluate the complete end-to-end pipeline with careful memory management"""
        
        print("\n🔄 Evaluating End-to-End Pipeline...")
        
        predictions = []
        ground_truths = []
        timing_data = []
        failure_cases = []
        skipped_batches = []
        skipped_images = []
        
        # Memory safety check
        initial_memory = MemoryManager.get_memory_usage()
        print(f"🔍 Initial memory usage: {initial_memory:.1f}%")
        
        if initial_memory > 70:
            print("⚠️ High initial memory usage, performing cleanup before starting...")
            MemoryManager.cleanup_memory()
        
        # Process images in very small batches with aggressive memory management
        batch_size = 2  # Even smaller batches to prevent memory overflow
        
        for batch_start in range(0, len(self.test_images), batch_size):
            # Check memory before each batch
            current_memory = MemoryManager.get_memory_usage()
            if current_memory > 85:
                print(f"⚠️ High memory usage ({current_memory:.1f}%), skipping batch {batch_start//batch_size + 1}")
                # Skip this batch due to memory constraints
                batch_end = min(batch_start + batch_size, len(self.test_images))
                skipped_batch_info = {
                    'batch_number': batch_start//batch_size + 1,
                    'batch_range': f"{batch_start}-{batch_end-1}",
                    'reason': 'memory_overflow_prevention',
                    'memory_usage': current_memory
                }
                skipped_batches.append(skipped_batch_info)
                
                # Perform aggressive cleanup and continue with next batch
                MemoryManager.cleanup_memory()
                print(f"   Batch {batch_start//batch_size + 1} skipped, continuing with next batch...")
                continue
            
            batch_end = min(batch_start + batch_size, len(self.test_images))
            batch = self.test_images[batch_start:batch_end]
            
            print(f"Processing pipeline batch {batch_start//batch_size + 1}/{(len(self.test_images)-1)//batch_size + 1} (Memory: {current_memory:.1f}%)")
            
            # Reset performance monitor for each batch to prevent accumulation
            self.performance_monitor = PerformanceMonitor()
            
            
            for idx in tqdm(range(batch_start, batch_end), desc=f"Pipeline Batch {batch_start//batch_size + 1}"):
                image_path = self.test_images[idx]
                
                try:
                    # Check memory before processing each image
                    current_memory = MemoryManager.get_memory_usage()
                    if current_memory > 80:
                        print(f"⚠️ High memory usage ({current_memory:.1f}%), skipping image {idx}")
                        # Skip this image due to memory constraints
                        skipped_image_info = {
                            'image_index': idx,
                            'filename': str(self.ground_truth.iloc[idx]['filename']) if idx < len(self.ground_truth) else f"image_{idx}",
                            'reason': 'memory_overflow_prevention',
                            'memory_usage': current_memory
                        }
                        skipped_images.append(skipped_image_info)
                        
                        # Perform cleanup and continue with next image
                        MemoryManager.cleanup_memory()
                        continue
                    
                    # Aggressive memory cleanup before each image
                    MemoryManager.cleanup_memory()
                    if MemoryManager.check_memory_threshold(threshold=65):  # Lower threshold
                        print(f"   Memory cleaned at image {idx}")
                    
                    # Additional memory check with pipeline reset if needed
                    current_memory = MemoryManager.get_memory_usage()
                    if current_memory > 75:
                        print(f"⚠️ High memory usage ({current_memory:.1f}%), attempting pipeline reset...")
                        try:
                            # Temporarily clear pipeline to free memory
                            if hasattr(self, 'pipeline') and self.pipeline is not None:
                                temp_pipeline = self.pipeline
                                self.pipeline = None
                                del temp_pipeline
                                gc.collect()
                                # Reinitialize pipeline
                                self.setup_pipeline()
                        except Exception as reset_error:
                            print(f"⚠️ Pipeline reset failed: {reset_error}, skipping image {idx}")
                            skipped_image_info = {
                                'image_index': idx,
                                'filename': str(self.ground_truth.iloc[idx]['filename']) if idx < len(self.ground_truth) else f"image_{idx}",
                                'reason': 'pipeline_reset_failed',
                                'error': str(reset_error)
                            }
                            skipped_images.append(skipped_image_info)
                            continue
                    
                    # Get ground truth
                    gt_row = self.ground_truth.iloc[idx]
                    gt_data = {
                        'has_crack': bool(gt_row['has_crack']),  # Convert to native Python bool
                        'crack_type': str(gt_row['crack_type']) if gt_row['has_crack'] else 'none',
                        'severity': str(gt_row.get('severity', 'unknown')),
                        'filename': str(gt_row['filename'])
                    }
                    ground_truths.append(gt_data)
                    
                    # Run pipeline with timing
                    start_time = time.time()
                    
                    # Start monitoring only for this specific image
                    monitor = PerformanceMonitor()
                    monitor.start_monitoring()
                    
                    result = self.pipeline.process_image(
                        str(image_path), 
                        use_rag=False,  # Disable RAG for faster processing
                        save_intermediate=False
                    )
                    
                    monitor.stop_monitoring()
                    total_time = time.time() - start_time
                    
                    # Get performance stats and immediately clear monitor
                    performance_stats = monitor.get_summary()
                    del monitor  # Immediately delete to free memory
                    
                    # Extract prediction
                    pred_data = {
                        'success': bool(result.get('success', False)),
                        'has_crack': bool(len(result.get('detection', {}).get('detections', [])) > 0),
                        'crack_type': 'none',
                        'confidence': float(0),
                        'detections': int(len(result.get('detection', {}).get('detections', [])))
                    }
                    
                    # If crack detected, get classification
                    if pred_data['has_crack'] and result.get('classification'):
                        pred_data['crack_type'] = str(result['classification'].get('crack_type', 'unknown'))
                        pred_data['confidence'] = float(result['classification'].get('confidence', 0))
                    
                    predictions.append(pred_data)
                    
                    # Record timing with minimal data
                    timing_data.append({
                        'total_time': float(total_time),
                        'detection_time': float(total_time * 0.3),  # Rough estimates
                        'segmentation_time': float(total_time * 0.4),
                        'classification_time': float(total_time * 0.3),
                        'avg_memory_usage': float(performance_stats.get('avg_memory_usage', 0))
                    })
                    
                    # Clear large objects immediately
                    if 'detection' in result:
                        result['detection'] = {'detections': len(result['detection'].get('detections', []))}
                    del result  # Clear result object
                    
                    # Identify failure cases
                    if gt_data['has_crack'] != pred_data['has_crack']:
                        failure_type = 'false_positive' if pred_data['has_crack'] else 'false_negative'
                        failure_cases.append({
                            'filename': str(gt_data['filename']),
                            'failure_type': str(failure_type),
                            'gt_crack_type': str(gt_data['crack_type']),
                            'pred_crack_type': str(pred_data['crack_type']),
                            'confidence': float(pred_data['confidence']),
                            'severity': str(gt_data['severity'])
                        })
                    elif (gt_data['has_crack'] and pred_data['has_crack'] and 
                          gt_data['crack_type'] != pred_data['crack_type']):
                        failure_cases.append({
                            'filename': str(gt_data['filename']),
                            'failure_type': 'misclassification',
                            'gt_crack_type': str(gt_data['crack_type']),
                            'pred_crack_type': str(pred_data['crack_type']),
                            'confidence': float(pred_data['confidence']),
                            'severity': str(gt_data['severity'])
                        })
                    
                except Exception as e:
                    print(f"⚠️ Error in pipeline for {image_path}: {e}")
                    # Log the error and skip this image instead of stopping
                    skipped_image_info = {
                        'image_index': idx,
                        'filename': str(self.ground_truth.iloc[idx]['filename']) if idx < len(self.ground_truth) else f"image_{idx}",
                        'reason': 'pipeline_processing_error',
                        'error': str(e)
                    }
                    skipped_images.append(skipped_image_info)
                    
                    # Perform cleanup after error
                    MemoryManager.cleanup_memory()
                    continue  # Skip to next image instead of adding dummy data
                
                # Cleanup after every image in pipeline evaluation
                gc.collect()
                if torch and torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            # Aggressive cleanup after each batch
            MemoryManager.cleanup_memory()
            print(f"Batch {batch_start//batch_size + 1} complete, memory cleaned")
            
            # Save intermediate results to prevent loss
            if (batch_start // batch_size + 1) % 5 == 0:  # Every 5 batches
                print(f"💾 Saving intermediate results at batch {batch_start//batch_size + 1}")
                intermediate_file = self.output_dir / f"intermediate_results_batch_{batch_start//batch_size + 1}.json"
                safe_json_dump({
                    'predictions': predictions,
                    'ground_truths': ground_truths,
                    'failure_cases': failure_cases,
                    'skipped_batches': skipped_batches,
                    'skipped_images': skipped_images,
                    'processed_batches': batch_start//batch_size + 1
                }, intermediate_file, indent=2)
        
        # Calculate pipeline metrics with whatever data we have
        if not predictions or not ground_truths:
            print("⚠️ No data collected, cannot calculate metrics")
            print(f"   Skipped {len(skipped_batches)} batches and {len(skipped_images)} images due to memory/errors")
            return {
                'skipped_batches': len(skipped_batches),
                'skipped_images': len(skipped_images),
                'total_processed': 0
            }, []
        
        try:
            pipeline_metrics = self.pipeline_metrics.calculate_end_to_end_metrics(predictions, ground_truths)
            timing_metrics = self.pipeline_metrics.calculate_processing_metrics(timing_data)
        except Exception as e:
            print(f"⚠️ Error calculating metrics: {e}")
            # Return minimal metrics
            pipeline_metrics = {
                'overall_accuracy': 0,
                'detection_accuracy': 0,
                'classification_accuracy': 0
            }
            timing_metrics = {'mean_total_time': 0, 'images_per_second': 0}
        
        # Combine results with skip information
        pipeline_results = {
            **pipeline_metrics,
            'timing_metrics': timing_metrics,
            'total_failure_cases': len(failure_cases),
            'skipped_batches': len(skipped_batches),
            'skipped_images': len(skipped_images),
            'total_processed': len(predictions),
            'total_attempted': len(self.test_images)
        }
        
        # Save results
        results_file = self.output_dir / "pipeline_evaluation_results.json"
        if not safe_json_dump(pipeline_results, results_file, indent=2):
            print(f"⚠️ Could not save results to {results_file}")
        
        failure_file = self.output_dir / "pipeline_failure_cases.json"
        if not safe_json_dump(failure_cases, failure_file, indent=2):
            print(f"⚠️ Could not save failure cases to {failure_file}")
        
        # Save skip information
        skip_file = self.output_dir / "pipeline_skipped_items.json"
        skip_info = {
            'skipped_batches': skipped_batches,
            'skipped_images': skipped_images,
            'summary': {
                'total_skipped_batches': len(skipped_batches),
                'total_skipped_images': len(skipped_images),
                'skip_reasons': {}
            }
        }
        
        # Analyze skip reasons
        for skip in skipped_batches + skipped_images:
            reason = skip.get('reason', 'unknown')
            skip_info['summary']['skip_reasons'][reason] = skip_info['summary']['skip_reasons'].get(reason, 0) + 1
        
        if not safe_json_dump(skip_info, skip_file, indent=2):
            print(f"⚠️ Could not save skip information to {skip_file}")
        
        print(f"📊 Pipeline Evaluation Metrics (processed {len(predictions)}/{len(self.test_images)} images):")
        print(f"   Overall Accuracy: {pipeline_metrics.get('overall_accuracy', 0):.3f}")
        print(f"   Detection Accuracy: {pipeline_metrics.get('detection_accuracy', 0):.3f}")
        print(f"   Classification Accuracy: {pipeline_metrics.get('classification_accuracy', 0):.3f}")
        print(f"   Avg Processing Time: {timing_metrics.get('mean_total_time', 0):.3f}s")
        print(f"   Throughput: {timing_metrics.get('images_per_second', 0):.2f} images/sec")
        print(f"   Failure Cases: {len(failure_cases)}")
        print(f"   Skipped Batches: {len(skipped_batches)}")
        print(f"   Skipped Images: {len(skipped_images)}")
        if skipped_batches or skipped_images:
            print(f"   ⚠️ Some items were skipped due to memory constraints or errors")
            print(f"   📋 Skip details saved to: {skip_file.name}")
        
        return pipeline_results, failure_cases
    
    def generate_pipeline_visualizations(self, pipeline_metrics: Dict[str, Any], failure_cases: List[Dict]):
        """Generate visualizations for pipeline evaluation"""
        
        print("\n📊 Generating Pipeline Visualizations...")
        
        plot_paths = []
        
        # Pipeline overview plot
        timing_metrics = pipeline_metrics.get('timing_metrics', {})
        path = self.visualizer.plot_pipeline_overview(pipeline_metrics, timing_metrics)
        plot_paths.append(path)
        print(f"   ✅ Pipeline overview: {path.name}")
        
        # Failure analysis plot
        path = self.visualizer.plot_failure_analysis(failure_cases)
        plot_paths.append(path)
        print(f"   ✅ Failure analysis: {path.name}")
        
        # Performance monitoring plot
        if timing_metrics:
            try:
                path = self.visualizer.plot_performance_monitoring(timing_metrics)
                plot_paths.append(path)
                print(f"   ✅ Performance monitoring: {path.name}")
            except:
                print("   ⚠️ Could not create performance monitoring plot")
        
        # Interactive dashboard
        all_metrics = {'pipeline': pipeline_metrics}
        dashboard_path = self.visualizer.create_interactive_dashboard(all_metrics)
        print(f"   ✅ Interactive dashboard: {Path(dashboard_path).name}")
        
        return plot_paths, dashboard_path
    
    def run_pipeline_evaluation(self) -> Dict[str, Any]:
        """Run the complete pipeline evaluation"""
        
        print("🚀 Starting End-to-End Pipeline Evaluation")
        print("=" * 60)
        
        start_time = time.time()
        
        # Setup
        self.setup_pipeline()
        
        try:
            # Run end-to-end pipeline evaluation
            pipeline_metrics, failure_cases = self.evaluate_end_to_end_pipeline()
            
            # Generate visualizations
            plot_paths, dashboard_path = self.generate_pipeline_visualizations(pipeline_metrics, failure_cases)
            
            # Generate summary report
            total_time = time.time() - start_time
            summary = self._generate_summary_report(pipeline_metrics, total_time, plot_paths, dashboard_path)
            
            print("\n" + "=" * 60)
            print("✅ Pipeline Evaluation Complete!")
            print(f"📁 Results saved to: {self.output_dir}")
            print(f"📊 Dashboard: {dashboard_path}")
            print(f"⏱️ Total time: {total_time:.2f} seconds")
            
            return pipeline_metrics
            
        except Exception as e:
            print(f"\n❌ Pipeline evaluation failed: {e}")
            import traceback
            traceback.print_exc()
            raise
        
        finally:
            # Cleanup
            if hasattr(self, 'performance_monitor'):
                self.performance_monitor.stop_monitoring()
            if hasattr(self, 'pipeline'):
                del self.pipeline
            gc.collect()
    
    def _generate_summary_report(self, pipeline_metrics: Dict, total_time: float, 
                                plot_paths: List[Path], dashboard_path: str) -> Dict[str, Any]:
        """Generate a comprehensive summary report for pipeline evaluation"""
        
        summary = {
            'evaluation_info': {
                'total_images_attempted': len(self.test_images),
                'total_images_processed': pipeline_metrics.get('total_processed', 0),
                'skipped_batches': pipeline_metrics.get('skipped_batches', 0),
                'skipped_images': pipeline_metrics.get('skipped_images', 0),
                'total_time': total_time,
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                'hardware_config': HARDWARE_CONFIG,
                'evaluation_type': 'end_to_end_pipeline'
            },
            'performance_summary': {
                'overall_accuracy': pipeline_metrics['overall_accuracy'],
                'detection_accuracy': pipeline_metrics['detection_accuracy'],
                'classification_accuracy': pipeline_metrics['classification_accuracy'],
                'images_per_second': pipeline_metrics['timing_metrics'].get('images_per_second', 0),
                'avg_processing_time': pipeline_metrics['timing_metrics'].get('mean_total_time', 0)
            },
            'skip_summary': {
                'skipped_batches': pipeline_metrics.get('skipped_batches', 0),
                'skipped_images': pipeline_metrics.get('skipped_images', 0),
                'completion_rate': (pipeline_metrics.get('total_processed', 0) / len(self.test_images)) * 100 if self.test_images else 0
            },
            'plot_files': [str(p) for p in plot_paths],
            'dashboard_file': dashboard_path
        }
        
        # Save summary
        summary_file = self.output_dir / "pipeline_evaluation_summary.json"
        safe_json_dump(summary, summary_file, indent=2)
        
        return summary


def main():
    """Main function to run the pipeline evaluation"""
    
    parser = argparse.ArgumentParser(description="Run end-to-end crack analysis pipeline evaluation")
    parser.add_argument("--test-dir", type=str, default=str(TEST_DATA_DIR),
                       help="Directory containing test data")
    parser.add_argument("--output", type=str, default=str(RESULTS_DIR / "pipeline_evaluation"),
                       help="Output directory for results")
    parser.add_argument("--device", type=str, choices=['cuda', 'cpu'], 
                       default=HARDWARE_CONFIG['device'],
                       help="Device to use for inference")
    
    args = parser.parse_args()
    
    # Update hardware config
    HARDWARE_CONFIG['device'] = args.device
    
    # Validate inputs
    test_data_dir = Path(args.test_dir)
    if not test_data_dir.exists():
        print(f"❌ Test data directory not found: {test_data_dir}")
        return 1
    
    output_dir = Path(args.output)
    
    # Run pipeline evaluation
    try:
        evaluator = PipelineEvaluator(test_data_dir, output_dir)
        metrics = evaluator.run_pipeline_evaluation()
        return 0
        
    except Exception as e:
        print(f"❌ Pipeline evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
