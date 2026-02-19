"""
Main benchmarking script for the crack analysis pipeline
Runs comprehensive evaluation of YOLO detection, segmentation, and classification models
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
    from Crack_Detection_YOLO.crack_yolo_train.crack_detection_inference import CrackDetector
    
    # Import benchmark modules
    from config_benchmark import (
        MODEL_PATHS, DETECTION_THRESHOLDS, METRICS_CONFIG, 
        HARDWARE_CONFIG, BENCHMARK_CONFIG, TEST_DATA_DIR, RESULTS_DIR
    )
    from evaluation_metrics import (
        YOLOEvaluationMetrics, SegmentationEvaluationMetrics,
        ClassificationEvaluationMetrics, PipelineEvaluationMetrics
    )
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
    def cleanup_memory():
        """Force memory cleanup"""
        gc.collect()
        if torch and torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    @staticmethod
    def check_memory_threshold(threshold=85):
        """Check if memory usage exceeds threshold and cleanup if needed"""
        memory_usage = MemoryManager.get_memory_usage()
        if memory_usage > threshold:
            print(f"⚠️ High memory usage ({memory_usage:.1f}%), performing cleanup...")
            MemoryManager.cleanup_memory()
            return True
        return False


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
        """Monitor performance metrics"""
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
                    # Use torch to get GPU memory usage
                    memory_used = torch.cuda.memory_allocated(0) if torch.cuda.device_count() > 0 else 0
                    memory_total = torch.cuda.max_memory_allocated(0) if torch.cuda.device_count() > 0 else 1
                    gpu_percent = (memory_used / memory_total) * 100 if memory_total > 0 else 0
                else:
                    try:
                        import GPUtil
                        gpus = GPUtil.getGPUs()
                        if gpus:
                            gpu_percent = gpus[0].load * 100
                    except ImportError:
                        pass
            except Exception:
                gpu_percent = 0
            
            self.stats['cpu_usage'].append(cpu_percent)
            self.stats['memory_usage'].append(memory_percent)
            self.stats['gpu_usage'].append(gpu_percent)
            self.stats['timestamps'].append(timestamp)
            
            time.sleep(0.5)  # Monitor every 0.5 seconds
    
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


class CrackAnalysisBenchmark:
    """Main benchmarking class for the crack analysis pipeline"""
    
    def __init__(self, test_data_dir: Path, output_dir: Path):
        self.test_data_dir = Path(test_data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.pipeline = None
        self.yolo_detector = None
        self.performance_monitor = PerformanceMonitor()
        
        # Load test data
        self.test_images = []
        self.ground_truth = None
        self._load_test_data()
        
        # Initialize metrics calculators
        self.yolo_metrics = YOLOEvaluationMetrics(
            iou_threshold=DETECTION_THRESHOLDS['confidence_threshold'],
            confidence_threshold=DETECTION_THRESHOLDS['iou_threshold']
        )
        self.seg_metrics = SegmentationEvaluationMetrics()
        self.class_metrics = ClassificationEvaluationMetrics(METRICS_CONFIG['classification_classes'])
        self.pipeline_metrics = PipelineEvaluationMetrics()
        
        # Initialize visualizer
        self.visualizer = BenchmarkVisualizer(self.output_dir / "plots")
        
        print(f"🚀 Benchmark initialized with {len(self.test_images)} test images")
    
    def _load_test_data(self):
        """Load test images and ground truth data"""
        
        # Load ground truth CSV
        ground_truth_path = self.test_data_dir / "metadata" / "ground_truth.csv"
        if ground_truth_path.exists():
            self.ground_truth = pd.read_csv(ground_truth_path)
            print(f"📊 Loaded ground truth for {len(self.ground_truth)} images")
        else:
            raise FileNotFoundError(f"Ground truth file not found: {ground_truth_path}")
        
        # Check available memory and limit test images if needed
        available_memory_gb = psutil.virtual_memory().available / (1024**3)
        if available_memory_gb < 4:  # Less than 4GB available
            max_images = min(50, len(self.ground_truth))  # Limit to 50 images
            print(f"⚠️ Low memory detected ({available_memory_gb:.1f}GB), limiting to {max_images} images")
            self.ground_truth = self.ground_truth.head(max_images)
        elif available_memory_gb < 8:  # Less than 8GB available
            max_images = min(100, len(self.ground_truth))  # Limit to 100 images
            print(f"⚠️ Limited memory detected ({available_memory_gb:.1f}GB), limiting to {max_images} images")
            self.ground_truth = self.ground_truth.head(max_images)
        
        # Load test images
        images_dir = self.test_data_dir / "images"
        for filename in self.ground_truth['filename']:
            image_path = images_dir / filename
            if image_path.exists():
                self.test_images.append(image_path)
        
        print(f"📂 Found {len(self.test_images)} test images")
    
    def _clean_metrics_for_json(self, metrics: Dict) -> Dict:
        """Clean metrics dictionary to avoid circular references in JSON serialization"""
        
        if not metrics:
            return {}
        
        cleaned = {}
        
        for key, value in metrics.items():
            if isinstance(value, np.ndarray):
                cleaned[key] = value.tolist()
            elif isinstance(value, (np.integer, np.floating)):
                cleaned[key] = float(value)
            elif isinstance(value, dict):
                # Recursively clean nested dictionaries
                cleaned[key] = self._clean_metrics_for_json(value)
            elif isinstance(value, list):
                # Clean lists
                cleaned_list = []
                for item in value:
                    if isinstance(item, np.ndarray):
                        cleaned_list.append(item.tolist())
                    elif isinstance(item, (np.integer, np.floating)):
                        cleaned_list.append(float(item))
                    elif isinstance(item, dict):
                        cleaned_list.append(self._clean_metrics_for_json(item))
                    else:
                        # Skip complex objects that might cause circular references
                        try:
                            json.dumps(item)  # Test if serializable
                            cleaned_list.append(item)
                        except:
                            continue
                cleaned[key] = cleaned_list
            else:
                # Only include serializable objects
                try:
                    json.dumps(value)  # Test if serializable
                    cleaned[key] = value
                except:
                    # Skip objects that can't be serialized (like matplotlib figures, etc.)
                    continue
        
        return cleaned
    
    def setup_pipeline(self):
        """Initialize the crack analysis pipeline with memory optimization"""
        
        print("🔧 Setting up crack analysis pipeline...")
        
        try:
            # Initialize main pipeline with memory optimization
            self.pipeline = CrackAnalysisPipeline(
                yolo_model_path=str(MODEL_PATHS['yolo']),
                segmentation_model_path=str(MODEL_PATHS['segmentation']),
                classification_model_path=str(MODEL_PATHS['classification']),
                rag_data_dir=str(MODEL_PATHS['rag_data']) if MODEL_PATHS['rag_data'].exists() else None,
                device=HARDWARE_CONFIG['device']
            )
            
            # Also initialize standalone YOLO detector for detailed detection analysis
            self.yolo_detector = CrackDetector(
                model_path=str(MODEL_PATHS['yolo']),
                conf_threshold=DETECTION_THRESHOLDS['confidence_threshold'],
                iou_threshold=DETECTION_THRESHOLDS['iou_threshold']
            )
            
            # Force initial cleanup
            gc.collect()
            if torch and torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            print("✅ Pipeline setup complete")
            
        except Exception as e:
            print(f"❌ Error setting up pipeline: {e}")
            raise
    
    def benchmark_yolo_detection(self) -> Dict[str, Any]:
        """Benchmark YOLO crack detection performance"""
        
        print("\n🎯 Benchmarking YOLO Detection...")
        
        predictions = []
        ground_truths = []
        timing_data = []
        
        for idx, image_path in enumerate(tqdm(self.test_images, desc="YOLO Detection")):
            try:
                # Memory management
                if idx % 20 == 0 and idx > 0:
                    MemoryManager.check_memory_threshold(threshold=80)
                
                # Get ground truth for this image
                gt_row = self.ground_truth.iloc[idx]
                
                # Prepare ground truth format
                gt_detections = []
                if gt_row['has_crack'] and not pd.isna(gt_row['bbox_x1']):
                    gt_detections.append({
                        'bbox': [
                            int(gt_row['bbox_x1']), int(gt_row['bbox_y1']),
                            int(gt_row['bbox_x2']), int(gt_row['bbox_y2'])
                        ]
                    })
                
                ground_truths.append({'detections': gt_detections})
                
                # Run detection with timing
                start_time = time.time()
                result = self.yolo_detector.detect_cracks(str(image_path), save_results=False)
                detection_time = time.time() - start_time
                
                # Convert result to evaluation format
                pred_detections = []
                for detection in result.get('detections', []):
                    pred_detections.append({
                        'bbox': detection['bbox'],
                        'confidence': detection['confidence'],
                        'class': detection['class']
                    })
                
                predictions.append({'detections': pred_detections})
                timing_data.append({'detection_time': detection_time})
                
            except Exception as e:
                print(f"⚠️ Error processing {image_path}: {e}")
                # Add empty results to maintain alignment
                predictions.append({'detections': []})
                timing_data.append({'detection_time': 0})
        
        # Calculate metrics
        detection_metrics = self.yolo_metrics.calculate_precision_recall(predictions, ground_truths)
        map_metrics = self.yolo_metrics.calculate_map(predictions, ground_truths, 
                                                     METRICS_CONFIG['iou_thresholds'])
        detection_metrics.update(map_metrics)
        
        # Add timing information
        avg_detection_time = np.mean([t['detection_time'] for t in timing_data])
        detection_metrics['avg_detection_time'] = avg_detection_time
        detection_metrics['detection_fps'] = 1.0 / avg_detection_time if avg_detection_time > 0 else 0
        
        # Save detailed results
        results_file = self.output_dir / "yolo_detection_results.json"
        with open(results_file, 'w') as f:
            json.dump(detection_metrics, f, indent=2)
        
        print(f"📊 Detection Metrics:")
        print(f"   Precision: {detection_metrics['precision']:.3f}")
        print(f"   Recall: {detection_metrics['recall']:.3f}")
        print(f"   F1-Score: {detection_metrics['f1_score']:.3f}")
        print(f"   mAP@0.5: {detection_metrics.get('mAP@0.5', 0):.3f}")
        print(f"   Avg Detection Time: {avg_detection_time:.3f}s")
        
        return detection_metrics
    
    def benchmark_segmentation(self) -> Dict[str, Any]:
        """Benchmark segmentation performance on images with cracks"""
        
        print("\n🎨 Benchmarking Segmentation...")
        
        predicted_masks = []
        ground_truth_masks = []
        
        # Filter to only images with cracks
        crack_images = []
        for idx, image_path in enumerate(self.test_images):
            gt_row = self.ground_truth.iloc[idx]
            if gt_row['has_crack']:
                crack_images.append((image_path, idx))
        
        print(f"📍 Testing segmentation on {len(crack_images)} images with cracks")
        
        # Process in smaller batches to prevent memory overflow
        batch_size = 5  # Process 5 images at a time
        
        for batch_start in range(0, len(crack_images), batch_size):
            batch_end = min(batch_start + batch_size, len(crack_images))
            batch = crack_images[batch_start:batch_end]
            
            print(f"Processing segmentation batch {batch_start//batch_size + 1}/{(len(crack_images)-1)//batch_size + 1}")
            
            for image_path, idx in tqdm(batch, desc=f"Segmentation Batch {batch_start//batch_size + 1}"):
                try:
                    # Run segmentation
                    img = cv2.imread(str(image_path))
                    if img is None:
                        continue
                    
                    # Use pipeline segmentation
                    if hasattr(self.pipeline, 'segment_cracks'):
                        # Assuming the pipeline has a segmentation method
                        mask = self.pipeline.segment_cracks(img)
                        if mask is not None:
                            predicted_masks.append(mask)
                        else:
                            # Create empty mask if segmentation fails
                            predicted_masks.append(np.zeros(img.shape[:2], dtype=np.uint8))
                    else:
                        # Create dummy mask for now - in real implementation, 
                        # you'd call the actual segmentation model
                        predicted_masks.append(np.zeros(img.shape[:2], dtype=np.uint8))
                    
                    # Create ground truth mask from bounding box
                    gt_row = self.ground_truth.iloc[idx]
                    gt_mask = np.zeros(img.shape[:2], dtype=np.uint8)
                    if not pd.isna(gt_row['bbox_x1']):
                        x1, y1, x2, y2 = (
                            int(gt_row['bbox_x1']), int(gt_row['bbox_y1']),
                            int(gt_row['bbox_x2']), int(gt_row['bbox_y2'])
                        )
                        gt_mask[y1:y2, x1:x2] = 1
                    
                    ground_truth_masks.append(gt_mask)
                    
                except Exception as e:
                    print(f"⚠️ Error processing segmentation for {image_path}: {e}")
            
            # Cleanup after each batch
            MemoryManager.cleanup_memory()
            MemoryManager.check_memory_threshold(threshold=75)
        
        # Calculate segmentation metrics
        seg_metrics = {}
        if predicted_masks and ground_truth_masks:
            seg_metrics = self.seg_metrics.evaluate_segmentation(predicted_masks, ground_truth_masks)
        
        # Clean metrics for JSON serialization
        cleaned_metrics = self._clean_metrics_for_json(seg_metrics)
        
        # Save results
        results_file = self.output_dir / "segmentation_results.json"
        with open(results_file, 'w') as f:
            json.dump(cleaned_metrics, f, indent=2)
        
        if seg_metrics:
            print(f"📊 Segmentation Metrics:")
            print(f"   Mean IoU: {seg_metrics['mean_iou']:.3f}")
            print(f"   Mean Dice: {seg_metrics['mean_dice']:.3f}")
            print(f"   Mean Pixel Accuracy: {seg_metrics['mean_pixel_accuracy']:.3f}")
        
        return seg_metrics
    
    def benchmark_classification(self) -> Dict[str, Any]:
        """Benchmark crack classification performance using standalone classifier"""
        
        print("\n🏷️ Benchmarking Classification...")
        
        y_true = []
        y_pred = []
        y_scores = []
        
        # Initialize standalone classification model
        try:
            from pipeline_orchestrator import ClassificationModel
            classifier = ClassificationModel(str(MODEL_PATHS['classification']))
            print(f"✅ Loaded classification model: {MODEL_PATHS['classification']}")
        except Exception as e:
            print(f"❌ Failed to load classification model: {e}")
            return {}
        
        # Filter to only images with cracks for classification
        crack_images = []
        for idx, image_path in enumerate(self.test_images):
            gt_row = self.ground_truth.iloc[idx]
            # Only include images that have cracks AND a valid crack type (not 'none' or NaN)
            if (gt_row['has_crack'] and 
                gt_row['crack_type'] != 'none' and 
                pd.notna(gt_row['crack_type']) and 
                gt_row['crack_type'].strip() != ''):
                crack_images.append((image_path, idx))
        
        print(f"📍 Testing classification on {len(crack_images)} images with labeled cracks")
        
        if len(crack_images) == 0:
            print("⚠️ No images with labeled cracks found for classification evaluation")
            return {}
        
        for image_path, idx in tqdm(crack_images, desc="Classification"):
            try:
                # Memory management - more frequent for heavy models
                if len(y_true) % 5 == 0 and len(y_true) > 0:
                    MemoryManager.check_memory_threshold(threshold=75)
                
                # Get ground truth
                gt_row = self.ground_truth.iloc[idx]
                gt_crack_type = gt_row['crack_type'].strip()
                
                # Validate ground truth crack type
                if gt_crack_type not in classifier.categories:
                    print(f"⚠️ Unknown crack type '{gt_crack_type}' in ground truth, skipping...")
                    continue
                
                y_true.append(gt_crack_type)
                
                # Load image and run YOLO detection first to get crack region
                img = cv2.imread(str(image_path))
                if img is None:
                    print(f"⚠️ Could not load image {image_path}, skipping...")
                    y_pred.append('vertical')  # Default fallback
                    y_scores.append([0.25, 0.25, 0.25, 0.25])
                    continue
                
                # Run YOLO detection to get crack region
                yolo_result = self.yolo_detector.detect_cracks(str(image_path), save_results=False)
                
                if yolo_result.get('detections'):
                    # Get the first detected crack region
                    detection = yolo_result['detections'][0]
                    bbox = detection['bbox']
                    x1, y1, x2, y2 = bbox
                    
                    # Validate bounding box
                    if x2 > x1 and y2 > y1 and x1 >= 0 and y1 >= 0 and x2 <= img.shape[1] and y2 <= img.shape[0]:
                        # Extract crack region
                        crack_region = img[y1:y2, x1:x2]
                        
                        # Create a simple mask (binary mask for the region)
                        # In a real scenario, you'd use the segmentation model here
                        if crack_region.size > 0:
                            mask = np.ones((crack_region.shape[0], crack_region.shape[1]), dtype=np.uint8) * 255
                            
                            # Run classification on the mask
                            predicted_type, confidence = classifier.classify(mask)
                            
                            # Convert confidence to class scores
                            confidence_scores = [0.01] * len(classifier.categories)  # Small non-zero values
                            if predicted_type in classifier.categories:
                                class_idx = classifier.categories.index(predicted_type)
                                confidence_scores[class_idx] = max(confidence, 0.25)  # Ensure minimum confidence
                            else:
                                predicted_type = classifier.categories[0]  # Default to first category
                        else:
                            # Empty region, use default
                            predicted_type = classifier.categories[0]
                            confidence_scores = [0.25] * len(classifier.categories)
                    else:
                        # Invalid bounding box, use default
                        predicted_type = classifier.categories[0]
                        confidence_scores = [0.25] * len(classifier.categories)
                    
                else:
                    # No crack detected, classify as first category (fallback)
                    predicted_type = classifier.categories[0]
                    confidence_scores = [0.25] * len(classifier.categories)
                
                y_pred.append(predicted_type)
                y_scores.append(confidence_scores)
                
            except Exception as e:
                print(f"⚠️ Error processing classification for {image_path}: {e}")
                # Add default values to maintain list alignment
                if len(y_true) > len(y_pred):
                    y_pred.append(classifier.categories[0])
                    y_scores.append([0.25] * len(classifier.categories))
        
        # Calculate classification metrics
        class_metrics = {}
        if y_true and y_pred:
            class_metrics = self.class_metrics.calculate_metrics(y_true, y_pred, y_scores)
        
        # Clean metrics for JSON serialization to avoid circular references
        cleaned_metrics = self._clean_metrics_for_json(class_metrics)
        
        # Save results
        results_file = self.output_dir / "classification_results.json"
        with open(results_file, 'w') as f:
            json.dump(cleaned_metrics, f, indent=2)
        
        if class_metrics:
            print(f"📊 Classification Metrics:")
            print(f"   Accuracy: {class_metrics['accuracy']:.3f}")
            print(f"   Macro F1: {class_metrics['f1_macro']:.3f}")
            print(f"   Weighted F1: {class_metrics['f1_weighted']:.3f}")
        
        # Cleanup
        del classifier
        MemoryManager.cleanup_memory()
        
        return class_metrics
    
    def generate_visualizations(self, all_metrics: Dict[str, Any], failure_cases: List[Dict]):
        """Generate all benchmark visualizations"""
        
        print("\n📊 Generating Visualizations...")
        
        plot_paths = []
        
        # Detection metrics plot
        if 'detection' in all_metrics:
            path = self.visualizer.plot_detection_metrics(all_metrics['detection'])
            plot_paths.append(path)
            print(f"   ✅ Detection metrics: {path.name}")
        
        # Segmentation metrics plot
        if 'segmentation' in all_metrics:
            path = self.visualizer.plot_segmentation_metrics(all_metrics['segmentation'])
            plot_paths.append(path)
            print(f"   ✅ Segmentation metrics: {path.name}")
        
        # Classification metrics plot
        if 'classification' in all_metrics:
            path = self.visualizer.plot_classification_metrics(all_metrics['classification'])
            plot_paths.append(path)
            print(f"   ✅ Classification metrics: {path.name}")
        
        # Create summary comparison plot for individual models
        if len(all_metrics) > 1:
            try:
                path = self.visualizer.plot_model_comparison(all_metrics)
                plot_paths.append(path)
                print(f"   ✅ Model comparison: {path.name}")
            except:
                print("   ⚠️ Could not create model comparison plot")
        
        # Interactive dashboard
        dashboard_path = self.visualizer.create_interactive_dashboard(all_metrics)
        print(f"   ✅ Interactive dashboard: {Path(dashboard_path).name}")
        
        return plot_paths, dashboard_path
    
    def run_full_benchmark(self) -> Dict[str, Any]:
        """Run the individual model benchmarks (YOLO, Segmentation, Classification)"""
        
        print("🚀 Starting Individual Model Benchmarks")
        print("=" * 60)
        
        start_time = time.time()
        
        # Setup
        self.setup_pipeline()
        
        all_metrics = {}
        
        try:
            # 1. YOLO Detection Benchmark
            all_metrics['detection'] = self.benchmark_yolo_detection()
            
            # 2. Segmentation Benchmark  
            all_metrics['segmentation'] = self.benchmark_segmentation()
            
            # 3. Classification Benchmark
            all_metrics['classification'] = self.benchmark_classification()
            
            # 4. Generate Visualizations
            plot_paths, dashboard_path = self.generate_visualizations(all_metrics, [])
            
            # 5. Generate Summary Report
            total_time = time.time() - start_time
            summary = self._generate_summary_report(all_metrics, total_time, plot_paths, dashboard_path)
            
            print("\n" + "=" * 60)
            print("✅ Individual Model Benchmarks Complete!")
            print(f"📁 Results saved to: {self.output_dir}")
            print(f"📊 Dashboard: {dashboard_path}")
            print(f"⏱️ Total time: {total_time:.2f} seconds")
            
            return all_metrics
            
        except Exception as e:
            print(f"\n❌ Benchmark failed: {e}")
            import traceback
            traceback.print_exc()
            raise
        
        finally:
            # Cleanup
            if hasattr(self, 'performance_monitor'):
                self.performance_monitor.stop_monitoring()
            gc.collect()
    
    def _generate_summary_report(self, all_metrics: Dict, total_time: float, 
                                plot_paths: List[Path], dashboard_path: str) -> Dict[str, Any]:
        """Generate a comprehensive summary report"""
        
        summary = {
            'benchmark_info': {
                'total_images': len(self.test_images),
                'total_time': total_time,
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                'hardware_config': HARDWARE_CONFIG
            },
            'performance_summary': {},
            'plot_files': [str(p) for p in plot_paths],
            'dashboard_file': dashboard_path
        }
        
        # Extract key metrics
        if 'detection' in all_metrics:
            summary['performance_summary']['detection'] = {
                'precision': all_metrics['detection']['precision'],
                'recall': all_metrics['detection']['recall'],
                'f1_score': all_metrics['detection']['f1_score'],
                'mAP@0.5': all_metrics['detection'].get('mAP@0.5', 0)
            }
        
        if 'classification' in all_metrics:
            summary['performance_summary']['classification'] = {
                'accuracy': all_metrics['classification']['accuracy'],
                'macro_f1': all_metrics['classification']['f1_macro']
            }
        
        if 'segmentation' in all_metrics:
            summary['performance_summary']['segmentation'] = {
                'mean_iou': all_metrics['segmentation'].get('mean_iou', 0),
                'mean_dice': all_metrics['segmentation'].get('mean_dice', 0)
            }
        
        # Save summary
        summary_file = self.output_dir / "benchmark_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        return summary


def main():
    """Main function to run the benchmark"""
    
    parser = argparse.ArgumentParser(description="Run crack analysis pipeline benchmark")
    parser.add_argument("--test-dir", type=str, default=str(TEST_DATA_DIR),
                       help="Directory containing test data")
    parser.add_argument("--output", type=str, default=str(RESULTS_DIR),
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
    
    # Run benchmark
    try:
        benchmark = CrackAnalysisBenchmark(test_data_dir, output_dir)
        metrics = benchmark.run_full_benchmark()
        return 0
        
    except Exception as e:
        print(f"❌ Benchmark failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
