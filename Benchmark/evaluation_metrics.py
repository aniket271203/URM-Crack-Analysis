"""
Evaluation metrics calculation for crack analysis benchmarking
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Any
from sklearn.metrics import (
    precision_recall_fscore_support, accuracy_score, confusion_matrix,
    roc_curve, auc, average_precision_score, precision_recall_curve
)
from sklearn.preprocessing import label_binarize
import cv2
from pathlib import Path


class YOLOEvaluationMetrics:
    """Evaluation metrics for YOLO crack detection"""
    
    def __init__(self, iou_threshold: float = 0.5, confidence_threshold: float = 0.25):
        self.iou_threshold = iou_threshold
        self.confidence_threshold = confidence_threshold
    
    def calculate_iou(self, box1: List[float], box2: List[float]) -> float:
        """Calculate Intersection over Union (IoU) between two bounding boxes"""
        x1_min, y1_min, x1_max, y1_max = box1
        x2_min, y2_min, x2_max, y2_max = box2
        
        # Calculate intersection area
        inter_x_min = max(x1_min, x2_min)
        inter_y_min = max(y1_min, y2_min)
        inter_x_max = min(x1_max, x2_max)
        inter_y_max = min(y1_max, y2_max)
        
        if inter_x_max <= inter_x_min or inter_y_max <= inter_y_min:
            return 0.0
        
        inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)
        
        # Calculate union area
        box1_area = (x1_max - x1_min) * (y1_max - y1_min)
        box2_area = (x2_max - x2_min) * (y2_max - y2_min)
        union_area = box1_area + box2_area - inter_area
        
        return inter_area / union_area if union_area > 0 else 0.0
    
    def match_detections(self, predicted_boxes: List[Dict], 
                        ground_truth_boxes: List[Dict]) -> Tuple[List, List, List]:
        """Match predicted boxes with ground truth boxes"""
        true_positives = []
        false_positives = []
        false_negatives = []
        
        # Convert ground truth to list for matching
        gt_matched = [False] * len(ground_truth_boxes)
        
        # Sort predictions by confidence (highest first)
        pred_sorted = sorted(predicted_boxes, key=lambda x: x.get('confidence', 0), reverse=True)
        
        for pred in pred_sorted:
            if pred.get('confidence', 0) < self.confidence_threshold:
                continue
                
            best_iou = 0
            best_gt_idx = -1
            
            # Find best matching ground truth box
            for gt_idx, gt_box in enumerate(ground_truth_boxes):
                if gt_matched[gt_idx]:
                    continue
                
                iou = self.calculate_iou(pred['bbox'], gt_box['bbox'])
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = gt_idx
            
            # Check if match is good enough
            if best_iou >= self.iou_threshold and best_gt_idx != -1:
                true_positives.append({
                    'confidence': pred['confidence'],
                    'iou': best_iou,
                    'pred_box': pred['bbox'],
                    'gt_box': ground_truth_boxes[best_gt_idx]['bbox']
                })
                gt_matched[best_gt_idx] = True
            else:
                false_positives.append({
                    'confidence': pred['confidence'],
                    'pred_box': pred['bbox']
                })
        
        # Add unmatched ground truth boxes as false negatives
        for gt_idx, matched in enumerate(gt_matched):
            if not matched:
                false_negatives.append({
                    'gt_box': ground_truth_boxes[gt_idx]['bbox']
                })
        
        return true_positives, false_positives, false_negatives
    
    def calculate_precision_recall(self, predictions: List[Dict], 
                                 ground_truths: List[Dict]) -> Dict[str, float]:
        """Calculate precision, recall, F1-score, and mAP"""
        all_tp, all_fp, all_fn = [], [], []
        
        for pred, gt in zip(predictions, ground_truths):
            tp, fp, fn = self.match_detections(
                pred.get('detections', []), 
                gt.get('detections', [])
            )
            all_tp.extend(tp)
            all_fp.extend(fp)
            all_fn.extend(fn)
        
        # Calculate metrics
        tp_count = len(all_tp)
        fp_count = len(all_fp)
        fn_count = len(all_fn)
        
        precision = tp_count / (tp_count + fp_count) if (tp_count + fp_count) > 0 else 0
        recall = tp_count / (tp_count + fn_count) if (tp_count + fn_count) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        return {
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'tp_count': tp_count,
            'fp_count': fp_count,
            'fn_count': fn_count,
            'total_gt': tp_count + fn_count,
            'total_pred': tp_count + fp_count
        }
    
    def calculate_map(self, predictions: List[Dict], ground_truths: List[Dict],
                     iou_thresholds: List[float] = None) -> Dict[str, float]:
        """Calculate mean Average Precision (mAP) at different IoU thresholds"""
        if iou_thresholds is None:
            iou_thresholds = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
        
        aps = []
        for iou_thresh in iou_thresholds:
            self.iou_threshold = iou_thresh
            metrics = self.calculate_precision_recall(predictions, ground_truths)
            # Simplified AP calculation (can be enhanced with full PR curve)
            ap = metrics['precision'] * metrics['recall'] if metrics['recall'] > 0 else 0
            aps.append(ap)
        
        return {
            'mAP@0.5': aps[0] if len(aps) > 0 else 0,
            'mAP@0.5:0.95': np.mean(aps),
            'mAP_per_iou': dict(zip(iou_thresholds, aps))
        }


class SegmentationEvaluationMetrics:
    """Evaluation metrics for segmentation"""
    
    def calculate_iou(self, pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
        """Calculate IoU for segmentation masks"""
        intersection = np.logical_and(pred_mask, gt_mask)
        union = np.logical_or(pred_mask, gt_mask)
        
        if np.sum(union) == 0:
            return 1.0 if np.sum(intersection) == 0 else 0.0
        
        return np.sum(intersection) / np.sum(union)
    
    def calculate_dice_coefficient(self, pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
        """Calculate Dice coefficient"""
        intersection = np.logical_and(pred_mask, gt_mask)
        
        if np.sum(pred_mask) + np.sum(gt_mask) == 0:
            return 1.0
        
        return 2.0 * np.sum(intersection) / (np.sum(pred_mask) + np.sum(gt_mask))
    
    def calculate_pixel_accuracy(self, pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
        """Calculate pixel-wise accuracy"""
        correct_pixels = np.sum(pred_mask == gt_mask)
        total_pixels = pred_mask.size
        return correct_pixels / total_pixels
    
    def evaluate_segmentation(self, predicted_masks: List[np.ndarray],
                            ground_truth_masks: List[np.ndarray]) -> Dict[str, float]:
        """Evaluate segmentation performance"""
        ious = []
        dice_scores = []
        pixel_accuracies = []
        
        for pred_mask, gt_mask in zip(predicted_masks, ground_truth_masks):
            # Ensure masks are binary
            pred_binary = (pred_mask > 0.5).astype(np.uint8)
            gt_binary = (gt_mask > 0.5).astype(np.uint8)
            
            ious.append(self.calculate_iou(pred_binary, gt_binary))
            dice_scores.append(self.calculate_dice_coefficient(pred_binary, gt_binary))
            pixel_accuracies.append(self.calculate_pixel_accuracy(pred_binary, gt_binary))
        
        return {
            'mean_iou': np.mean(ious),
            'std_iou': np.std(ious),
            'mean_dice': np.mean(dice_scores),
            'std_dice': np.std(dice_scores),
            'mean_pixel_accuracy': np.mean(pixel_accuracies),
            'std_pixel_accuracy': np.std(pixel_accuracies),
            'per_image_metrics': {
                'ious': ious,
                'dice_scores': dice_scores,
                'pixel_accuracies': pixel_accuracies
            }
        }


class ClassificationEvaluationMetrics:
    """Evaluation metrics for crack classification"""
    
    def __init__(self, class_names: List[str]):
        self.class_names = class_names
    
    def calculate_metrics(self, y_true: List[str], y_pred: List[str], 
                         y_scores: Optional[List[List[float]]] = None) -> Dict[str, Any]:
        """Calculate comprehensive classification metrics"""
        
        # Convert to numerical labels
        label_to_idx = {label: idx for idx, label in enumerate(self.class_names)}
        y_true_idx = [label_to_idx.get(label, 0) for label in y_true]
        y_pred_idx = [label_to_idx.get(label, 0) for label in y_pred]
        
        # Basic metrics
        accuracy = accuracy_score(y_true_idx, y_pred_idx)
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true_idx, y_pred_idx, average=None, zero_division=0
        )
        
        # Macro and weighted averages
        precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
            y_true_idx, y_pred_idx, average='macro', zero_division=0
        )
        precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
            y_true_idx, y_pred_idx, average='weighted', zero_division=0
        )
        
        # Confusion matrix
        cm = confusion_matrix(y_true_idx, y_pred_idx)
        
        # Per-class metrics
        per_class_metrics = {}
        for i, class_name in enumerate(self.class_names):
            per_class_metrics[class_name] = {
                'precision': precision[i] if i < len(precision) else 0,
                'recall': recall[i] if i < len(recall) else 0,
                'f1_score': f1[i] if i < len(f1) else 0,
                'support': support[i] if i < len(support) else 0
            }
        
        results = {
            'accuracy': accuracy,
            'precision_macro': precision_macro,
            'recall_macro': recall_macro,
            'f1_macro': f1_macro,
            'precision_weighted': precision_weighted,
            'recall_weighted': recall_weighted,
            'f1_weighted': f1_weighted,
            'confusion_matrix': cm,
            'per_class_metrics': per_class_metrics,
            'class_names': self.class_names
        }
        
        # Add ROC/AUC metrics if scores are provided
        if y_scores is not None:
            results.update(self._calculate_roc_metrics(y_true_idx, y_scores))
        
        return results
    
    def _calculate_roc_metrics(self, y_true: List[int], y_scores: List[List[float]]) -> Dict[str, Any]:
        """Calculate ROC curves and AUC scores"""
        n_classes = len(self.class_names)
        
        # Binarize the output
        y_true_binarized = label_binarize(y_true, classes=list(range(n_classes)))
        
        if n_classes == 2:
            y_true_binarized = np.column_stack([1 - y_true_binarized, y_true_binarized])
        
        # Calculate ROC curve and AUC for each class
        fpr = {}
        tpr = {}
        roc_auc = {}
        
        for i in range(n_classes):
            if i < len(y_scores[0]):
                scores_class = [scores[i] for scores in y_scores]
                fpr[i], tpr[i], _ = roc_curve(y_true_binarized[:, i], scores_class)
                roc_auc[i] = auc(fpr[i], tpr[i])
            else:
                fpr[i], tpr[i], roc_auc[i] = [], [], 0.0
        
        return {
            'fpr': fpr,
            'tpr': tpr,
            'roc_auc': roc_auc,
            'mean_roc_auc': np.mean(list(roc_auc.values()))
        }


class PipelineEvaluationMetrics:
    """End-to-end pipeline evaluation metrics"""
    
    def __init__(self):
        self.yolo_metrics = YOLOEvaluationMetrics()
        self.seg_metrics = SegmentationEvaluationMetrics()
        self.class_metrics = ClassificationEvaluationMetrics(
            ['vertical', 'horizontal', 'diagonal', 'step']
        )
    
    def calculate_end_to_end_metrics(self, predictions: List[Dict], 
                                   ground_truths: List[Dict]) -> Dict[str, Any]:
        """Calculate comprehensive end-to-end pipeline metrics"""
        
        # Overall accuracy (correct end-to-end prediction)
        total_correct = 0
        total_images = len(ground_truths)
        
        detection_errors = 0
        classification_errors = 0
        
        for pred, gt in zip(predictions, ground_truths):
            gt_has_crack = gt.get('has_crack', False)
            pred_has_crack = len(pred.get('detections', [])) > 0
            
            # Detection accuracy
            if gt_has_crack == pred_has_crack:
                if not gt_has_crack:
                    # Correctly identified no crack
                    total_correct += 1
                else:
                    # Has crack - check classification
                    gt_type = gt.get('crack_type', '')
                    pred_type = pred.get('classification', {}).get('crack_type', '')
                    
                    if gt_type == pred_type:
                        total_correct += 1
                    else:
                        classification_errors += 1
            else:
                detection_errors += 1
        
        overall_accuracy = total_correct / total_images if total_images > 0 else 0
        
        return {
            'overall_accuracy': overall_accuracy,
            'detection_accuracy': (total_images - detection_errors) / total_images,
            'classification_accuracy': (total_images - classification_errors) / total_images,
            'total_images': total_images,
            'correct_predictions': total_correct,
            'detection_errors': detection_errors,
            'classification_errors': classification_errors
        }
    
    def calculate_processing_metrics(self, timing_data: List[Dict]) -> Dict[str, float]:
        """Calculate processing time metrics"""
        if not timing_data:
            return {}
        
        detection_times = [t.get('detection_time', 0) for t in timing_data]
        segmentation_times = [t.get('segmentation_time', 0) for t in timing_data]
        classification_times = [t.get('classification_time', 0) for t in timing_data]
        total_times = [t.get('total_time', 0) for t in timing_data]
        
        return {
            'mean_detection_time': np.mean(detection_times),
            'mean_segmentation_time': np.mean(segmentation_times),
            'mean_classification_time': np.mean(classification_times),
            'mean_total_time': np.mean(total_times),
            'std_total_time': np.std(total_times),
            'min_total_time': np.min(total_times),
            'max_total_time': np.max(total_times),
            'images_per_second': 1.0 / np.mean(total_times) if np.mean(total_times) > 0 else 0
        }
