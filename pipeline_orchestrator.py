"""
Pipeline Orchestrator for Complete Crack Analysis System
Integrates YOLO Detection, Segmentation, Classification, and RAG Analysis
"""

import os
import sys
import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
from pathlib import Path
import tempfile
import gc
from typing import Dict, Any, Optional, Tuple

# Add paths for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'Crack_Detection_YOLO', 'crack_yolo_train'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'Rag_and_Reasoning', 'crack_analysis_rag', 'src'))

from ultralytics import YOLO
import tensorflow as tf

# Import RAG components
try:
    from main_rag import StructuralCrackRAG
    from crack_types import CrackType
except ImportError:
    print("Warning: RAG components not found. RAG analysis will be disabled.")
    StructuralCrackRAG = None
    CrackType = None


class SegmentationModel:
    """Wrapper for the DeepCrack segmentation model"""
    
    def __init__(self, model_path: str, device: str = 'cuda'):
        """
        Initialize segmentation model
        
        Args:
            model_path: Path to pretrained_net_G.pth
            device: 'cuda' or 'cpu'
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model_path = model_path
        
        # Import model architecture
        sys.path.append(os.path.join(os.path.dirname(__file__), 'Masking_and_Classification_model'))
        from model_utils import define_deepcrack
        
        # Build model
        gpu_ids = [0] if self.device.type == 'cuda' and torch.cuda.is_available() else []
        self.model = define_deepcrack(
            in_nc=3,
            num_classes=1,
            ngf=64,
            norm='batch',
            init_type='xavier',
            init_gain=0.02,
            gpu_ids=gpu_ids
        )
        
        # Load weights
        state_dict = torch.load(model_path, map_location=self.device, weights_only=True)
        
        # Handle DataParallel wrapper if present
        if isinstance(self.model, torch.nn.DataParallel):
            self.model.load_state_dict(state_dict)
        else:
            # Remove 'module.' prefix if weights were saved with DataParallel
            new_state_dict = {}
            for k, v in state_dict.items():
                name = k[7:] if k.startswith('module.') else k
                new_state_dict[name] = v
            self.model.load_state_dict(new_state_dict)
        self.model.eval()
        
        # Use half precision if GPU supports it
        if self.device.type == 'cuda' and torch.cuda.get_device_capability()[0] >= 7:
            self.model = self.model.half()
            self.use_fp16 = True
        else:
            self.use_fp16 = False
        
        # Image transformation
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        
        print(f"Segmentation model loaded on {self.device}")
    
    def segment(self, image_path: str) -> np.ndarray:
        """
        Segment an image to extract crack mask
        
        Args:
            image_path: Path to input image
            
        Returns:
            Segmented mask as numpy array (uint8, 0-255)
        """
        try:
            img = Image.open(image_path).convert("RGB")
            img_tensor = self.transform(img).unsqueeze(0).to(self.device)
            
            if self.use_fp16:
                img_tensor = img_tensor.half()
            
            with torch.no_grad():
                outputs = self.model(img_tensor)
                # Handle both single output and tuple/list of outputs
                if isinstance(outputs, (tuple, list)):
                    pred = outputs[-1]  # Get fused output
                else:
                    pred = outputs
            
            # Handle different output shapes
            if len(pred.shape) == 4:  # [batch, channels, H, W]
                pred = pred.squeeze(0)  # Remove batch dimension
            if len(pred.shape) == 3 and pred.shape[0] == 1:  # [1, H, W]
                pred = pred.squeeze(0)  # Remove channel dimension
            
            pred = torch.sigmoid(pred)
            pred = pred.cpu().numpy()
            pred = (pred * 255).astype("uint8")
            
            # Check if mask is too faint (low confidence) and boost if needed
            if pred.max() < 50:
                # Normalize to full range 0-255
                pred = cv2.normalize(pred, None, 0, 255, cv2.NORM_MINMAX)
                # Apply gamma correction to further boost faint details
                pred = np.power(pred / 255.0, 0.5) * 255.0
                pred = pred.astype("uint8")
            
            # Clean up
            del img_tensor
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            return pred
        except Exception as e:
            print(f"Error in segmentation: {e}")
            raise


class ClassificationModel:
    """Wrapper for the crack orientation classifier"""
    
    def __init__(self, model_path: str):
        """
        Initialize classification model
        
        Args:
            model_path: Path to crack_orientation_classifier.h5
        """
        self.model_path = model_path
        # Custom InputLayer to handle 'batch_shape' argument which might cause errors in some TF versions
        class PatchedInputLayer(tf.keras.layers.InputLayer):
            def __init__(self, batch_shape=None, **kwargs):
                if batch_shape is not None:
                    # Convert batch_shape to batch_input_shape for newer TF versions
                    kwargs['batch_input_shape'] = batch_shape
                super().__init__(**kwargs)
        
        # Custom DTypePolicy to handle 'DTypePolicy' serialization from newer Keras versions
        class PatchedDTypePolicy:
            def __init__(self, name=None, **kwargs):
                self._name = name or "float32"
                self._compute_dtype = self._name
                self._variable_dtype = self._name
            
            @property
            def name(self):
                return self._name
            
            @property
            def compute_dtype(self):
                return self._compute_dtype
            
            @property
            def variable_dtype(self):
                return self._variable_dtype
                
            def get_config(self):
                return {"name": self._name}
            
            @classmethod
            def from_config(cls, config):
                return cls(**config)

        # Dictionary of custom objects to handle compatibility issues
        custom_objects = {
            'InputLayer': PatchedInputLayer,
            'DTypePolicy': PatchedDTypePolicy
        }

        try:
            # Try loading with compile=False first to avoid optimizer issues
            self.model = tf.keras.models.load_model(model_path, compile=False)
        except (TypeError, ValueError) as e:
            print(f"Caught model loading error: {e}. Retrying with patched custom objects...")
            try:
                self.model = tf.keras.models.load_model(model_path, custom_objects=custom_objects, compile=False)
            except Exception as e2:
                print(f"Retry failed: {e2}")
                # Try one more time with just InputLayer if DTypePolicy wasn't the issue, 
                # or if DTypePolicy needs to be ignored differently.
                # But usually passing extra custom objects is fine.
                raise e2
        except Exception as e:
            print(f"Caught unexpected error loading model: {e}. Retrying with patched custom objects...")
            self.model = tf.keras.models.load_model(model_path, custom_objects=custom_objects, compile=False)

        self.categories = ['vertical', 'horizontal', 'diagonal', 'step']
        print(f"Classification model loaded from {model_path}")
    
    def classify(self, mask_image: np.ndarray) -> Tuple[str, float]:
        """
        Classify crack orientation from masked image
        
        Args:
            mask_image: Segmented mask image (numpy array, grayscale)
            
        Returns:
            Tuple of (crack_type, confidence)
        """
        # Preprocess
        if len(mask_image.shape) == 3:
            mask_image = cv2.cvtColor(mask_image, cv2.COLOR_RGB2GRAY)
        
        img = cv2.resize(mask_image, (256, 256))
        img = img.astype('float32') / 255.0
        img = img.reshape(1, 256, 256, 1)
        
        # Predict
        predictions = self.model.predict(img, verbose=0)
        class_idx = np.argmax(predictions[0])
        confidence = float(predictions[0][class_idx])
        
        crack_type = self.categories[class_idx]
        return crack_type, confidence


class CrackAnalysisPipeline:
    """Main pipeline orchestrator for complete crack analysis"""
    
    def __init__(self, 
                 yolo_model_path: str,
                 segmentation_model_path: str,
                 classification_model_path: str,
                 rag_data_dir: Optional[str] = None,
                 device: str = 'cuda'):
        """
        Initialize the complete pipeline
        
        Args:
            yolo_model_path: Path to YOLO best.pt
            segmentation_model_path: Path to pretrained_net_G.pth
            classification_model_path: Path to crack_orientation_classifier.h5
            rag_data_dir: Directory for RAG system (optional)
            device: 'cuda' or 'cpu'
        """
        print("Initializing Crack Analysis Pipeline...")
        
        # Initialize YOLO detector
        print("Loading YOLO detection model...")
        self.yolo_detector = YOLO(yolo_model_path)
        self.yolo_device = 'cuda' if torch.cuda.is_available() and device == 'cuda' else 'cpu'
        print(f"YOLO model loaded on {self.yolo_device}")
        
        # Initialize segmentation model
        print("Loading segmentation model...")
        self.segmentation_model = SegmentationModel(segmentation_model_path, device)
        
        # Initialize classification model
        print("Loading classification model...")
        self.classification_model = ClassificationModel(classification_model_path)
        
        # Initialize RAG system (optional)
        self.rag_system = None
        if rag_data_dir and StructuralCrackRAG:
            try:
                print("Initializing RAG system...")
                self.rag_system = StructuralCrackRAG(data_dir=rag_data_dir)
                print("RAG system initialized successfully")
            except Exception as e:
                print(f"Warning: RAG system initialization failed: {e}")
                print("Continuing without RAG analysis...")
        
        print("Pipeline initialization complete!")
    
    def process_image(self, 
                     image_path: str,
                     use_rag: bool = True,
                     save_intermediate: bool = False,
                     output_dir: Optional[str] = None) -> Dict[str, Any]:
        """
        Process a single image through the complete pipeline
        
        Args:
            image_path: Path to input image
            use_rag: Whether to use RAG analysis
            save_intermediate: Whether to save intermediate results
            output_dir: Directory to save intermediate results
            
        Returns:
            Complete analysis results dictionary
        """
        results = {
            'success': False,
            'image_path': image_path,
            'detection': None,
            'segmentation': None,
            'classification': None,
            'rag_analysis': None,
            'error': None
        }
        
        try:
            # Step 1: YOLO Detection
            print("\n" + "="*50)
            print("STEP 1: Crack Detection (YOLO)")
            print("="*50)
            
            detection_results = self.yolo_detector(
                image_path,
                conf=0.25,
                iou=0.45,
                device=self.yolo_device
            )
            
            result = detection_results[0]
            detections = []
            
            if result.boxes is not None:
                boxes = result.boxes.xyxy.cpu().numpy()
                confidences = result.boxes.conf.cpu().numpy()
                classes = result.boxes.cls.cpu().numpy()
                
                for box, conf, cls in zip(boxes, confidences, classes):
                    x1, y1, x2, y2 = box
                    detections.append({
                        'bbox': [int(x1), int(y1), int(x2), int(y2)],
                        'confidence': float(conf),
                        'class': self.yolo_detector.names[int(cls)]
                    })
            
            num_cracks = len(detections)
            results['detection'] = {
                'detections': detections,
                'num_cracks': num_cracks,
                'crack_detected': num_cracks > 0,
                'boxes': [det['bbox'] for det in detections]  # Add boxes for compatibility
            }
            
            print(f"Detected {num_cracks} crack(s)")
            
            # Save detection visualization if cracks detected and save_intermediate is True
            if num_cracks > 0 and save_intermediate and output_dir:
                os.makedirs(output_dir, exist_ok=True)
                
                # Load original image for visualization
                import cv2
                image = cv2.imread(image_path)
                
                # Draw bounding boxes
                for detection in detections:
                    bbox = detection['bbox']
                    conf = detection['confidence']
                    x1, y1, x2, y2 = bbox
                    
                    # Draw rectangle
                    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    
                    # Add confidence text
                    label = f"Crack {conf:.2f}"
                    label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                    cv2.rectangle(image, (x1, y1 - label_size[1] - 10), 
                                (x1 + label_size[0], y1), (0, 255, 0), -1)
                    cv2.putText(image, label, (x1, y1 - 5), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
                
                # Save the detection image
                base_name = os.path.splitext(os.path.basename(image_path))[0]
                detection_image_path = os.path.join(output_dir, f'{base_name}_detections.jpg')
                cv2.imwrite(detection_image_path, image)
                results['detection']['detection_image_path'] = detection_image_path
                print(f"Detection visualization saved to {detection_image_path}")
            
            # If no cracks detected, return early
            if num_cracks == 0:
                results['success'] = True
                results['message'] = "No cracks detected in the image"
                return results
            
            # Step 2: Process each detected crack region
            print("\n" + "="*50)
            print("STEP 2: Image Segmentation & Classification")
            print("="*50)
            
            # Load original image for cropping
            original_image = cv2.imread(image_path)
            
            # Process each detected crack region
            all_masks = []
            all_classifications = []
            best_classification = None
            highest_confidence = 0.0
            
            for i, detection in enumerate(detections):
                bbox = detection['bbox']
                x1, y1, x2, y2 = bbox
                print(f"\nProcessing crack region {i+1}/{num_cracks}: bbox [{x1}, {y1}, {x2}, {y2}]")
                
                # Add padding around bounding box (10% of bbox dimensions)
                width = x2 - x1
                height = y2 - y1
                padding_x = max(5, int(0.1 * width))
                padding_y = max(5, int(0.1 * height))
                
                # Calculate padded coordinates (ensure within image bounds)
                img_height, img_width = original_image.shape[:2]
                x1_padded = max(0, x1 - padding_x)
                y1_padded = max(0, y1 - padding_y)
                x2_padded = min(img_width, x2 + padding_x)
                y2_padded = min(img_height, y2 + padding_y)
                
                # Crop the region
                cropped_region = original_image[y1_padded:y2_padded, x1_padded:x2_padded]
                
                # Save cropped region if requested
                if save_intermediate and output_dir:
                    base_name = os.path.splitext(os.path.basename(image_path))[0]
                    crop_path = os.path.join(output_dir, f'{base_name}_crop_{i+1}.jpg')
                    cv2.imwrite(crop_path, cropped_region)
                    print(f"Cropped region {i+1} saved to {crop_path}")
                
                # Create temporary file for cropped region
                with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp_file:
                    crop_temp_path = tmp_file.name
                    cv2.imwrite(crop_temp_path, cropped_region)
                
                try:
                    # Segmentation on cropped region
                    print(f"Running segmentation on crop {i+1}...")
                    mask = self.segmentation_model.segment(crop_temp_path)
                    
                    # Store mask with region info
                    mask_info = {
                        'mask': mask,
                        'mask_shape': mask.shape,
                        'region_id': i+1,
                        'bbox': bbox,
                        'padded_bbox': [x1_padded, y1_padded, x2_padded, y2_padded]
                    }
                    all_masks.append(mask_info)
                    
                    # Save individual mask if requested
                    if save_intermediate and output_dir:
                        mask_path = os.path.join(output_dir, f'segmented_mask_region_{i+1}.jpg')
                        cv2.imwrite(mask_path, mask)
                        mask_info['mask_path'] = mask_path
                        print(f"Mask for region {i+1} saved to {mask_path}")
                    
                    # Classification on the segmented mask
                    print(f"Running classification on crop {i+1}...")
                    crack_type, confidence = self.classification_model.classify(mask)
                    
                    classification_info = {
                        'crack_type': crack_type,
                        'confidence': confidence,
                        'region_id': i+1,
                        'bbox': bbox
                    }
                    all_classifications.append(classification_info)
                    
                    print(f"Region {i+1}: {crack_type} (confidence: {confidence:.2%})")
                    
                    # Track best classification (highest confidence)
                    if confidence > highest_confidence:
                        highest_confidence = confidence
                        best_classification = classification_info
                
                finally:
                    # Clean up temporary file
                    if os.path.exists(crop_temp_path):
                        os.unlink(crop_temp_path)
            
            # Store segmentation results
            results['segmentation'] = {
                'masks': all_masks,
                'num_regions': len(all_masks),
                'combined_mask_available': False
            }
            
            # Create combined mask if requested and save_intermediate is True
            if save_intermediate and output_dir and all_masks:
                # Create a combined mask by placing individual masks back in full image coordinates
                combined_mask = np.zeros((original_image.shape[0], original_image.shape[1]), dtype=np.uint8)
                
                for mask_info in all_masks:
                    mask = mask_info['mask']
                    x1_padded, y1_padded, x2_padded, y2_padded = mask_info['padded_bbox']
                    
                    # Resize mask to match the padded region size
                    target_height = y2_padded - y1_padded
                    target_width = x2_padded - x1_padded
                    mask_resized = cv2.resize(mask, (target_width, target_height))
                    
                    # Place mask in combined image
                    combined_mask[y1_padded:y2_padded, x1_padded:x2_padded] = np.maximum(
                        combined_mask[y1_padded:y2_padded, x1_padded:x2_padded], 
                        mask_resized
                    )
                
                # Save combined mask
                base_name = os.path.splitext(os.path.basename(image_path))[0]
                combined_mask_path = os.path.join(output_dir, f'{base_name}_combined_mask.jpg')
                cv2.imwrite(combined_mask_path, combined_mask)
                
                results['segmentation']['combined_mask'] = combined_mask
                results['segmentation']['combined_mask_path'] = combined_mask_path
                results['segmentation']['combined_mask_available'] = True
                print(f"Combined mask saved to {combined_mask_path}")
            
            # Store classification results
            results['classification'] = {
                'all_classifications': all_classifications,
                'best_classification': best_classification,
                'num_regions': len(all_classifications)
            }
            
            # Add backward compatibility fields
            if best_classification:
                results['classification']['crack_type'] = best_classification['crack_type']
                results['classification']['confidence'] = best_classification['confidence']
                results['classification']['predicted_class'] = best_classification['crack_type']  # For batch script compatibility
            
            print(f"\nBest classification: {best_classification['crack_type']} (confidence: {best_classification['confidence']:.2%})")
            
            # Step 3: RAG Analysis (if enabled and available)
            if use_rag and self.rag_system and CrackType and best_classification:
                print("\n" + "="*50)
                print("STEP 3: RAG Analysis")
                print("="*50)
                
                try:
                    # Use the best classification for RAG analysis
                    crack_type_str = best_classification['crack_type']
                    crack_type_enum = CrackType(crack_type_str)
                    
                    # Use ORIGINAL full image for RAG analysis (not cropped regions)
                    rag_results = self.rag_system.analyze_crack(
                        image_path=image_path,  # Original full image
                        crack_type=crack_type_enum,
                        use_rag=True
                    )
                    
                    results['rag_analysis'] = rag_results
                    
                    print("RAG analysis complete")
                except Exception as e:
                    print(f"RAG analysis failed: {e}")
                    results['rag_analysis'] = {
                        'success': False,
                        'error': str(e)
                    }
            else:
                if not use_rag:
                    print("RAG analysis skipped (disabled)")
                elif not self.rag_system:
                    print("RAG analysis skipped (system not initialized)")
                elif not best_classification:
                    print("RAG analysis skipped (no valid classification)")
                else:
                    print("RAG analysis skipped (CrackType not available)")
            
            results['success'] = True
            print("\n" + "="*50)
            print("PIPELINE COMPLETE")
            print("="*50)
            
        except Exception as e:
            results['error'] = str(e)
            print(f"\nPipeline error: {e}")
            import traceback
            traceback.print_exc()
        
        return results
    
    def get_summary(self, results: Dict[str, Any]) -> str:
        """Generate a human-readable summary of results"""
        if not results['success']:
            return f"Analysis failed: {results.get('error', 'Unknown error')}"
        
        summary_lines = []
        summary_lines.append("CRACK ANALYSIS SUMMARY")
        summary_lines.append("=" * 50)
        
        # Detection
        detection = results.get('detection', {})
        num_cracks = detection.get('num_cracks', 0)
        summary_lines.append(f"\nDetection: {num_cracks} crack(s) detected")
        
        if num_cracks == 0:
            return "\n".join(summary_lines)
        
        # Classification
        classification = results.get('classification', {})
        
        # Handle new structure with multiple regions
        if 'all_classifications' in classification:
            all_classifications = classification['all_classifications']
            best_classification = classification['best_classification']
            
            summary_lines.append(f"Regions processed: {len(all_classifications)}")
            
            if best_classification:
                crack_type = best_classification['crack_type']
                confidence = best_classification['confidence']
                region_id = best_classification['region_id']
                summary_lines.append(f"Best classification: {crack_type} (confidence: {confidence:.2%}, region {region_id})")
            
            # Add details for each region
            if len(all_classifications) > 1:
                summary_lines.append("\nAll regions:")
                for cls_info in all_classifications:
                    region_id = cls_info['region_id']
                    crack_type = cls_info['crack_type']
                    confidence = cls_info['confidence']
                    summary_lines.append(f"  Region {region_id}: {crack_type} ({confidence:.2%})")
        else:
            # Backward compatibility with old structure
            crack_type = classification.get('crack_type', 'Unknown')
            confidence = classification.get('confidence', 0)
            summary_lines.append(f"Classification: {crack_type} (confidence: {confidence:.2%})")
        
        # RAG Analysis
        rag_analysis = results.get('rag_analysis', {})
        if rag_analysis and rag_analysis.get('success'):
            summary = rag_analysis.get('summary', {})
            report = summary.get('comprehensive_report', '')
            if report:
                summary_lines.append(f"\nRAG Analysis Report:\n{report}")
        
        return "\n".join(summary_lines)


# Standalone utility functions for model loading
def load_segmentation_model_utils():
    """Load model architecture utilities from notebook"""
    # This will be implemented by extracting the model definition
    # For now, we'll create a simplified version
    pass

