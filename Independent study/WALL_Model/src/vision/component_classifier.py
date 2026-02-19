import cv2
import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
from .utils import is_above, is_below, is_left_of, is_right_of, get_bbox_center, rectangles_intersect

@dataclass
class WallComponent:
    id: str
    type: str  # 'Pier', 'Spandrel', 'Opening', 'Node'
    bbox: List[int]  # [x, y, w, h]
    description: str = ""

class WallClassifier:
    def __init__(self, use_ml=True):
        self.components = []
        self.detector = None
        if use_ml:
            try:
                from .detector import OwlVitDetector
                self.detector = OwlVitDetector()
            except ImportError as e:
                print(f"Warning: Could not import OwlVitDetector: {e}. Falling back to heuristic.")
                self.detector = None
            except Exception as e:
                print(f"Warning: Could not load OwlVitDetector: {e}. Falling back to heuristic.")
                self.detector = None
    
    def detect_openings(self, image: np.ndarray) -> List[dict]:
        """
        Detects structural elements.
        Prioritizes ML Detector if available, otherwise falls back to heuristic.
        """
        if self.detector:
            try:
                print("Running OWL-ViT detection...")
                # Use default expanded prompts defined in detector class if not overridden
                return self.detector.detect(image)
            except Exception as e:
                print(f"ML Detection failed: {e}. Falling back to heuristic.")
        
        # Fallback Heuristic
        print("Running Heuristic detection...")
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # 1. Adaptive Threshold processing
        # Invert because windows usually darker? Or adapt based on image.
        # Let's assume drawn lines or distinct rectangles.
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edged = cv2.Canny(blurred, 50, 150)
        
        # Find contours
        contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        openings = []
        min_area = (image.shape[0] * image.shape[1]) * 0.01 # 1% of image
        
        for c in contours:
            x, y, w, h = cv2.boundingRect(c)
            area = w * h
            
            # Filter small noise
            if area < min_area:
                continue
                
            # Filter non-rectangular shapes
            # Return dict format
            openings.append({
                "label": "opening",
                "score": 1.0,
                "bbox": [x, y, w, h]
            })
            
        return openings

    def infer_structure(self, image_shape: Tuple[int, int], detections: List[dict]) -> List[WallComponent]:
        """
        Infers Piers and Spandrels.
        1. Identifies Wall Boundaries (Floor/Roof) to set Y-limits.
        2. Identifies Openings (Windows/Doors).
        3. Infers components relative to Openings within Y-limits.
        """
        self.components = []
        img_h, img_w = image_shape[:2]
        
        # 1. Separate Detections
        openings = []
        wall_top_y = 0
        wall_bottom_y = img_h
        
        for d in detections:
            label = d['label'].lower()
            bbox = d['bbox']
            x, y, w, h = bbox
            
            if any(k in label for k in ["floor", "ground", "skirting", "baseboard"]):
                # If it's a baseboard, the wall ends at the *bottom* of the baseboard? 
                # Or usually the baseboard is PART of the wall. 
                # Let's say wall ends at the top of the floor.
                # If we detect "floor", y is the top of the floor.
                # If we detect "baseboard", the floor starts at y + h (bottom of baseboard)? 
                # Actually, spandrel (wall material) usually stops AT the floor. Baseboard is on the wall.
                # Let's set wall_bottom_y to the top of the detected "floor" concept.
                
                # Logic:
                # If "floor" or "ground": y is top of object.
                # If "baseboard": baseboard is attached to wall bottom. So floor starts at y + h?
                # Let's simplify: 
                # If 'floor' in label: boundary is y.
                # If 'board' in label: boundary is y + h (bottom of board).
                
                current_boundary = y # Default for floor
                if "board" in label:
                    current_boundary = y + h
                    
                if current_boundary < wall_bottom_y:
                    wall_bottom_y = current_boundary
                
                self.components.append(WallComponent(f"Floor_{len(self.components)}", "Boundary", bbox, label))
                
            elif any(k in label for k in ["roof", "ceiling"]):
                if (y + h) > wall_top_y:
                    wall_top_y = y + h
                self.components.append(WallComponent(f"Ceiling_{len(self.components)}", "Boundary", bbox, label))
                
            else:
                # It's an opening
                openings.append(bbox)
        
        if not openings:
            self.components.append(WallComponent(
                id="Unknown_1", 
                type="Undeterminable", 
                bbox=[0, wall_top_y, img_w, wall_bottom_y - wall_top_y], 
                description="No openings detected."
            ))
            return self.components

        # 2. Sort Openings
        openings.sort(key=lambda b: (b[1], b[0]))
        
        # 3. Store Openings
        for i, bbox in enumerate(openings):
             self.components.append(WallComponent(
                id=f"Opening_{i+1}",
                type="Opening",
                bbox=bbox,
                description="Detected Window/Door"
            ))

        # 4. Infer Structure (Piers & Spandrels)
        # A. PIERS
        x_projections = [0, img_w]
        for x, y, w, h in openings:
            x_projections.append(x)
            x_projections.append(x + w)
        x_projections = sorted(list(set(x_projections)))
        
        pier_count = 1
        for i in range(len(x_projections) - 1):
            start_x = x_projections[i]
            end_x = x_projections[i+1]
            width = end_x - start_x
            if width < 5: continue 
   
            is_opening_col = False
            for ox, oy, ow, oh in openings:
                if (start_x < ox + ow) and (end_x > ox):
                     is_opening_col = True
                     break
            
            if not is_opening_col:
                self.components.append(WallComponent(
                    id=f"Pier_{pier_count}",
                    type="Pier",
                    bbox=[start_x, wall_top_y, width, wall_bottom_y - wall_top_y],
                    description=f"Pier at x={start_x}"
                ))
                pier_count += 1

        # B. SPANDRELS
        spandrel_count = 1
        for i, (ox, oy, ow, oh) in enumerate(openings):
            # Top Spandrel
            if oy > wall_top_y + 10:
                self.components.append(WallComponent(
                    id=f"Spandrel_{spandrel_count}",
                    type="Spandrel",
                    bbox=[ox, wall_top_y, ow, oy - wall_top_y],
                    description=f"Spandrel above Opening {i+1}"
                ))
                spandrel_count += 1
                
            # Bottom Spandrel
            if (oy + oh) < (wall_bottom_y - 10):
                self.components.append(WallComponent(
                    id=f"Spandrel_{spandrel_count}",
                    type="Spandrel",
                    bbox=[ox, oy + oh, ow, wall_bottom_y - (oy + oh)],
                    description=f"Spandrel below Opening {i+1}"
                ))
                spandrel_count += 1
        
        return self.components


    def visualize(self, image: np.ndarray, components: List[WallComponent]) -> np.ndarray:
        """Draws bounding boxes and labels on the image."""
        vis_img = image.copy()
        
        colors = {
            "Pier": (0, 0, 255),    # Red
            "Spandrel": (0, 255, 0),# Green
            "Opening": (255, 0, 0), # Blue
            "Node": (0, 255, 255)   # Yellow
        }
        
        for comp in components:
            x, y, w, h = comp.bbox
            color = colors.get(comp.type, (255, 255, 255))
            
            cv2.rectangle(vis_img, (x, y), (x + w, y + h), color, 2)
            cv2.putText(vis_img, f"{comp.type} {comp.id}", (x + 5, y + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                        
        return vis_img
