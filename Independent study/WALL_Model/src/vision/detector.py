import torch
from transformers import OwlViTProcessor, OwlViTForObjectDetection
from PIL import Image
import numpy as np
import cv2
from typing import List, Tuple

class OwlVitDetector:
    def __init__(self, model_name="google/owlvit-base-patch32", score_threshold=0.05):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Loading OWL-ViT model '{model_name}' on {self.device}...")
        
        self.processor = OwlViTProcessor.from_pretrained(model_name)
        self.model = OwlViTForObjectDetection.from_pretrained(model_name).to(self.device)
        self.model.eval()
        self.score_threshold = score_threshold

    def detect(self, image: np.ndarray, text_queries: List[str] = ["window", "door", "frame", "door frame", "window frame", "archway", "entrance", "opening", "lintel", "floor", "roof", "ceiling", "ground", "baseboard", "skirting board"]) -> List[dict]:
        """
        Detects objects in the image based on text queries.
        Returns a list of dicts: {'label': str, 'score': float, 'bbox': [x, y, w, h]}
        """
        # Convert OpenCV Image (BGR) to PIL Image (RGB)
        pil_img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        
        inputs = self.processor(text=text_queries, images=pil_img, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            
        # Target image size (height, width)
        target_sizes = torch.Tensor([pil_img.size[::-1]]).to(self.device)
        
        # Convert outputs to bounding boxes
        results = self.processor.post_process_object_detection(
            outputs=outputs, 
            target_sizes=target_sizes, 
            threshold=self.score_threshold
        )
        
        detected_objects = []
        
        # We only have one image in the batch
        result = results[0]
        boxes = result["boxes"].cpu().numpy()
        scores = result["scores"].cpu().numpy()
        labels = result["labels"].cpu().numpy()
        
        for i, (box, score, label_idx) in enumerate(zip(boxes, scores, labels)):
            # Label is index into text_queries
            label_text = text_queries[label_idx]
            
            # box is [xmin, ymin, xmax, ymax]
            xmin, ymin, xmax, ymax = box.astype(int)
            w = xmax - xmin
            h = ymax - ymin
            
            # Filter unrealistic boxes (e.g. 1px wide) if needed, 
            # although OWL-ViT is usually decent.
            if w > 10 and h > 10:
                print(f"Detected {label_text} with score {score:.3f} at [{xmin}, {ymin}, {w}, {h}]")
                detected_objects.append({
                    "label": label_text,
                    "score": float(score),
                    "bbox": [xmin, ymin, w, h]
                })
                
        return detected_objects
