"""
Main script for generating artificial cracks on brick wall images.
"""

import os
import cv2
import random
import argparse
import logging
import time
import base64
import json
import csv
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()

import config
from crack_generator import add_realistic_crack, add_blur_and_noise
from texture_overlay import apply_random_texture_overlay, create_texture_directory
from data_augmentation import apply_random_augmentation
from annotation import save_annotation, validate_generation
from utils import load_image, save_image, create_dataset_splits, create_data_yaml, ensure_min_bbox_size, save_visualization

# Try to import Gemini (optional dependency)
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    print("Warning: google-generativeai not installed. Reasoning generation will be disabled.")
    print("Install with: pip install google-generativeai")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("crack_generation.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Gemini Configuration
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')  # Set your API key here or use environment variable
GEMINI_MODEL = "gemini-2.5-flash"  # Updated to supported model

def create_csv_file(output_dir):
    """Create CSV file for storing crack analysis results."""
    csv_path = os.path.join(output_dir, "crack_analysis_results.csv")
    
    # Create CSV file with headers if it doesn't exist
    if not os.path.exists(csv_path):
        with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Image Name', 'Image Path (Relative)', 'Crack Type', 'Classification Categories'])
    
    return csv_path

def add_to_csv(csv_path, image_name, relative_path, crack_type, classification_categories):
    """Add a row to the CSV file with error handling and file locking."""
    import fcntl
    max_retries = 5
    retry_delay = 0.1
    
    for attempt in range(max_retries):
        try:
            # Use file locking to prevent concurrent writes
            with open(csv_path, 'a', newline='', encoding='utf-8') as csvfile:
                # Try to lock the file
                fcntl.flock(csvfile.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                
                writer = csv.writer(csvfile)
                writer.writerow([image_name, relative_path, crack_type, classification_categories])
                
                # Explicitly flush and sync
                csvfile.flush()
                os.fsync(csvfile.fileno())
                
            logger.info(f"Added to CSV: {image_name} - {classification_categories}")
            return  # Success
            
        except (IOError, OSError) as e:
            if attempt < max_retries - 1:
                logger.warning(f"CSV write attempt {attempt + 1} failed, retrying: {e}")
                time.sleep(retry_delay * (2 ** attempt))  # Exponential backoff
                continue
            else:
                logger.error(f"Failed to add to CSV after {max_retries} attempts: {str(e)}")
                break
        except Exception as e:
            logger.error(f"Error adding to CSV: {str(e)}")
            break

def extract_classification_from_analysis(analysis_text):
    """Extract clean classification categories from the analysis text."""
    if not analysis_text:
        return ""
    
    # Clean up the text
    cleaned_text = analysis_text.strip()
    
    # Remove common prefixes that might appear
    prefixes_to_remove = [
        "Based on the image analysis:",
        "The crack can be classified as:",
        "Classification:",
        "Categories:",
        "The categories are:",
        "Analysis result:",
        "Result:",
        "CRACK ANALYSIS REPORT",
        "Generated:",
        "Model:",
        "Image:"
    ]
    
    for prefix in prefixes_to_remove:
        if cleaned_text.lower().startswith(prefix.lower()):
            cleaned_text = cleaned_text[len(prefix):].strip()
    
    # Remove any trailing explanatory text (look for common patterns)
    if "." in cleaned_text and not cleaned_text.endswith("Damage"):
        # Split on period and take only the first part if it seems to be explanatory
        parts = cleaned_text.split(".")
        if len(parts) > 1 and len(parts[0]) < 100:  # Reasonable length for categories
            cleaned_text = parts[0].strip()
    
    # Remove timestamps and other metadata patterns
    lines = cleaned_text.split('\n')
    category_line = ""
    
    for line in lines:
        line = line.strip()
        # Skip empty lines and metadata lines
        if not line or ':' in line and len(line.split(':')[0]) < 20:
            continue
        # Skip lines that look like timestamps or file paths
        if any(x in line for x in ['2025-', '.jpg', '.png', 'Model:', 'Generated:', 'Image:']):
            continue
        # Skip lines that start with dashes (separators)
        if line.startswith('---'):
            continue
        
        # This should be our categories line
        category_line = line
        break
    
    # Clean up quotes and brackets
    category_line = category_line.strip('"\'[]{}()')
    
    return category_line if category_line else cleaned_text

def setup_gemini():
    """Setup Gemini AI for crack analysis."""
    global GEMINI_API_KEY
    if not GEMINI_AVAILABLE:
        return None
    
    # Try to get API key from environment or config
    GEMINI_API_KEY = os.getenv('GEMINI_API_KEY') or GEMINI_API_KEY
    
    if not GEMINI_API_KEY:
        logger.warning("No Gemini API key found. Set GEMINI_API_KEY environment variable or update the code.")
        return None
    
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        model = genai.GenerativeModel(GEMINI_MODEL)
        logger.info("Gemini AI configured successfully")
        return model
    except Exception as e:
        logger.error(f"Failed to setup Gemini: {str(e)}")
        return None

def image_to_base64(image_path):
    """Convert image to base64 string."""
    try:
        with open(image_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
            return encoded_string
    except Exception as e:
        logger.error(f"Error converting image to base64: {str(e)}")
        return None

def analyze_crack_image(model, image_path, crack_type):
    """Analyze crack image using Gemini AI and return detailed analysis."""
    if not model:
        return None
    
    try:
        # Load the image
        image = cv2.imread(image_path)
        if image is None:
            logger.error(f"Could not load image for analysis: {image_path}")
            return None
        # Convert to PIL Image format for Gemini
        from PIL import Image as PILImage
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_image = PILImage.fromarray(image_rgb)
        
        # Create the simplified classification prompt
        prompt = f"""Analyze this {crack_type} crack image and identify ONLY the most likely and definite causes based on clear visual evidence. Do not include speculative or merely possible causes.

Classify into one or more categories from this exact list ONLY if you can see clear evidence: Foundation Settling, Thermal Stress, Moisture Damage, Structural Overload, Material Shrinkage, Vegetation Damage, Poor Construction Quality, Seismic Activity, Chemical Deterioration, Freeze-Thaw Cycles, Corrosion-Induced, Age-Related Deterioration, Material Fatigue, Ground Movement, Impact Damage.

Requirements:
- Only select causes that have clear visual indicators in the image
- Do not include causes that are merely possible but not evident
- Be conservative in your classification
- If no definite cause is visible, return "Undetermined"

IMPORTANT: Return ONLY the category names separated by commas. No other text, explanations, or formatting.

Examples of what to look for:
- Moisture Damage: visible water stains, efflorescence, discoloration
- Poor Construction Quality: irregular mortar joints, misaligned bricks
- Thermal Stress: cracks following expansion joint patterns
- Foundation Settling: stepped cracks following mortar lines
"""

        # Generate response
        response = model.generate_content([prompt, pil_image])
        
        if response and hasattr(response, 'text'):
            # Clean up the response to ensure only categories are returned
            analysis_text = response.text.strip()
            
            # Remove any common prefixes/suffixes that might appear
            prefixes_to_remove = [
                "Based on the image analysis:",
                "The crack can be classified as:",
                "Classification:",
                "Categories:",
                "The categories are:",
                "Analysis result:",
                "Result:"
            ]
            
            for prefix in prefixes_to_remove:
                if analysis_text.lower().startswith(prefix.lower()):
                    analysis_text = analysis_text[len(prefix):].strip()
            
            # Remove any trailing explanatory text (look for common patterns)
            if "." in analysis_text and not analysis_text.endswith("Damage"):
                # Split on period and take only the first part if it seems to be explanatory
                parts = analysis_text.split(".")
                if len(parts) > 1 and len(parts[0]) < 100:  # Reasonable length for categories
                    analysis_text = parts[0].strip()
            
            # Clean up quotes and brackets
            analysis_text = analysis_text.strip('"\'[]{}()')
            
            return {
                'crack_type': crack_type,
                'analysis': analysis_text,
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                'model_used': GEMINI_MODEL
            }
        else:
            logger.error("No response received from Gemini")
            return None
            
    except Exception as e:
        logger.error(f"Error analyzing crack image: {str(e)}")
        return None

def save_crack_reasoning(analysis_result, image_path, reasons_dir):
    """Save the crack analysis reasoning to a text file."""
    if not analysis_result:
        return None
    
    try:
        # Create reasoning filename based on image name
        image_name = os.path.splitext(os.path.basename(image_path))[0]
        reason_filename = f"{image_name}.txt"
        reason_path = os.path.join(reasons_dir, reason_filename)
        
        # Format the analysis result
        reasoning_content = f"""CRACK ANALYSIS REPORT
Generated: {analysis_result['timestamp']}
Model: {analysis_result['model_used']}
Image: {os.path.basename(image_path)}

{analysis_result['analysis']}

---
This analysis was generated automatically using AI and should be reviewed by a structural engineer for critical assessments.
"""
        
        # Write to file
        with open(reason_path, 'w', encoding='utf-8') as f:
            f.write(reasoning_content)
        
        logger.info(f"Reasoning saved to: {reason_path}")
        return reason_path
        
    except Exception as e:
        logger.error(f"Error saving reasoning: {str(e)}")
        return None

def process_image(image_path, output_dir, crack_type="random", use_augmentation=True, use_texture_overlay=False, visualize=False, vis_dir=None, gemini_model=None, use_mortar_joints=True, mortar_probability=0.6, csv_path=None):
    """
    Process a single image to add artificial cracks and save annotations.
    
    Args:
        image_path (str): Path to the input image
        output_dir (str): Directory to save the output
        crack_type (str): Type of crack to generate
        use_augmentation (bool): Whether to apply augmentations
        use_texture_overlay (bool): Whether to use texture overlay method
        visualize (bool): Whether to save a visualization of the bounding box
        vis_dir (str): Directory to save visualizations
        gemini_model: Gemini AI model for crack analysis
        use_mortar_joints (bool): Whether to detect and follow mortar joints for realistic cracks
        mortar_probability (float): Probability of following mortar joints vs going through bricks
        
    Returns:
        tuple: (Path to the generated image, Path to the annotation file, Path to visualization if enabled, Path to reasoning file)
    """
    try:
        # Load the image
        original_image = load_image(image_path)
        
        # Generate a unique ID using timestamp, random number, and process ID to avoid collisions
        timestamp = int(time.time() * 1000000)  # microseconds for better uniqueness
        random_id = random.randint(1000, 9999)
        process_id = os.getpid() % 1000  # Add process ID for multi-process safety
        image_id = f"{timestamp}_{process_id}_{random_id}"
        
        image_name = os.path.splitext(os.path.basename(image_path))[0]
        output_image_name = f"{image_name}_crack_{image_id}.jpg"
        
        # Determine crack type first to create appropriate directory structure
        temp_crack_type = crack_type
        if crack_type == "random":
            temp_crack_type = random.choice(list(config.CRACK_COUNTS.keys()))
        
        # Create crack-type specific directories with subdirectories
        crack_type_dir = os.path.join(output_dir, temp_crack_type)
        crack_images_dir = os.path.join(crack_type_dir, "images")
        crack_annotations_dir = os.path.join(crack_type_dir, "annotations")
        crack_reasons_dir = os.path.join(crack_type_dir, "reasons")
        
        # Ensure unique filename by checking if file already exists
        base_output_path = os.path.join(crack_images_dir, output_image_name)
        counter = 1
        output_image_path = base_output_path
        final_output_name = output_image_name
        
        while os.path.exists(output_image_path):
            name_without_ext = os.path.splitext(output_image_name)[0]
            ext = os.path.splitext(output_image_name)[1]
            final_output_name = f"{name_without_ext}_{counter}{ext}"
            output_image_path = os.path.join(crack_images_dir, final_output_name)
            counter += 1
        
        # Create the output directories if they don't exist
        os.makedirs(crack_images_dir, exist_ok=True)
        os.makedirs(crack_annotations_dir, exist_ok=True)
        os.makedirs(crack_reasons_dir, exist_ok=True)
        
        # Choose the crack generation method
        if use_texture_overlay and random.random() < 0.5:
            # Use texture overlay method
            cracked_image, crack_type, bbox = apply_random_texture_overlay(original_image.copy())
        else:
            # Use procedural drawing method with mortar joint detection
            crack_type = temp_crack_type  # Use the determined crack type
            cracked_image, crack_type, bbox = add_realistic_crack(
                original_image.copy(), 
                crack_type, 
                use_mortar_joints=use_mortar_joints, 
                mortar_probability=mortar_probability
            )
            cracked_image = add_blur_and_noise(cracked_image)
        
        # Ensure minimum bounding box size
        bbox = ensure_min_bbox_size(bbox, min_size=20)
        
        # Apply augmentations if requested
        if use_augmentation:
            cracked_image, bbox = apply_random_augmentation(cracked_image, bbox)
        
        # Save the image
        save_image(cracked_image, output_image_path)
        logger.info(f"    Saved image: {output_image_path}")
        
        # Create annotation filename that exactly matches the image filename
        annotation_base_name = os.path.splitext(final_output_name)[0]
        logger.info(f"    Creating annotation with base name: {annotation_base_name}")
        
        # Save annotation in the configured format with explicit filename matching
        annotation_path = save_annotation(
            output_image_path, 
            bbox, 
            crack_annotations_dir,
            config.DEFAULT_ANNOTATION_FORMAT,
            annotation_filename=annotation_base_name
        )
        
        logger.info(f"    Annotation saved to: {annotation_path}")
        
        # Validate that both image and annotation were created successfully
        if not validate_generation(output_image_path, annotation_path, config.DEFAULT_ANNOTATION_FORMAT):
            logger.error(f"    Validation failed for: {output_image_path}")
            # Clean up any files that might have been partially created
            try:
                if os.path.exists(output_image_path):
                    os.remove(output_image_path)
                if annotation_path and os.path.exists(annotation_path):
                    os.remove(annotation_path)
            except Exception as cleanup_error:
                logger.error(f"    Error cleaning up failed files: {cleanup_error}")
            return None, None, None, None
        
        logger.info(f"    Successfully validated: image and annotation created")
        
        # Log successful generation
        logger.info(f"    Generated: {os.path.basename(output_image_path)}")
        
        # Save visualization if requested
        vis_path = None
        if visualize and vis_dir:
            os.makedirs(vis_dir, exist_ok=True)
            vis_filename = f"vis_{final_output_name}"
            vis_path = os.path.join(vis_dir, vis_filename)
            save_visualization(cracked_image, bbox, vis_path)
        
        # Generate crack reasoning using Gemini AI
        reason_path = None
        classification_categories = ""
        if gemini_model:
            logger.info(f"    Generating reasoning for {crack_type} crack...")
            analysis_result = analyze_crack_image(gemini_model, output_image_path, crack_type)
            if analysis_result:
                reason_path = save_crack_reasoning(analysis_result, output_image_path, crack_reasons_dir)
                # Extract classification categories for CSV
                classification_categories = extract_classification_from_analysis(analysis_result.get('analysis', ''))
        
        # Add to CSV if provided
        if csv_path:
            # Calculate relative path from output directory
            relative_path = os.path.relpath(output_image_path, output_dir)
            image_filename = os.path.basename(output_image_path)
            add_to_csv(csv_path, image_filename, relative_path, crack_type, classification_categories)
        
        return output_image_path, annotation_path, vis_path, reason_path
        
    except Exception as e:
        logger.error(f"Error processing image {image_path}: {str(e)}")
        return None, None, None, None

def main():
    """
    Main function to run the crack generation process.
    """
    # ================================
    # CONFIGURATION - MODIFY THESE VALUES
    # ================================
    IMAGES_PER_CRACK_TYPE = 42  # Number of images to generate per crack type
    ENABLE_REASONING = True     # Whether to generate crack reasoning using Gemini AI
    USE_MORTAR_JOINTS = True    # Whether to detect and follow mortar joints for realistic cracks
    MORTAR_PROBABILITY = 0.9    # Probability of following mortar joints vs going through bricks (0.0-1.0) - High value for realistic step cracks
    
    parser = argparse.ArgumentParser(description="Generate artificial cracks on brick wall images")
    parser.add_argument("--input_dir", default=config.DATA_DIR, help="Directory containing input images")
    parser.add_argument("--output_dir", default=config.OUTPUT_DIR, help="Directory to save output images and annotations")
    parser.add_argument("--count", type=int, default=IMAGES_PER_CRACK_TYPE, help="Number of cracked images to generate per original image per crack type")
    parser.add_argument("--no_augmentation", action="store_true", help="Disable data augmentation")
    parser.add_argument("--texture_overlay", action="store_true", help="Enable texture overlay method")
    parser.add_argument("--create_splits", action="store_true", help="Create train/val/test splits")
    parser.add_argument("--visualize", action="store_true", help="Save visualizations of generated images with bounding boxes")
    parser.add_argument("--no_reasoning", action="store_true", help="Disable AI-powered crack reasoning generation")
    parser.add_argument("--no_mortar_joints", action="store_true", help="Disable mortar joint detection and following")
    parser.add_argument("--mortar_probability", type=float, default=MORTAR_PROBABILITY, 
                       help="Probability of following mortar joints vs going through bricks (0.0-1.0)")
    parser.add_argument("--debug_mortar", action="store_true", help="Save debug images showing mortar joint detection")
    
    args = parser.parse_args()
    
    # Setup Gemini AI for reasoning (if enabled)
    gemini_model = None
    if ENABLE_REASONING and not args.no_reasoning:
        gemini_model = setup_gemini()
        if gemini_model:
            logger.info("Crack reasoning generation enabled")
        else:
            logger.warning("Crack reasoning disabled due to setup issues")
    else:
        logger.info("Crack reasoning generation disabled")
    
    # Configure mortar joint detection
    use_mortar_joints = USE_MORTAR_JOINTS and not args.no_mortar_joints
    mortar_probability = max(0.0, min(1.0, args.mortar_probability))  # Clamp to valid range
    
    if use_mortar_joints:
        logger.info(f"Mortar joint detection enabled (probability: {mortar_probability:.2f})")
        if args.debug_mortar:
            logger.info("Debug mode enabled - will save mortar joint detection images")
    else:
        logger.info("Mortar joint detection disabled")
    
    # Set visualization directory
    args.vis_dir = os.path.join(args.output_dir, "visualizations") if args.visualize else None
    
    # Create main output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Create CSV file for tracking results
    csv_path = create_csv_file(args.output_dir)
    logger.info(f"CSV file created: {csv_path}")
    
    # Create texture directory if using texture overlay
    if args.texture_overlay:
        create_texture_directory()
    
    # Get list of input images
    input_images = [
        os.path.join(args.input_dir, f) for f in os.listdir(args.input_dir)
        if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp'))
    ]
    
    if not input_images:
        logger.error(f"No images found in {args.input_dir}")
        return
    
    logger.info(f"Found {len(input_images)} input images")
    logger.info(f"Crack types to generate: {config.CRACK_TYPES}")
    logger.info(f"Images per crack type per input image: {args.count}")
    logger.info(f"Total images to generate: {len(input_images) * len(config.CRACK_TYPES) * args.count}")
    
    # Track total images generated
    total_generated = 0
    failed_generations = 0
    total_reasonings = 0
    
    # Process each image
    for image_path in tqdm(input_images, desc="Processing images"):
        image_name = os.path.basename(image_path)
        logger.info(f"Processing image: {image_name}")
        
        # Generate images for each crack type
        for crack_type in config.CRACK_TYPES:
            logger.info(f"  Generating {crack_type} cracks...")
            
            # Generate multiple versions of this crack type
            for crack_idx in range(args.count):
                output_path, annotation_path, vis_path, reason_path = process_image(
                    image_path, 
                    args.output_dir, 
                    crack_type=crack_type,  # Use specific crack type
                    use_augmentation=not args.no_augmentation,
                    use_texture_overlay=args.texture_overlay,
                    visualize=args.visualize,
                    vis_dir=args.vis_dir,
                    gemini_model=gemini_model,
                    use_mortar_joints=use_mortar_joints,
                    mortar_probability=mortar_probability,
                    csv_path=csv_path
                )
                
                if output_path and annotation_path:
                    total_generated += 1
                    if vis_path:
                        logger.info(f"    Visualization saved to {vis_path}")
                    if reason_path:
                        total_reasonings += 1
                        logger.info(f"    Reasoning saved to {reason_path}")
                else:
                    failed_generations += 1
    
    logger.info(f"Generation complete. Generated {total_generated} cracked images.")
    if total_reasonings > 0:
        logger.info(f"Generated {total_reasonings} crack analysis reports.")
    if failed_generations > 0:
        logger.warning(f"{failed_generations} generations failed. Check the log for details.")
    
    # Create dataset splits if requested
    if args.create_splits and total_generated > 0:
        logger.info("Creating dataset splits...")
        
        # Create splits for each crack type
        for crack_type in config.CRACK_TYPES:
            crack_type_dir = os.path.join(args.output_dir, crack_type)
            crack_images_dir = os.path.join(crack_type_dir, "images")
            crack_annotations_dir = os.path.join(crack_type_dir, "annotations")
            
            if os.path.exists(crack_images_dir) and os.path.exists(crack_annotations_dir):
                logger.info(f"Creating splits for {crack_type} cracks...")
                train_dir, val_dir, test_dir = create_dataset_splits(
                    crack_images_dir,
                    crack_annotations_dir,
                    crack_type_dir
                )
                logger.info(f"  {crack_type} splits created: {train_dir}, {val_dir}, {test_dir}")
        
        # Create overall data.yaml for YOLOv5 training
        yaml_path = create_data_yaml(args.output_dir)
        logger.info(f"YAML file created: {yaml_path}")

if __name__ == "__main__":
    start_time = time.time()
    main()
    elapsed = time.time() - start_time
    logger.info(f"Total execution time: {elapsed:.2f} seconds")
