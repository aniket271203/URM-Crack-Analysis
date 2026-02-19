"""
Mortar joint detection and crack propagation through joints for realistic masonry crack generation.
"""

import cv2
import numpy as np
import random
import math
from typing import List, Tuple, Optional


def find_peaks_simple(data, height_threshold=None, distance=20):
    """
    Simple peak finding algorithm to replace scipy.signal.find_peaks
    """
    peaks = []
    
    if height_threshold is None:
        height_threshold = np.mean(data)
    
    for i in range(1, len(data) - 1):
        # Check if this point is higher than threshold
        if data[i] > height_threshold:
            # Check if it's a local maximum
            if data[i] > data[i-1] and data[i] > data[i+1]:
                # Check distance from previous peaks
                if not peaks or all(abs(i - peak) >= distance for peak in peaks):
                    peaks.append(i)
    
    return peaks


def detect_mortar_joints(image, debug_mode=False):
    """
    Detect horizontal and vertical mortar joints in a brick wall image.
    
    Args:
        image (numpy.ndarray): Input brick wall image
        debug_mode (bool): If True, saves debug images showing detection steps
        
    Returns:
        tuple: (horizontal_joints, vertical_joints, joint_mask)
            - horizontal_joints: List of y-coordinates of horizontal joints
            - vertical_joints: List of x-coordinates of vertical joints  
            - joint_mask: Binary mask showing detected joints
    """
    height, width = image.shape[:2]
    
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    # Step 1: Enhance contrast to make mortar joints more visible
    # Mortar is typically lighter than bricks
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(gray)
    
    # Step 2: Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(enhanced, (5, 5), 0)
    
    # Step 3: Detect edges using adaptive threshold
    # This helps identify the boundaries between bricks and mortar
    adaptive_thresh = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    )
    
    # Step 4: Morphological operations to clean up the image
    kernel_horizontal = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 1))
    kernel_vertical = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 15))
    
    # Detect horizontal lines (horizontal mortar joints)
    horizontal_lines = cv2.morphologyEx(adaptive_thresh, cv2.MORPH_CLOSE, kernel_horizontal)
    horizontal_lines = cv2.morphologyEx(horizontal_lines, cv2.MORPH_OPEN, kernel_horizontal)
    
    # Detect vertical lines (vertical mortar joints)
    vertical_lines = cv2.morphologyEx(adaptive_thresh, cv2.MORPH_CLOSE, kernel_vertical)
    vertical_lines = cv2.morphologyEx(vertical_lines, cv2.MORPH_OPEN, kernel_vertical)
    
    # Step 5: Use Hough Line Transform to find strong line candidates
    # For horizontal joints
    horizontal_joints = []
    lines_h = cv2.HoughLinesP(horizontal_lines, 1, np.pi/180, threshold=int(width*0.3), 
                             minLineLength=int(width*0.4), maxLineGap=30)
    
    if lines_h is not None:
        for line in lines_h:
            x1, y1, x2, y2 = line[0]
            # Filter for mostly horizontal lines
            if abs(y2 - y1) < 10:  # Allow slight variations
                y_avg = (y1 + y2) // 2
                if y_avg not in [j for j in horizontal_joints if abs(j - y_avg) < 20]:
                    horizontal_joints.append(y_avg)
    
    # For vertical joints
    vertical_joints = []
    lines_v = cv2.HoughLinesP(vertical_lines, 1, np.pi/180, threshold=int(height*0.3),
                             minLineLength=int(height*0.4), maxLineGap=30)
    
    if lines_v is not None:
        for line in lines_v:
            x1, y1, x2, y2 = line[0]
            # Filter for mostly vertical lines
            if abs(x2 - x1) < 10:  # Allow slight variations
                x_avg = (x1 + x2) // 2
                if x_avg not in [j for j in vertical_joints if abs(j - x_avg) < 20]:
                    vertical_joints.append(x_avg)
    
    # Step 6: Create joint mask
    joint_mask = np.zeros((height, width), dtype=np.uint8)
    
    # Add horizontal joints to mask
    for y in horizontal_joints:
        if 0 <= y < height:
            cv2.line(joint_mask, (0, y), (width, y), 255, 3)
    
    # Add vertical joints to mask
    for x in vertical_joints:
        if 0 <= x < width:
            cv2.line(joint_mask, (x, 0), (x, height), 255, 3)
    
    # Step 7: Alternative method using intensity analysis
    # Analyze row and column intensities to find consistent light bands
    row_means = np.mean(gray, axis=1)
    col_means = np.mean(gray, axis=0)
    
    # Find peaks in intensity (lighter mortar joints)
    h_peaks = find_peaks_simple(row_means, height_threshold=np.mean(row_means) + 5, distance=20)
    for peak in h_peaks:
        if peak not in [j for j in horizontal_joints if abs(j - peak) < 15]:
            horizontal_joints.append(peak)
            cv2.line(joint_mask, (0, peak), (width, peak), 255, 2)
    
    # Find vertical joints by analyzing column means
    v_peaks = find_peaks_simple(col_means, height_threshold=np.mean(col_means) + 5, distance=20)
    for peak in v_peaks:
        if peak not in [j for j in vertical_joints if abs(j - peak) < 15]:
            vertical_joints.append(peak)
            cv2.line(joint_mask, (peak, 0), (peak, height), 255, 2)
    
    # Sort joints
    horizontal_joints.sort()
    vertical_joints.sort()
    
    # Save debug images if requested
    if debug_mode:
        cv2.imwrite('debug_enhanced.jpg', enhanced)
        cv2.imwrite('debug_threshold.jpg', adaptive_thresh)
        cv2.imwrite('debug_horizontal_lines.jpg', horizontal_lines)
        cv2.imwrite('debug_vertical_lines.jpg', vertical_lines)
        cv2.imwrite('debug_joint_mask.jpg', joint_mask)
        
        # Create visualization
        debug_image = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        for y in horizontal_joints:
            cv2.line(debug_image, (0, y), (width, y), (0, 255, 0), 2)
        for x in vertical_joints:
            cv2.line(debug_image, (x, 0), (x, height), (255, 0, 0), 2)
        cv2.imwrite('debug_detected_joints.jpg', debug_image)
    
    return horizontal_joints, vertical_joints, joint_mask


def generate_mortar_following_crack(image, start, end, horizontal_joints, vertical_joints, 
                                   joint_follow_probability=0.9, thickness_range=(2, 5)):
    """
    Generate a crack that follows mortar joints when possible with reduced vibration.
    
    Args:
        image (numpy.ndarray): Image to draw crack on
        start (tuple): Starting point (x, y)
        end (tuple): Ending point (x, y)
        horizontal_joints (list): Y-coordinates of horizontal joints
        vertical_joints (list): X-coordinates of vertical joints
        joint_follow_probability (float): Probability of following a joint when encountered
        thickness_range (tuple): Min and max thickness of crack
        
    Returns:
        tuple: (crack_points, bounding_box)
    """
    height, width = image.shape[:2]
    crack_points = [start]
    
    # Initialize bounding box
    x_min, y_min = start
    x_max, y_max = start
    
    current_x, current_y = start
    target_x, target_y = end
    
    step_size = 8  # Larger steps for smoother lines
    joint_tolerance = 15  # Slightly larger tolerance for joint detection
    max_iterations = 1000  # Safety limit to prevent infinite loops
    iteration_count = 0
    
    while (math.sqrt((current_x - target_x)**2 + (current_y - target_y)**2) > step_size and 
           iteration_count < max_iterations):
        # Calculate direction to target
        dx = target_x - current_x
        dy = target_y - current_y
        distance = math.sqrt(dx**2 + dy**2)
        
        # If we're very close to the target but still in the loop, adjust step size
        if distance < step_size * 2:
            step_size = max(1, distance / 3)  # Reduce step size when close
        
        # Normalize direction
        dx_norm = dx / distance if distance > 0 else 0
        dy_norm = dy / distance if distance > 0 else 0
        
        # Check if we're near any mortar joints
        near_horizontal_joint = None
        near_vertical_joint = None
        
        for joint_y in horizontal_joints:
            if abs(current_y - joint_y) <= joint_tolerance:
                near_horizontal_joint = joint_y
                break
                
        for joint_x in vertical_joints:
            if abs(current_x - joint_x) <= joint_tolerance:
                near_vertical_joint = joint_x
                break
        
        # Decide whether to follow a joint (higher probability)
        follow_joint = random.random() < joint_follow_probability
        
        if follow_joint and near_horizontal_joint is not None:
            # Follow horizontal joint with minimal vibration
            if abs(dx_norm) > 0.1:  # Only if we're generally moving horizontally
                next_x = current_x + step_size * (1 if dx > 0 else -1)
                next_y = near_horizontal_joint + random.randint(-1, 1) # Minimal variation
            else:
                # Normal movement with reduced randomness
                next_x = current_x + dx_norm * step_size + random.randint(-1, 1)
                next_y = current_y + dy_norm * step_size + random.randint(-1, 1)
                
        elif follow_joint and near_vertical_joint is not None:
            # Follow vertical joint with minimal vibration
            if abs(dy_norm) > 0.1:  # Only if we're generally moving vertically
                next_x = near_vertical_joint + random.randint(-1, 1)  # Minimal variation
                next_y = current_y + step_size * (1 if dy > 0 else -1)
            else:
                # Normal movement with reduced randomness
                next_x = current_x + dx_norm * step_size + random.randint(-1, 1)
                next_y = current_y + dy_norm * step_size + random.randint(-1, 1)
        else:
            # Reduced jagged movement through brick
            next_x = current_x + dx_norm * step_size + random.randint(-2, 2)
            next_y = current_y + dy_norm * step_size + random.randint(-2, 2)
        
        # Keep within bounds
        next_x = max(0, min(width - 1, int(next_x)))
        next_y = max(0, min(height - 1, int(next_y)))
        
        # Ensure we're making progress - if next point is the same as current, force movement
        if next_x == current_x and next_y == current_y and distance > step_size:
            next_x = current_x + (1 if dx > 0 else -1)
            next_y = current_y + (1 if dy > 0 else -1)
            next_x = max(0, min(width - 1, next_x))
            next_y = max(0, min(height - 1, next_y))
        
        crack_points.append((next_x, next_y))
        
        # Update bounding box
        x_min = min(x_min, next_x)
        y_min = min(y_min, next_y)
        x_max = max(x_max, next_x)
        y_max = max(y_max, next_y)
        
        current_x, current_y = next_x, next_y
        iteration_count += 1
        
        # Additional safety check - if we're not making progress, break
        if iteration_count > 10:
            distance_to_target = math.sqrt((current_x - target_x)**2 + (current_y - target_y)**2)
            if iteration_count % 50 == 0:  # Check every 50 iterations
                print(f"Warning: Crack generation taking long time. Distance to target: {distance_to_target:.2f}, Iteration: {iteration_count}")
            
            # Emergency break if distance isn't decreasing significantly
            if iteration_count > 500 and distance_to_target > step_size * 10:
                print(f"Emergency break: Breaking infinite loop at iteration {iteration_count}")
                break
    
    # Log if we hit the safety limit
    if iteration_count >= max_iterations:
        print(f"Warning: Hit maximum iterations ({max_iterations}) in crack generation")
    
    # Add the final point
    crack_points.append(end)
    x_min = min(x_min, end[0])
    y_min = min(y_min, end[1])
    x_max = max(x_max, end[0])
    y_max = max(y_max, end[1])
    
    # Draw the crack with reduced vibration
    for i in range(len(crack_points) - 1):
        # More consistent thickness
        thickness = random.randint(thickness_range[0], thickness_range[1])
        
        # Use darker color for mortar joint cracks
        crack_color = (random.randint(15, 25), random.randint(15, 25), random.randint(15, 25))
        
        cv2.line(image, crack_points[i], crack_points[i + 1], crack_color, thickness)
        
        # Reduce branching frequency for cleaner look
        if random.random() < 0.05:  # 5% chance of small branch (reduced from 10%)
            branch_length = random.randint(3, 8)  # Shorter branches
            angle = random.uniform(0, 2 * math.pi)
            branch_end = (
                int(crack_points[i][0] + math.cos(angle) * branch_length),
                int(crack_points[i][1] + math.sin(angle) * branch_length)
            )
            # Keep branch within bounds
            branch_end = (
                max(0, min(width - 1, branch_end[0])),
                max(0, min(height - 1, branch_end[1]))
            )
            cv2.line(image, crack_points[i], branch_end, crack_color, max(1, thickness - 1))
    
    # Ensure minimum bounding box size
    if x_max - x_min < 20:
        x_min = max(0, x_min - 10)
        x_max = min(width - 1, x_max + 10)
    if y_max - y_min < 20:
        y_min = max(0, y_min - 10)
        y_max = min(height - 1, y_max + 10)
    
    return crack_points, [x_min, y_min, x_max, y_max]


def generate_mixed_crack_pattern(image, crack_type, mortar_probability=0.9):
    """
    Generate a crack that's a mixture of following mortar joints and going through bricks.
    All crack types now follow mortar joint patterns like the step crack shown in the image.
    
    Args:
        image (numpy.ndarray): Image to draw crack on
        crack_type (str): Type of crack (vertical, horizontal, diagonal, etc.)
        mortar_probability (float): Probability of following mortar joints vs going through bricks
        
    Returns:
        tuple: (crack_points, bounding_box, actual_crack_type)
    """
    height, width = image.shape[:2]
    
    # Detect mortar joints
    horizontal_joints, vertical_joints, joint_mask = detect_mortar_joints(image)
    
    # For all crack types, we'll create step-like patterns that follow mortar joints
    # This matches the excellent pattern shown in the reference image
    
    if crack_type == "vertical":
        # Vertical cracks will step down following mortar joints
        return generate_stepped_vertical_crack(image, horizontal_joints, vertical_joints)
        
    elif crack_type == "horizontal":
        # Horizontal cracks will step across following mortar joints  
        return generate_stepped_horizontal_crack(image, horizontal_joints, vertical_joints)
        
    elif crack_type == "diagonal":
        # Diagonal cracks will follow a diagonal step pattern
        return generate_stepped_diagonal_crack(image, horizontal_joints, vertical_joints)
        
    elif crack_type == "step":
        # For step cracks, use the perfect pattern from the reference image
        return generate_step_crack_with_mortar(image, horizontal_joints, vertical_joints)
        
    else:
        # For other crack types, create a mixed step pattern
        return generate_step_crack_with_mortar(image, horizontal_joints, vertical_joints)


def generate_stepped_vertical_crack(image, horizontal_joints, vertical_joints):
    """
    Generate a vertical crack that steps down following mortar joints.
    """
    height, width = image.shape[:2]
    crack_points = []
    
    # Start from top, choose a vertical joint or create one
    if vertical_joints:
        start_x = random.choice(vertical_joints[len(vertical_joints)//4:3*len(vertical_joints)//4])
    else:
        start_x = random.randint(width // 4, 3 * width // 4)
    
    start_y = random.randint(0, height // 8)
    current_x, current_y = start_x, start_y
    crack_points.append((current_x, current_y))
    
    # Initialize bounding box
    x_min, y_min = current_x, current_y
    x_max, y_max = current_x, current_y
    
    # Create stepping pattern downward
    target_y = height - random.randint(0, height // 8)
    num_steps = random.randint(4, 7)  # Similar to reference image
    
    for step in range(num_steps):
        if step < num_steps - 1:
            # Find next horizontal joint to step to
            available_joints = [j for j in horizontal_joints if j > current_y and j < target_y]
            if available_joints:
                next_y = random.choice(available_joints[:len(available_joints)//2 + 1])
            else:
                next_y = current_y + random.randint(30, 60)
            
            # Vertical segment along mortar joint
            v_points, v_bbox = generate_mortar_following_crack(
                image, (current_x, current_y), (current_x, next_y),
                horizontal_joints, vertical_joints, joint_follow_probability=0.95
            )
            crack_points.extend(v_points[1:])
            
            # Update bounding box
            x_min = min(x_min, v_bbox[0])
            y_min = min(y_min, v_bbox[1])
            x_max = max(x_max, v_bbox[2])
            y_max = max(y_max, v_bbox[3])
            
            current_y = next_y
            
            # Horizontal segment along mortar joint
            step_distance = random.randint(20, 50)
            direction = random.choice([-1, 1])
            next_x = current_x + (direction * step_distance)
            next_x = max(20, min(width - 20, next_x))
            
            h_points, h_bbox = generate_mortar_following_crack(
                image, (current_x, current_y), (next_x, current_y),
                horizontal_joints, vertical_joints, joint_follow_probability=0.95
            )
            crack_points.extend(h_points[1:])
            
            # Update bounding box
            x_min = min(x_min, h_bbox[0])
            y_min = min(y_min, h_bbox[1])
            x_max = max(x_max, h_bbox[2])
            y_max = max(y_max, h_bbox[3])
            
            current_x = next_x
    
    # Final vertical segment to complete the crack
    if current_y < target_y:
        final_points, final_bbox = generate_mortar_following_crack(
            image, (current_x, current_y), (current_x, target_y),
            horizontal_joints, vertical_joints, joint_follow_probability=0.95
        )
        crack_points.extend(final_points[1:])
        
        x_min = min(x_min, final_bbox[0])
        y_min = min(y_min, final_bbox[1])
        x_max = max(x_max, final_bbox[2])
        y_max = max(y_max, final_bbox[3])
    
    return crack_points, [x_min, y_min, x_max, y_max], "vertical"


def generate_stepped_horizontal_crack(image, horizontal_joints, vertical_joints):
    """
    Generate a horizontal crack that steps across following mortar joints.
    """
    height, width = image.shape[:2]
    crack_points = []

    # Start from left, choose a horizontal joint
    if horizontal_joints:
        start_y = random.choice(horizontal_joints[len(horizontal_joints)//4:3*len(horizontal_joints)//4])
    else:
        start_y = height // 2  # Default to center if no joints detected

    start_x = random.randint(0, width // 8)
    current_x, current_y = start_x, start_y
    crack_points.append((current_x, current_y))

    # Initialize bounding box
    x_min, y_min = current_x, current_y
    x_max, y_max = current_x, current_y

    # Create stepping pattern across
    target_x = width - random.randint(0, width // 8)
    num_steps = random.randint(4, 6)

    for step in range(num_steps):
        if step < num_steps - 1:
            # Horizontal segment strictly along mortar joint
            step_distance = random.randint(40, 80)
            next_x = min(target_x, current_x + step_distance)

            if horizontal_joints:
                next_y = current_y  # Stay on the same joint
            else:
                next_y = current_y  # Default to current position if no joints

            h_points, h_bbox = generate_mortar_following_crack(
                image, (current_x, current_y), (next_x, next_y),
                horizontal_joints, vertical_joints, joint_follow_probability=1.0
            )
            crack_points.extend(h_points[1:])

            # Update bounding box
            x_min = min(x_min, h_bbox[0])
            y_min = min(y_min, h_bbox[1])
            x_max = max(x_max, h_bbox[2])
            y_max = max(y_max, h_bbox[3])

            current_x = next_x

        # Vertical step with reduced randomness
        if step < num_steps - 1:
            step_distance = random.randint(10, 20)  # Reduced randomness
            direction = random.choice([-1, 1])
            next_y = current_y + (direction * step_distance)
            next_y = max(20, min(height - 20, next_y))

            v_points, v_bbox = generate_mortar_following_crack(
                image, (current_x, current_y), (current_x, next_y),
                horizontal_joints, vertical_joints, joint_follow_probability=1.0
            )
            crack_points.extend(v_points[1:])

            # Update bounding box
            x_min = min(x_min, v_bbox[0])
            y_min = min(y_min, v_bbox[1])
            x_max = max(x_max, v_bbox[2])
            y_max = max(y_max, v_bbox[3])

            current_y = next_y

    bbox = (x_min, y_min, x_max, y_max)
    return crack_points, bbox, "horizontal"


def generate_stepped_diagonal_crack(image, horizontal_joints, vertical_joints):
    """
    Generate a diagonal crack that follows mortar joints in a stepped pattern.
    """
    height, width = image.shape[:2]
    crack_points = []
    
    # Choose diagonal direction
    if random.choice([True, False]):
        # Top-left to bottom-right
        start = (random.randint(0, width // 3), random.randint(0, height // 3))
        end = (random.randint(2 * width // 3, width), random.randint(2 * height // 3, height))
    else:
        # Top-right to bottom-left  
        start = (random.randint(2 * width // 3, width), random.randint(0, height // 3))
        end = (random.randint(0, width // 3), random.randint(2 * height // 3, height))
    
    current_x, current_y = start
    target_x, target_y = end
    crack_points.append((current_x, current_y))
    
    # Initialize bounding box
    x_min, y_min = current_x, current_y
    x_max, y_max = current_x, current_y
    
    # Create diagonal stepping pattern
    num_steps = random.randint(5, 8)
    
    for step in range(num_steps):
        if step < num_steps - 1:
            # Calculate intermediate target
            progress = (step + 1) / num_steps
            intermediate_x = int(current_x + (target_x - current_x) * progress * random.uniform(0.8, 1.2))
            intermediate_y = int(current_y + (target_y - current_y) * progress * random.uniform(0.8, 1.2))
            
            # Constrain to image bounds
            intermediate_x = max(0, min(width - 1, intermediate_x))
            intermediate_y = max(0, min(height - 1, intermediate_y))
            
            # Generate crack segment following mortar joints
            seg_points, seg_bbox = generate_mortar_following_crack(
                image, (current_x, current_y), (intermediate_x, intermediate_y),
                horizontal_joints, vertical_joints, joint_follow_probability=0.9
            )
            crack_points.extend(seg_points[1:])
            
            # Update bounding box
            x_min = min(x_min, seg_bbox[0])
            y_min = min(y_min, seg_bbox[1])
            x_max = max(x_max, seg_bbox[2])
            y_max = max(y_max, seg_bbox[3])
            
            current_x, current_y = intermediate_x, intermediate_y
    
    # Final segment to target
    final_points, final_bbox = generate_mortar_following_crack(
        image, (current_x, current_y), (target_x, target_y),
        horizontal_joints, vertical_joints, joint_follow_probability=0.9
    )
    crack_points.extend(final_points[1:])
    
    x_min = min(x_min, final_bbox[0])
    y_min = min(y_min, final_bbox[1])
    x_max = max(x_max, final_bbox[2])
    y_max = max(y_max, final_bbox[3])
    
    return crack_points, [x_min, y_min, x_max, y_max], "diagonal"


def generate_step_crack_with_mortar(image, horizontal_joints, vertical_joints):
    """
    Generate a step crack that follows mortar joints exactly like the reference image.
    This creates the perfect step pattern with controlled length and minimal vibration.
    
    Args:
        image (numpy.ndarray): Image to draw crack on
        horizontal_joints (list): Y-coordinates of horizontal joints
        vertical_joints (list): X-coordinates of vertical joints
        
    Returns:
        tuple: (crack_points, bounding_box, crack_type)
    """
    height, width = image.shape[:2]
    crack_points = []
    
    # Start from upper portion, choose a good starting point
    if horizontal_joints:
        # Start from a horizontal joint in the upper third
        upper_joints = [j for j in horizontal_joints if j < height // 2]
        if upper_joints:
            start_y = random.choice(upper_joints)
        else:
            start_y = random.randint(height // 4, height // 2)
    else:
        start_y = random.randint(height // 4, height // 2)
    
    start_x = random.randint(width // 6, width // 3)  # Start from left side
    current_x, current_y = start_x, start_y
    crack_points.append((current_x, current_y))
    
    # Initialize bounding box
    x_min, y_min = current_x, current_y
    x_max, y_max = current_x, current_y
    
    # Create the perfect step pattern from the reference image
    # The reference shows 4-5 clear steps with good proportions
    num_steps = random.randint(4, 5)  # Match reference image
    
    # Calculate step sizes to achieve good proportions like the reference
    total_width_coverage = random.randint(width // 3, 2 * width // 3)  # Cover 1/3 to 2/3 of width
    total_height_coverage = random.randint(height // 4, height // 3)   # Cover 1/4 to 1/3 of height
    
    horizontal_step_size = total_width_coverage // num_steps
    vertical_step_size = total_height_coverage // num_steps
    
    for step in range(num_steps):
        # Horizontal segment along mortar joint - this is the key feature
        target_x = current_x + horizontal_step_size + random.randint(-10, 10)
        target_x = min(width - 20, target_x)  # Keep within bounds
        
        # Try to follow the current horizontal joint
        if horizontal_joints:
            # Find the closest horizontal joint to current position
            closest_h_joint = min(horizontal_joints, key=lambda j: abs(j - current_y))
            if abs(closest_h_joint - current_y) < 20:  # If close enough
                current_y = closest_h_joint  # Snap to joint
        
        # Generate horizontal segment with minimal vibration
        h_points, h_bbox = generate_mortar_following_crack(
            image, (current_x, current_y), (target_x, current_y),
            horizontal_joints, vertical_joints, joint_follow_probability=0.98  # Very high probability
        )
        crack_points.extend(h_points[1:])  # Skip first point to avoid duplication
        
        # Update bounding box
        x_min = min(x_min, h_bbox[0])
        y_min = min(y_min, h_bbox[1])
        x_max = max(x_max, h_bbox[2])
        y_max = max(y_max, h_bbox[3])
        
        current_x = target_x
        
        # Vertical segment down to next level (except for last step)
        if step < num_steps - 1:
            target_y = current_y + vertical_step_size + random.randint(-5, 5)
            target_y = min(height - 20, target_y)
            
            # Try to find a suitable horizontal joint to step down to
            if horizontal_joints:
                suitable_joints = [j for j in horizontal_joints if j > current_y and j <= target_y + 20]
                if suitable_joints:
                    target_y = min(suitable_joints)  # Take the first available joint
            
            # Try to follow a vertical joint if available
            if vertical_joints:
                closest_v_joint = min(vertical_joints, key=lambda j: abs(j - current_x))
                if abs(closest_v_joint - current_x) < 25:  # If close enough
                    current_x = closest_v_joint  # Snap to vertical joint
            
            # Generate vertical segment with minimal vibration
            v_points, v_bbox = generate_mortar_following_crack(
                image, (current_x, current_y), (current_x, target_y),
                horizontal_joints, vertical_joints, joint_follow_probability=0.98  # Very high probability
            )
            crack_points.extend(v_points[1:])  # Skip first point to avoid duplication
            
            # Update bounding box
            x_min = min(x_min, v_bbox[0])
            y_min = min(y_min, v_bbox[1])
            x_max = max(x_max, v_bbox[2])
            y_max = max(y_max, v_bbox[3])
            
            current_y = target_y
    
    return crack_points, [x_min, y_min, x_max, y_max], "step"
