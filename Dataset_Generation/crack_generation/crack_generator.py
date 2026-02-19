"""
Core crack generation methods using procedural drawing techniques.
"""

import cv2
import numpy as np
import random
import math
from scipy.interpolate import splprep, splev
from mortar_joint_detection import detect_mortar_joints, generate_mixed_crack_pattern

def generate_jagged_crack(image, start, end, thickness_range=(3, 6), branch_prob=0.2, jaggedness=4):
    """
    Generates a jagged, natural-looking crack with branches and reduced vibration.
    
    Args:
        image (numpy.ndarray): Image to draw the crack on
        start (tuple): Starting (x, y) coordinates
        end (tuple): Ending (x, y) coordinates
        thickness_range (tuple): Min and max thickness of the crack
        branch_prob (float): Probability of generating branches (0-1) - reduced for cleaner look
        jaggedness (int): How jagged the crack will be (0=straight, higher=more jagged) - reduced for smoother cracks
    
    Returns:
        list: List of points defining the crack path
        list: Bounding box coordinates [x_min, y_min, x_max, y_max]
    """
    height, width = image.shape[:2]
    crack_points = [start]
    
    # For bounding box calculation
    x_min, y_min = start
    x_max, y_max = start

    num_steps = max(50, int(np.sqrt((end[0] - start[0])**2 + (end[1] - start[1])**2) / 5))
    x_step = (end[0] - start[0]) / num_steps
    y_step = (end[1] - start[1]) / num_steps
    
    branch_points = []

    for i in range(1, num_steps):
        # Reduced randomness for smoother appearance
        x = int(start[0] + i * x_step + random.randint(-jaggedness, jaggedness))
        y = int(start[1] + i * y_step + random.randint(-jaggedness, jaggedness))
        
        # Keep points within image boundaries
        x = max(0, min(width - 1, x))
        y = max(0, min(height - 1, y))
        
        crack_points.append((x, y))
        
        # Update bounding box
        x_min = min(x_min, x)
        y_min = min(y_min, y)
        x_max = max(x_max, x)
        y_max = max(y_max, y)

        # Reduced branching frequency for cleaner look
        if random.random() < branch_prob:
            branch_length = random.randint(10, 20)  # Shorter branches
            angle = random.uniform(0, 2 * math.pi)
            bx = int(x + math.cos(angle) * branch_length)
            by = int(y + math.sin(angle) * branch_length)
            
            # Keep branch points within image boundaries
            bx = max(0, min(width - 1, bx))
            by = max(0, min(height - 1, by))
            
            branch_points.append((x, y, bx, by))
            
            # Update bounding box with branch points
            x_min = min(x_min, bx)
            y_min = min(y_min, by)
            x_max = max(x_max, bx)
            y_max = max(y_max, by)
            
            # Draw the branch with reduced thickness
            cv2.line(image, (x, y), (bx, by), (random.randint(20, 40),) * 3, 1)

    # Draw main crack with varying thickness
    for i in range(len(crack_points) - 1):
        thickness = max(2, thickness_range[0] + (thickness_range[1] - thickness_range[0]) * (1 - i / num_steps))
        crack_color = (random.randint(5, 15), random.randint(5, 15), random.randint(5, 15))  # Darker, more consistent color
        cv2.line(image, crack_points[i], crack_points[i + 1], crack_color, int(thickness))
    
    # Ensure bounding box has some minimum size
    if x_max - x_min < 10:
        x_min = max(0, x_min - 5)
        x_max = min(width - 1, x_max + 5)
    if y_max - y_min < 10:
        y_min = max(0, y_min - 5)
        y_max = min(height - 1, y_max + 5)
        
    return crack_points, [x_min, y_min, x_max, y_max]

def generate_bezier_crack(image, start, end, control_points=2, thickness_range=(2, 6)):
    """
    Generates a smooth crack using Bezier curves with multiple control points.
    
    Args:
        image (numpy.ndarray): Image to draw the crack on
        start (tuple): Starting (x, y) coordinates
        end (tuple): Ending (x, y) coordinates
        control_points (int): Number of control points for the curve
        thickness_range (tuple): Min and max thickness of the crack
    
    Returns:
        list: List of points defining the crack path
        list: Bounding box coordinates [x_min, y_min, x_max, y_max]
    """
    height, width = image.shape[:2]
    
    # Generate random control points between start and end
    points = [start]
    
    # Add control points
    for i in range(control_points):
        x = random.randint(
            min(start[0], end[0]) + int(i * abs(end[0] - start[0]) / (control_points + 1)),
            max(start[0], end[0]) - int((control_points - i) * abs(end[0] - start[0]) / (control_points + 1))
        )
        y = random.randint(
            min(start[1], end[1]) + int(i * abs(end[1] - start[1]) / (control_points + 1)),
            max(start[1], end[1]) - int((control_points - i) * abs(end[1] - start[1]) / (control_points + 1))
        )
        # Add some randomness to control points
        x = x + random.randint(-width//10, width//10)
        y = y + random.randint(-height//10, height//10)
        points.append((x, y))
    
    points.append(end)
    
    # Convert points to numpy array
    points = np.array(points)
    
    # Fit a spline to the points
    tck, u = splprep([points[:, 0], points[:, 1]], s=0, k=min(3, len(points)-1))
    
    # Evaluate the spline at more points for a smoother curve
    u_new = np.linspace(0, 1, 100)
    crack_points = np.column_stack(splev(u_new, tck))
    
    # Convert to integer coordinates
    crack_points = crack_points.astype(int)
    
    # Find bounding box
    x_min = np.min(crack_points[:, 0])
    y_min = np.min(crack_points[:, 1])
    x_max = np.max(crack_points[:, 0])
    y_max = np.max(crack_points[:, 1])
    
    # Draw the crack with varying thickness
    for i in range(len(crack_points) - 1):
        progress = i / (len(crack_points) - 1)
        thickness = int(thickness_range[0] + (thickness_range[1] - thickness_range[0]) * (1 - progress))
        point1 = (crack_points[i][0], crack_points[i][1])
        point2 = (crack_points[i+1][0], crack_points[i+1][1])
        cv2.line(image, point1, point2, (0, 0, 0), thickness)

    return crack_points, [x_min, y_min, x_max, y_max]

def generate_step_crack(image, direction='left_to_right'):
    """
    Generates a step-like crack with horizontal and vertical segments.
    
    Args:
        image (numpy.ndarray): Image to draw the crack on
        direction (str): 'left_to_right' or 'right_to_left'
    
    Returns:
        list: List of points defining the crack path
        list: Bounding box coordinates [x_min, y_min, x_max, y_max]
    """
    height, width = image.shape[:2]
    
    # Scale step size according to image dimensions
    base_step_size = min(width, height) // 10
    step_size_variation = base_step_size // 10
    
    if direction == 'left_to_right':
        # Start from left side
        x = random.randint(width // 8, width // 3)
        y = random.randint(height // 4, 3 * height // 4)
        horizontal_direction = 1  # Move right
    else:  # right_to_left
        # Start from right side
        x = random.randint(2 * width // 3, 7 * width // 8)
        y = random.randint(height // 4, 3 * height // 4)
        horizontal_direction = -1  # Move left
    
    # Scale line thickness according to image size
    min_thickness = max(1, min(width, height) // 200)
    max_thickness = max(2, min(width, height) // 100)
    
    # Track points for bounding box and path
    crack_points = [(x, y)]
    x_min, y_min = x, y
    x_max, y_max = x, y
    
    # Dynamically determine number of steps based on image size
    num_steps = random.randint(3, max(4, min(width, height) // base_step_size))
    
    for _ in range(num_steps):
        # Calculate next position ensuring we stay within image boundaries
        step = random.randint(base_step_size - step_size_variation, base_step_size + step_size_variation)
        
        # Horizontal segment
        potential_x_end = x + (horizontal_direction * step)
        # Ensure x_end stays within image boundaries
        if horizontal_direction > 0:  # Moving right
            x_end = min(potential_x_end, width - 1)
        else:  # Moving left
            x_end = max(potential_x_end, 0)
            
        # Stop if we've reached the edge
        if x_end == x:
            break
            
        cv2.line(image, (x, y), (x_end, y), (0, 0, 0), random.randint(min_thickness, max_thickness))
        x = x_end
        crack_points.append((x, y))
        
        # Update bounding box
        x_min = min(x_min, x)
        x_max = max(x_max, x)
        y_min = min(y_min, y)
        y_max = max(y_max, y)

        # Vertical segment (always goes down)
        potential_y_end = y + step
        # Ensure y_end stays within image boundaries
        y_end = min(potential_y_end, height - 1)
        
        # Stop if we've reached the bottom
        if y_end == y:
            break
            
        cv2.line(image, (x, y), (x, y_end), (0, 0, 0), random.randint(min_thickness, max_thickness))
        y = y_end
        crack_points.append((x, y))
        
        # Update bounding box
        y_min = min(y_min, y)
        y_max = max(y_max, y)
        
        # Stop if we've reached the bottom of the image
        if y >= height - step:
            break
    
    return crack_points, [x_min, y_min, x_max, y_max]

def generate_radial_crack(image):
    """
    Generates impact-like radial cracks from a central point.
    
    Args:
        image (numpy.ndarray): Image to draw the crack on
    
    Returns:
        list: List of points defining the main crack paths
        list: Bounding box coordinates [x_min, y_min, x_max, y_max]
    """
    height, width = image.shape[:2]
    center_x, center_y = random.randint(width//3, 2*width//3), random.randint(height//3, 2*height//3)
    
    all_points = []
    x_min, y_min = center_x, center_y
    x_max, y_max = center_x, center_y

    for _ in range(random.randint(5, 10)):  # 5-10 radial cracks
        angle = random.uniform(0, 2*np.pi)
        length = random.randint(40, 100)
        end_x = int(center_x + length * np.cos(angle))
        end_y = int(center_y + length * np.sin(angle))
        
        # Keep end points within image boundaries
        end_x = max(0, min(width - 1, end_x))
        end_y = max(0, min(height - 1, end_y))
        
        crack_points, bbox = generate_jagged_crack(
            image, 
            (center_x, center_y), 
            (end_x, end_y), 
            thickness_range=(3, 6)
        )
        
        all_points.extend(crack_points)
        
        # Update global bounding box
        x_min = min(x_min, bbox[0])
        y_min = min(y_min, bbox[1])
        x_max = max(x_max, bbox[2])
        y_max = max(y_max, bbox[3])

    return all_points, [x_min, y_min, x_max, y_max]

def generate_x_crack(image):
    """
    Generates an 'X' type crack failure pattern to simulate cyclic loading failure.
    
    Args:
        image (numpy.ndarray): Image to draw the crack on
    
    Returns:
        list: List of points defining the main crack paths
        list: Bounding box coordinates [x_min, y_min, x_max, y_max]
    """
    height, width = image.shape[:2]
    center_x, center_y = width // 2, height // 2  # Center point of the image

    # Define diagonal start and end points for both cracks
    diag1_start = (random.randint(0, width // 3), random.randint(0, height // 3))
    diag1_end = (random.randint(2 * width // 3, width), random.randint(2 * height // 3, height))

    diag2_start = (random.randint(2 * width // 3, width), random.randint(0, height // 3))
    diag2_end = (random.randint(0, width // 3), random.randint(2 * height // 3, height))

    # Generate two diagonal jagged cracks crossing each other
    crack_points1, bbox1 = generate_jagged_crack(image, diag1_start, diag1_end, thickness_range=(4, 10))
    crack_points2, bbox2 = generate_jagged_crack(image, diag2_start, diag2_end, thickness_range=(4, 10))
    
    all_points = crack_points1 + crack_points2
    
    # Combine bounding boxes
    x_min = min(bbox1[0], bbox2[0])
    y_min = min(bbox1[1], bbox2[1])
    x_max = max(bbox1[2], bbox2[2])
    y_max = max(bbox1[3], bbox2[3])

    # Optional: Add minor branching cracks near the center
    for _ in range(random.randint(2, 4)):
        bx = center_x + random.randint(-30, 30)
        by = center_y + random.randint(-30, 30)
        
        # Keep points within image boundaries
        bx = max(0, min(width - 1, bx))
        by = max(0, min(height - 1, by))
        
        cv2.line(image, (center_x, center_y), (bx, by), (0, 0, 0), 2)

    return all_points, [x_min, y_min, x_max, y_max]

def add_realistic_crack(image, crack_type="random", use_mortar_joints=True, mortar_probability=0.6):
    """
    Adds a realistic crack to the image with optional mortar joint following.
    
    Args:
        image (numpy.ndarray): Image to draw the crack on
        crack_type (str): Type of crack to generate
            - "vertical": Vertical crack
            - "horizontal": Horizontal crack
            - "diagonal": Diagonal crack
            - "step": Step crack
            - "x": X-shaped crack
            - "radial": Radial cracks from an impact point
            - "random": Random selection from above types
        use_mortar_joints (bool): Whether to detect and follow mortar joints
        mortar_probability (float): Probability of following mortar joints vs going through bricks
    
    Returns:
        tuple: (modified image, crack_type, bounding_box)
    """
    if crack_type == "random":
        crack_type = random.choice(["vertical", "horizontal", "diagonal", "step", "x", "radial"])
    
    height, width = image.shape[:2]
    
    # Use mortar joint detection for more realistic cracks
    if use_mortar_joints and crack_type in ["vertical", "horizontal", "diagonal", "step"]:
        try:
            crack_points, bbox, actual_crack_type = generate_mixed_crack_pattern(
                image, crack_type, mortar_probability
            )
            return image, actual_crack_type, bbox
        except Exception as e:
            print(f"Warning: Mortar joint detection failed, falling back to traditional method: {e}")
            # Fall back to traditional method
            pass
    
    # Traditional crack generation (fallback or for x/radial cracks)
    if crack_type == "vertical":
        x = random.randint(width // 4, 3 * width // 4)
        start, end = (x, 0), (x, height)
        crack_points, bbox = generate_jagged_crack(image, start, end)

    elif crack_type == "horizontal":
        y = random.randint(height // 4, 3 * height // 4)
        start, end = (0, y), (width, y)
        crack_points, bbox = generate_jagged_crack(image, start, end)

    elif crack_type == "diagonal":
        if random.choice([True, False]):
            start, end = (random.randint(0, width // 2), 0), (random.randint(width // 2, width), height)
        else:
            start, end = (random.randint(width // 2, width), 0), (random.randint(0, width // 2), height)
        crack_points, bbox = generate_jagged_crack(image, start, end)
        
    elif crack_type == "step":
        direction = random.choice(["left_to_right", "right_to_left"])
        crack_points, bbox = generate_step_crack(image, direction)

    elif crack_type == "x":
        crack_points, bbox = generate_x_crack(image)
        
    elif crack_type == "radial":
        crack_points, bbox = generate_radial_crack(image)
    
    else:
        raise ValueError(f"Unknown crack type: {crack_type}")

    return image, crack_type, bbox

def add_blur_and_noise(image, blur_sigma=0.5, noise_scale=5):
    """
    Adds slight blur and noise to make the crack blend naturally with the texture.
    
    Args:
        image (numpy.ndarray): Image to process
        blur_sigma (float): Sigma value for Gaussian blur
        noise_scale (int): Scale of the noise to add
    
    Returns:
        numpy.ndarray: Processed image
    """
    image = cv2.GaussianBlur(image, (3, 3), blur_sigma)  # Smooth edges
    noise = np.random.normal(0, noise_scale, image.shape).astype(np.uint8)
    image = cv2.addWeighted(image, 0.95, noise, 0.05, 0)
    return image
