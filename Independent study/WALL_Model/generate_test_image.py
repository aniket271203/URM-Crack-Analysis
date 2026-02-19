import cv2
import numpy as np

# Create a blank white wall (500x500)
img = np.ones((500, 500, 3), dtype=np.uint8) * 255

# Draw a "Window" (Black rectangle)
# at (150, 150) with width 200, height 150
cv2.rectangle(img, (150, 150), (350, 300), (0, 0, 0), -1) 

cv2.imwrite("test_wall.jpg", img)
print("Created test_wall.jpg")
