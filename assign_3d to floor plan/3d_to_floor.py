import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the 3D image
image = cv2.imread('3d-floor-plans.jpg')

# Convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply edge detection with adjusted thresholds
edges = cv2.Canny(gray, 100, 200, apertureSize=3)

# Find contours
contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Create a blank image for the floor plan
floor_plan = np.zeros_like(gray)

# Draw the contours on the floor plan with thicker lines
cv2.drawContours(floor_plan, contours, -1, (255, 255, 255), 2)

# Normalize the image to improve visibility
floor_plan = cv2.normalize(floor_plan, None, 0, 255, cv2.NORM_MINMAX)

# Display the original image and the floor plan
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.title('Original Image')
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

plt.subplot(1, 2, 2)
plt.title('Floor Plan')
plt.imshow(floor_plan, cmap='gray')

plt.show()

