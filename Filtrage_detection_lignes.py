import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the image
image = cv2.imread('road.jpeg')

# Convert to grayscale and apply Gaussian blur
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (7, 7), 0)

# Apply thresholding
_, binary = cv2.threshold(blurred, 145, 255, cv2.THRESH_BINARY)

# Detect lines using Hough Line Transform
lines = cv2.HoughLinesP(binary, 1, np.pi / 180, 50, minLineLength=80, maxLineGap=20)

# Plot the original image with detected lines
plt.figure(figsize=(10, 5))

# Subplot 1: Original image with detected lines
plt.imshow(image)
plt.axis('off')

if lines is not None:
    for line in lines:
        x1, y1, x2, y2 = line[0]
        plt.plot([x1, x2], [y1, y2], 'r')

