import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the image
image_path = "./route.jpeg"
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

img = cv2.imread('route.jpeg')
img = cv2.resize(img, (img.shape[1]//3, img.shape[0]//3))
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)[1]

lines = cv2.HoughLinesP(gray, 1, np.pi/180, 50, minLineLength=80, maxLineGap=20)
lines = np.squeeze(lines)

cv2.imshow("Image", gray)
cv2.waitKey(0)



















# Invert the image to detect white lines on a black background
inverted_image = cv2.bitwise_not(image)

# Preprocess the image (e.g., edge detection)
edges = cv2.Canny(inverted_image, 50, 150, apertureSize=3)

# Perform Hough Line Transform
lines = cv2.HoughLines(edges, 1, np.pi / 180, 100)

# Plot the original image with detected lines
plt.figure(figsize=(10, 5))

# Subplot 1: Original image with detected lines
plt.subplot(1, 2, 1)
plt.imshow(image, cmap='gray')
plt.title('Detected Lines')
plt.axis('off')

if lines is not None:
    for rho, theta in lines[:, 0]:
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * (a))
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * (a))
        plt.plot([x1, x2], [y1, y2], 'r')

# Subplot 2: Hough space
plt.subplot(1, 2, 2)
plt.imshow(edges, cmap='gray')
plt.title('Edges')
plt.axis('off')

plt.show()