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







"""
@file hough_lines_filtered.py
@brief Détection de lignes avec filtrage pour éviter les doublons proches (lignes épaisses)
"""
import sys
import math
import cv2 as cv
import numpy as np

def get_angle(x1, y1, x2, y2):
    return math.degrees(math.atan2(y2 - y1, x2 - x1))

def main(argv):
    default_file = 'sudoku.png'
    filename = argv[0] if len(argv) > 0 else default_file

    # Charger l'image en niveaux de gris
    src = cv.imread(cv.samples.findFile(filename), cv.IMREAD_GRAYSCALE)
    if src is None:
        print('Erreur lors du chargement de l’image !')
        print('Usage: hough_lines_filtered.py [nom_image -- par défaut : ' + default_file + '] \n')
        return -1

    # Détection des bords avec Canny
    dst = cv.Canny(src, 50, 200, None, 3)

    # Convertir en BGR pour affichage
    cdst = cv.cvtColor(dst, cv.COLOR_GRAY2BGR)
    cdstP = np.copy(cdst)

    # Détection standard (HoughLines)
    lines = cv.HoughLines(dst, 1, np.pi / 180, 150, None, 0, 0)
    if lines is not None:
        for i in range(0, len(lines)):
            rho = lines[i][0][0]
            theta = lines[i][0][1]
            a = math.cos(theta)
            b = math.sin(theta)
            x0 = a * rho
            y0 = b * rho
            pt1 = (int(x0 + 1000*(-b)), int(y0 + 1000*(a)))
            pt2 = (int(x0 - 1000*(-b)), int(y0 - 1000*(a)))
            cv.line(cdst, pt1, pt2, (0, 0, 255), 3, cv.LINE_AA)

    # Détection probabiliste (HoughLinesP)
    linesP = cv.HoughLinesP(dst, 1, np.pi / 180, 50, None, 100, 50)

    # Filtrage des lignes proches et similaires
    filtered_lines = []
    angle_thresh = 5  # écart angulaire max en degrés
    dist_thresh = 20  # distance max entre lignes pour considérer que c'est un doublon

    if linesP is not None:
        for i in range(len(linesP)):
            x1, y1, x2, y2 = linesP[i][0]
            angle1 = get_angle(x1, y1, x2, y2)
            keep = True
            for x3, y3, x4, y4 in filtered_lines:
                angle2 = get_angle(x3, y3, x4, y4)
                if abs(angle1 - angle2) < angle_thresh and \
                   abs(x1 - x3) < dist_thresh and abs(y1 - y3) < dist_thresh:
                    keep = False
                    break
            if keep:
                filtered_lines.append((x1, y1, x2, y2))

        # Affichage des lignes filtrées
        for (x1, y1, x2, y2) in filtered_lines:
            cv.line(cdstP, (x1, y1), (x2, y2), (0, 255, 0), 3, cv.LINE_AA)

    # Affichage des résultats
    cv.imshow("Image d’origine", src)
    cv.imshow("Lignes - Hough standard", cdst)
    cv.imshow("Lignes filtrées - Hough probabiliste", cdstP)

    cv.waitKey()
    return 0

if __name__ == "__main__":
    main(sys.argv[1:])
