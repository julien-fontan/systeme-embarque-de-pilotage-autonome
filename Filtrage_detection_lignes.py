import cv2
import numpy as np
import math
import matplotlib.pyplot as plt

# Cette fonction détecte les bords de ligne. Faut que les lignes soient bien définies.
def canny(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (7, 7), 0)    # réduire le bruit
    _, binary = cv2.threshold(blur, 145, 255, cv2.THRESH_BINARY)
    canny = cv2.Canny(binary, 50, 150)  # canny applique son propre flou ? besoin de blur ?
    return canny

# Cette fonction affiche les lignes détectées sur fond noir
def display_lines(image, lines):
    line_image=np.zeros_like(image)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line.reshape(4)
            cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 10)   # 10 : épaisseur de la ligne
    return line_image

# Masque pour n'afficher que la route en face de nous (trapèze)
def region_of_interest(image, vertices):
    height = image.shape[0]
    polygons = np.array([[(400, height), (900, height), (650, 300), (550, 300)]])
    mask=np.zeros_like(image)
    cv2.fillPoly(mask, polygons, 255)
    masked_image = cv2.bitwise_and(image, mask)
    return masked_image

def average_slope_intercept(image, lines):
    left_fit=[]
    right_fit=[]
    for line in lines :
        x1, y1, x2, y2 = line.reshape(4)
        parameters = np.polyfit((x1,x2), (y1,y2), 1)
        slope = parameters[0]
        intercept = parameters[1]
        if slope < 0 :
            left_fit.append((slope, intercept))
        else :
            right_fit.append((slope, intercept))
        left_fit_average = np.average(left_fit, axis='0')
            

image = cv2.imread('road.jpeg')
copy=np.copy(image)
treated_image = canny(copy)
cropped_image = region_of_interest(treated_image, vertices=None)
# lines = cv2.HoughLinesP(binary, 1, np.pi / 180, 50, minLineLength=80, maxLineGap=20)
# 1eme paramètre : taille quadrillage, si diminue, plus de lignes détectées mais moins précises et temps de calcul plus long
# 3eme paramètre : seuil de pts d'intersection devant être détectés pour qu'une ligne soit considérée comme valide
# 4eme paramètre : longueur minimale d'une ligne
# 5eme paramètre : distance maximale entre deux points pour qu'ils soient considérés comme un seul point
lines = cv2.HoughLinesP(cropped_image, 2, np.pi / 180, 50, minLineLength=100, maxLineGap=4)
line_image=display_lines(copy, lines)
combo_image=cv2.addWeighted(copy, 0.8, line_image, 1, 1)
cv2.imshow("de base", image)
cv2.imshow("result", combo_image)
cv2.waitKey(0)









'''
Deuxième solution (abandonnée) :
'''

if lines is not None:
    for line in lines:
        x1, y1, x2, y2 = line[0]
        plt.plot([x1, x2], [y1, y2], 'r')


def get_angle(x1, y1, x2, y2):
    return math.degrees(math.atan2(y2 - y1, x2 - x1))

def process_image(binary):
    # dst=binary
    # cdst=binary
    cdstP=np.copy(binary)
    
    # dst = cv2.Canny(binary, 50, 200, None, 3)
    # binary = cv2.cvtColor(dst, cv2.COLOR_GRAY2BGR)
    # cdstP = np.copy(cdst)

    # lines = cv2.HoughLines(dst, 1, np.pi / 180, 150, None, 0, 0)
    # if lines is not None:
    #     for i in range(len(lines)):
    #         rho, theta = lines[i][0]
    #         a, b = math.cos(theta), math.sin(theta)
    #         x0, y0 = a * rho, b * rho
    #         pt1 = (int(x0 + 1000 * (-b)), int(y0 + 1000 * (a)))
    #         pt2 = (int(x0 - 1000 * (-b)), int(y0 - 1000 * (a)))
    #         cv2.line(cdst, pt1, pt2, (0, 0, 255), 3, cv2.LINE_AA)

    linesP = cv2.HoughLinesP(binary, 1, np.pi / 180, 50, None, 100, 50)
    filtered_lines = []
    angle_thresh, dist_thresh = 5, 10

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

        for (x1, y1, x2, y2) in filtered_lines:
            cv2.line(binary, (x1, y1), (x2, y2), (0, 255, 0), 3, cv2.LINE_AA)

    # cv2.imshow("Lignes - Hough standard", cdst)
    cv2.imshow("Lignes filtrées - Hough probabiliste", cdstP)
    cv2.waitKey()

process_image(binary)
