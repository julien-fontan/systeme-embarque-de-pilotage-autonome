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
    # cv2.imshow("Canny", canny)
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
def region_of_interest(image):
    height, width = image.shape[0:2]
    polygons = np.array([[
        (0, height),
        (width, height),
        (width, height * 1/2),
        (width * 3/5, 0),
        (width * 2/5, 0),
        (0, height * 1/2)]], dtype=np.int32)
    mask=np.zeros_like(image)
    cv2.fillPoly(mask, polygons, 255)
    masked_image = cv2.bitwise_and(image, mask)
    # cv2.imshow("Masque", masked_image)
    return masked_image
    # return image

def make_coordinates(image, line_parameters):
    slope, intercept = line_parameters
    y1=image.shape[0]   # on place le début de la ligne en bas de l'image
    y2 = y1*(1/4)  # limite haute de la ligne
    x1 = (y1 - intercept)/slope # y = ax + b, donc x = (y - b)/a
    x2 = (y2 - intercept)/slope
    return np.array([x1, y1, x2, y2], dtype=np.int32)

# Marche pour une image avec deux lignes droites
def average_slope_intercept(image, lines):
    """
    Processes a set of detected lines to calculate the average slope and intercept
    for the left and right lane lines in an image. Used to consolidate multiple line segments into two 
    representative lines (one for the left lane and one for the right lane).

    Args:
        image (numpy.ndarray): The image on which the lines are detected. This is 
            used to determine the dimensions for generating the final line coordinates.
        lines (numpy.ndarray): An array of detected lines, where each line is 
            represented by its endpoints [x1, y1, x2, y2].

    Returns:
        numpy.ndarray: An array containing two lines, one for the left lane and one 
            for the right lane. Each line is represented by its coordinates 
            [x1, y1, x2, y2].

    Notes:
        - Lines with a negative slope are classified as part of the left lane, 
          while lines with a positive slope are classified as part of the right lane.
        - The function uses linear regression to calculate the average slope and 
          intercept for the left and right lane lines.
        - The `make_coordinates` function is assumed to generate the final line 
          coordinates based on the slope and intercept.
    """
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
    left_fit_average = np.average(left_fit, axis=0)
    right_fit_average = np.average(right_fit, axis=0)
    left_line = make_coordinates(image, left_fit_average)
    right_line = make_coordinates(image, right_fit_average)
    return np.array([left_line, right_line])

'''
Traitement sur une image
'''
# image = cv2.imread('road.jpeg')
# lane_image=np.copy(image)
# treated_image = canny(lane_image)
# cropped_image = region_of_interest(treated_image)
# # 1eme paramètre : taille quadrillage, si diminue, plus de lignes détectées mais moins précises et temps de calcul plus long
# # 3eme paramètre : seuil de pts d'intersection devant être détectés pour qu'une ligne soit considérée comme valide
# # 4eme paramètre : longueur minimale d'une ligne
# # 5eme paramètre : distance maximale entre deux points pour qu'ils soient considérés comme un seul point
# lines = cv2.HoughLinesP(cropped_image, 2, np.pi / 180, 50, minLineLength=100, maxLineGap=4)
# average_lines = average_slope_intercept(lane_image, lines)
# line_image=display_lines(lane_image, average_lines)
# combo_image=cv2.addWeighted(lane_image, 0.8, line_image, 1, 1)
# cv2.imshow("result", combo_image)
# cv2.waitKey(0)

'''
Traitement sur une vidéo
'''
cap = cv2.VideoCapture("road_test_video.mp4")
while(cap.isOpened()):
    _, frame = cap.read()
    canny_image = canny(frame)
    cropped_image = region_of_interest(canny_image)
    lines = cv2.HoughLinesP(cropped_image, 2, np.pi / 180, 50, minLineLength=50, maxLineGap=4)
    averaged_lines = average_slope_intercept(frame, lines)
    line_image = display_lines(frame, averaged_lines)
    combo_image = cv2.addWeighted(frame, 0.8, line_image, 1, 1)
    # Resize the frame to make the window smaller
    resized_combo_image = cv2.resize(combo_image, (1200, 600))  # Adjust width and height as needed
    cv2.imshow("result", resized_combo_image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
