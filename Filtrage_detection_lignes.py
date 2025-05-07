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
            cv2.line(line_image, (x1, y1), (x2, y2), (0, 0, 255), 10)   # 10 : épaisseur de la ligne
    return line_image

def region_of_interest(image, top_width, bottom_width, trapezoid_height):
    """
    Applies a mask to keep only the region of interest defined by an isosceles trapezoid.
    """
    height, width = image.shape[:2]
    polygons = np.array([
        (int(width * (1 - bottom_width) / 2), height),
        (int(width * (1 + bottom_width) / 2), height),
        (int(width * (1 + top_width) / 2), int(height * (1 - trapezoid_height))),
        (int(width * (1 - top_width) / 2), int(height * (1 - trapezoid_height)))
    ], dtype=np.int32)
    mask = np.zeros_like(image)
    cv2.fillPoly(mask, [polygons], 255)
    return cv2.bitwise_and(image, mask)

def make_coordinates(image, line_parameters):
    slope, intercept = line_parameters
    y1=image.shape[0]   # on place le début de la ligne en bas de l'image
    y2 = y1*(2/3)  # limite haute de la ligne
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
    left_fit = []
    right_fit = []
    for line in lines:
        x1, y1, x2, y2 = line.reshape(4)
        parameters = np.polyfit((x1, x2), (y1, y2), 1)
        slope = parameters[0]
        intercept = parameters[1]
        if slope < 0:
            left_fit.append((slope, intercept))
        else:
            right_fit.append((slope, intercept))

    left_line = None
    right_line = None

    if left_fit:
        left_fit_average = np.average(left_fit, axis=0)
        left_line = make_coordinates(image, left_fit_average)
    if right_fit:
        right_fit_average = np.average(right_fit, axis=0)
        right_line = make_coordinates(image, right_fit_average)

    # Return only valid lines
    return np.array([line for line in [left_line, right_line] if line is not None])

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
Modification des différents paramètre intéractivement
'''

def adjust_canny_parameters(frame):

    def nothing(x):
        pass

    cv2.namedWindow("Canny")
    cv2.createTrackbar("Blur", "Canny", 7, 50, nothing)
    cv2.createTrackbar("Threshold", "Canny", 145, 255, nothing)
    cv2.createTrackbar("Canny Min", "Canny", 50, 255, nothing)
    cv2.createTrackbar("Canny Max", "Canny", 150, 255, nothing)

    while True:
        blur_kernel = cv2.getTrackbarPos("Blur", "Canny")
        threshold_value = cv2.getTrackbarPos("Threshold", "Canny")
        canny_min = cv2.getTrackbarPos("Canny Min", "Canny")
        canny_max = cv2.getTrackbarPos("Canny Max", "Canny")

        # Ensure blur kernel is odd
        blur_kernel = max(1, blur_kernel | 1)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (blur_kernel, blur_kernel), 0)
        _, binary = cv2.threshold(blur, threshold_value, 255, cv2.THRESH_BINARY)
        canny_image = cv2.Canny(binary, canny_min, canny_max)

        resized_canny_image = cv2.resize(canny_image, (1200, 600))
        cv2.imshow("Canny", resized_canny_image)

        if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to confirm and exit
            break

    cv2.destroyWindow("Canny")
    return blur_kernel, threshold_value, canny_min, canny_max

def adjust_trapezoid_parameters(frame):

    def nothing(x):
        pass

    cv2.namedWindow("Adjust Trapezoid")
    cv2.createTrackbar("Top Width (%)", "Adjust Trapezoid", 40, 100, nothing)
    cv2.createTrackbar("Bottom Width (%)", "Adjust Trapezoid", 100, 100, nothing)
    cv2.createTrackbar("Height (%)", "Adjust Trapezoid", 50, 100, nothing)

    height, width = frame.shape[:2]

    while True:
        top_width = cv2.getTrackbarPos("Top Width (%)", "Adjust Trapezoid") / 100
        bottom_width = cv2.getTrackbarPos("Bottom Width (%)", "Adjust Trapezoid") / 100
        trapezoid_height = cv2.getTrackbarPos("Height (%)", "Adjust Trapezoid") / 100

        polygons = np.array([
            (int(width * (1 - bottom_width) / 2), height),
            (int(width * (1 + bottom_width) / 2), height),
            (int(width * (1 + top_width) / 2), int(height * (1 - trapezoid_height))),
            (int(width * (1 - top_width) / 2), int(height * (1 - trapezoid_height)))
        ], dtype=np.int32)

        mask = np.zeros_like(frame)
        cv2.fillPoly(mask, [polygons], 255)
        masked_image = cv2.bitwise_and(frame, mask)

        resized_masked_image = cv2.resize(masked_image, (1200, 600))
        cv2.imshow("Adjust Trapezoid", resized_masked_image)

        if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to confirm and exit
            break

    cv2.destroyWindow("Adjust Trapezoid")
    return top_width, bottom_width, trapezoid_height

def adjust_hough_parameters(processed_image, original_frame):

    def nothing(x):
        pass

    cv2.namedWindow("Hough")
    cv2.createTrackbar("Min Length", "Hough", 50, 500, nothing)
    cv2.createTrackbar("Max Gap", "Hough", 4, 50, nothing)

    while True:
        min_line_length = cv2.getTrackbarPos("Min Length", "Hough")
        max_line_gap = cv2.getTrackbarPos("Max Gap", "Hough")

        lines = cv2.HoughLinesP(processed_image, 2, np.pi / 180, 50, minLineLength=min_line_length, maxLineGap=max_line_gap)
        averaged_lines = average_slope_intercept(original_frame, lines) if lines is not None else []
        line_image = display_lines(original_frame, averaged_lines)
        combo_image = cv2.addWeighted(original_frame, 0.8, line_image, 1, 1)

        resized_combo_image = cv2.resize(combo_image, (1200, 600))
        cv2.imshow("Hough", resized_combo_image)

        if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to confirm and exit
            break

    cv2.destroyWindow("Hough")
    return min_line_length, max_line_gap

cap = cv2.VideoCapture("road_test_video.mp4")
_, first_frame = cap.read()

# Step 1: Adjust Canny parameters
blur_kernel, threshold_value, canny_min, canny_max = adjust_canny_parameters(first_frame)

# Apply the adjusted Canny parameters to the first frame
gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (blur_kernel, blur_kernel), 0)
_, binary = cv2.threshold(blur, threshold_value, 255, cv2.THRESH_BINARY)
canny_image = cv2.Canny(binary, canny_min, canny_max)

# Step 2: Adjust trapezoid proportions
top_width, bottom_width, trapezoid_height = adjust_trapezoid_parameters(first_frame)
cropped_image = region_of_interest(canny_image, top_width, bottom_width, trapezoid_height)

# Step 3: Adjust HoughLinesP parameters
min_line_length, max_line_gap = adjust_hough_parameters(cropped_image, first_frame)

# Process the entire video with the adjusted parameters
while cap.isOpened():
    _, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (blur_kernel, blur_kernel), 0)
    _, binary = cv2.threshold(blur, threshold_value, 255, cv2.THRESH_BINARY)
    canny_image = cv2.Canny(binary, canny_min, canny_max)
    cropped_image = region_of_interest(canny_image, top_width, bottom_width, trapezoid_height)
    lines = cv2.HoughLinesP(cropped_image, 2, np.pi / 180, 50, minLineLength=min_line_length, maxLineGap=max_line_gap)
    averaged_lines = average_slope_intercept(frame, lines) if lines is not None else []
    line_image = display_lines(frame, averaged_lines)
    combo_image = cv2.addWeighted(frame, 0.8, line_image, 1, 1)
    resized_combo_image = cv2.resize(combo_image, (1200, 600))
    cv2.imshow("result", resized_combo_image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
