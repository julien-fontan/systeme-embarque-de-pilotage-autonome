import cv2
import numpy as np
import math
import matplotlib.pyplot as plt

image = cv2.imread('road.jpeg')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (7, 7), 0)
_, binary = cv2.threshold(blurred, 145, 255, cv2.THRESH_BINARY)


lines = cv2.HoughLinesP(binary, 1, np.pi / 180, 50, minLineLength=80, maxLineGap=20)
plt.figure(figsize=(10, 5))
plt.imshow(image)
plt.axis('off')

if lines is not None:
    for line in lines:
        x1, y1, x2, y2 = line[0]
        plt.plot([x1, x2], [y1, y2], 'r')


def get_angle(x1, y1, x2, y2):
    return math.degrees(math.atan2(y2 - y1, x2 - x1))

def process_image(binary):
    dst = cv2.Canny(binary, 50, 200, None, 3)
    cdst = cv2.cvtColor(dst, cv2.COLOR_GRAY2BGR)
    cdstP = np.copy(cdst)

    lines = cv2.HoughLines(dst, 1, np.pi / 180, 150, None, 0, 0)
    if lines is not None:
        for i in range(len(lines)):
            rho, theta = lines[i][0]
            a, b = math.cos(theta), math.sin(theta)
            x0, y0 = a * rho, b * rho
            pt1 = (int(x0 + 1000 * (-b)), int(y0 + 1000 * (a)))
            pt2 = (int(x0 - 1000 * (-b)), int(y0 - 1000 * (a)))
            cv2.line(cdst, pt1, pt2, (0, 0, 255), 3, cv2.LINE_AA)

    linesP = cv2.HoughLinesP(dst, 1, np.pi / 180, 50, None, 100, 50)
    filtered_lines = []
    angle_thresh, dist_thresh = 5, 20

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
            cv2.line(cdstP, (x1, y1), (x2, y2), (0, 255, 0), 3, cv2.LINE_AA)

    cv2.imshow("Lignes - Hough standard", cdst)
    cv2.imshow("Lignes filtrées - Hough probabiliste", cdstP)
    cv2.waitKey()

process_image(binary)
