import cv2
import numpy as np

# Chargement de l'image avec cv2
image = cv2.imread(r"C:\Users\NICOLAS DODE\Documents\PRONTO\white_line_detected.png")
frame = cv2.resize(image, (640, 480))

# Transformation de perspective

# Sélection des nouvelles coordonnées
tl = (200, 250)  # Coin supérieur gauche
bl = (50, 470)   # Coin inférieur gauche
tr = (440, 250)  # Coin supérieur droit
br = (590, 470)  # Coin inférieur droit

# Affichage des points sur l'image originale
cv2.circle(frame, tl, 5, (0, 0, 255), -1)
cv2.circle(frame, bl, 5, (0, 0, 255), -1)
cv2.circle(frame, tr, 5, (0, 0, 255), -1)
cv2.circle(frame, br, 5, (0, 0, 255), -1)

# Transformation de perspective - Transformation géométrique
pts1 = np.float32([tl, bl, tr, br])
pts2 = np.float32([[0, 0], [0, 480], [640, 0], [640, 480]])

matrix = cv2.getPerspectiveTransform(pts1, pts2)
transformed_frame = cv2.warpPerspective(frame, matrix, (640, 480))

# Affichage de l'image d'origine et de l'image transformée
cv2.imshow("Image Originale", frame)
cv2.imshow("Image Transformée", transformed_frame)

cv2.waitKey(0)  # Attendre une touche pour fermer
cv2.destroyAllWindows()
