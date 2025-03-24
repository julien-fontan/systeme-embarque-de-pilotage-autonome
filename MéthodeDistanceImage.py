'''
'Principe Méthode des Perspectives :
Une ligne blanche au sol semble plus large et plus proche en bas de l’image et plus fine en haut à cause de la perspective.
En appliquant une transformation de perspective, on obtient une vue « de dessus », où les distances peuvent être mesurées directement.
Mise en œuvre avec OpenCV :
Capture de l’image

Utilisation de la caméra pour prendre une image en temps réel.
Détection de la ligne blanche

Conversion en niveaux de gris et seuillage pour extraire la ligne.
Filtrage avec Canny pour détecter les contours.
Détermination de la région d'intérêt (ROI)

Sélection des 4 points de la ligne dans l’image d'origine correspondant à un trapèze.
Définition d’un rectangle cible où la vue transformée apparaîtra.
Application de la transformation de perspective

Utilisation de cv2.getPerspectiveTransform() et cv2.warpPerspective().
Mesure de la distance

Une fois transformée, la distance de la ligne à la caméra peut être estimée à partir de la position de la ligne dans l’image transformée.
'''
import cv2
import numpy as np

# Charger l'image capturée
image = cv2.imread("route.jpg")

# Définir les 4 points d'un trapèze autour de la ligne blanche
pts_src = np.float32([[100, 400], [500, 400], [50, 500], [550, 500]])  # À ajuster selon votre image

# Définir les 4 points correspondants dans la vue transformée
width, height = 400, 600  # Taille de la nouvelle image (vue de dessus)
pts_dst = np.float32([[0, 0], [width, 0], [0, height], [width, height]])

# Calculer la matrice de transformation
M = cv2.getPerspectiveTransform(pts_src, pts_dst)

# Appliquer la transformation de perspective
warped = cv2.warpPerspective(image, M, (width, height))

# Affichage du résultat
cv2.imshow("Original", image)
cv2.imshow("IPM", warped)
cv2.waitKey(0)
cv2.destroyAllWindows()


'''
Principe: Approche par Stéréovision (si deux caméras sont disponibles):

Cette méthode repose sur l’utilisation de deux caméras espacées d’une distance fixe pour créer une vision en 3D et générer une carte de profondeur.

Principe :
Deux images sont prises simultanément avec des caméras alignées horizontalement.
Une disparité (différence de position de la ligne blanche dans les deux images) est calculée.
En appliquant la relation de triangulation, on peut obtenir la distance réelle.
La distance d d’un point est donnée par la formule :
d = (f*b)/(disparité)
f = focale
b = distance entre les deux caméras
,
disparité = différence de position du même point dans les deux images.
Mise en œuvre avec OpenCV :
Capture simultanée des images des deux caméras.
Correction de la distorsion et rectification stéréoscopique.
Détection de la ligne blanche dans chaque image.
Calcul de la disparité avec OpenCV.
Conversion de la disparité en distance.
'''


# Charger les images gauche et droite capturées par les caméras
left_img = cv2.imread("left.jpg", 0)  # Image de la caméra gauche
right_img = cv2.imread("right.jpg", 0)  # Image de la caméra droite

# Créer un objet de correspondance stéréo
stereo = cv2.StereoBM_create(numDisparities=16, blockSize=15)

# Calculer la carte de disparité
disparity = stereo.compute(left_img, right_img)

# Affichage de la carte de profondeur
cv2.imshow("Disparity", disparity)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Convertir la disparité en distance (simplifié)
focal_length = 700  # À ajuster
baseline = 0.1  # Distance entre les caméras en mètres

distance = (focal_length * baseline) / (disparity + 0.0001)  # Éviter la division par 0

print("Distance estimée :", distance)




''' 
Nous réaliserons la solution de double mise en perspective dans le cas 
ou les cameras n'ont pas une orientation semblables des objetcifs. 
Nous choisirons la méthode stéréovision dans le cas ou nos deux caméras 
sont alignés horizontalement et de meme axe d'observation.

'''
import cv2
import numpy as np

# Paramètres pour l'IPM (définis selon votre configuration de caméra)
src_pts = np.float32([[100, 300], [500, 300], [0, 500], [600, 500]])  # Points source
dst_pts = np.float32([[100, 100], [500, 100], [100, 500], [500, 500]])  # Points destination
M = cv2.getPerspectiveTransform(src_pts, dst_pts)  # Matrice de transformation

# Distance seuil pour déclencher l’action (ex. 1.5 mètres)
DISTANCE_SEUIL = 1.5  

# Paramètres de la caméra
pixels_per_meter = 50  # À calibrer selon votre caméra et hauteur

def detect_line_distance(frame):
    """ Détecte la ligne blanche et mesure la distance à la caméra """
    
    # Transformation IPM (vue de dessus)
    top_view = cv2.warpPerspective(frame, M, (frame.shape[1], frame.shape[0]))

    # Conversion en niveaux de gris + seuillage pour détecter la ligne
    gray = cv2.cvtColor(top_view, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)

    # Détection des contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        # Trouver le point le plus proche de la caméra (bas de l’image)
        closest_y = frame.shape[0]  # Initialisé avec une grande valeur
        for contour in contours:
            for point in contour:
                _, y = point[0]
                if y < closest_y:
                    closest_y = y

        # Convertir la distance pixel → mètres
        distance_m = (frame.shape[0] - closest_y) / pixels_per_meter
        return distance_m

    return None  # Si aucune ligne détectée

# Capture vidéo (remplacez par la caméra réelle)
cap = cv2.VideoCapture("video_route.mp4")  # Remplacez par cap = cv2.VideoCapture(0) pour une caméra réelle

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Mesurer la distance
    distance = detect_line_distance(frame)

    # Affichage et déclenchement de l’action
    if distance is not None:
        print(f"Distance estimée : {distance:.2f} m")
        if distance < DISTANCE_SEUIL:
            print("⚠️ ACTION DÉCLENCHÉE ⚠️")

    cv2.imshow("Frame", frame)

    if cv2.waitKey(30) & 0xFF == 27:  # Quitter avec ESC
        break

cap.release()
cv2.destroyAllWindows()

