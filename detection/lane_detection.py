import cv2
import os
import numpy as np
from parameter_adjuster import ParameterAdjuster
from collections import deque

class LaneDetection:
    def __init__(self, video_source, dual_camera=False, camera_side=None, parameters=None, use_vanishing_point=True):
        """Initialise la détection de lignes avec la source vidéo et les paramètres."""
        # Si video_source a une méthode get_frame, c'est un objet CameraStream
        self.video_source = video_source
        if hasattr(video_source, "get_frame"):
            self.cap = None  # On utilisera get_frame()
        else:
            self.cap = cv2.VideoCapture(video_source)   # si on utilise un fichier vidéo
        self.dual_camera = dual_camera
        self.camera_side = camera_side  # Only relevant for dual cameras
        parameters = parameters or {}  # Remplace None par un dictionnaire vide
        self.blur_kernel = parameters.get("blur_kernel", 7)
        self.threshold_value = parameters.get("threshold_value", 145)
        self.canny_min = parameters.get("canny_min", 50)
        self.canny_max = parameters.get("canny_max", 150)
        self.top_width = parameters.get("top_width", 0.4)
        self.bottom_width = parameters.get("bottom_width", 1.0)
        self.trapezoid_height = parameters.get("trapezoid_height", 0.5)
        self.min_line_length = parameters.get("min_line_length", 50)
        self.max_line_gap = parameters.get("max_line_gap", 4)
        self.height = None
        self.width = None
        
        # Attributs pour le suivi temporel
        self.left_line_history = deque(maxlen=5)  # Historique des lignes gauches
        self.right_line_history = deque(maxlen=5)  # Historique des lignes droites

    def get_parameters(self):
        """Retourne les paramètres courants de détection de lignes."""
        return {
            "blur_kernel": self.blur_kernel,
            "threshold_value": self.threshold_value,
            "canny_min": self.canny_min,
            "canny_max": self.canny_max,
            "top_width": self.top_width,
            "bottom_width": self.bottom_width,
            "trapezoid_height": self.trapezoid_height,
            "min_line_length": self.min_line_length,
            "max_line_gap": self.max_line_gap,
        }

    def _set_image_shape(self, image):
        """Définit la hauteur et la largeur de l'image."""
        if self.height is None or self.width is None:
            self.height, self.width = image.shape[:2]

    def canny(self, image):
        """Retourne l'image après seuillage et détection de contours Canny."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (self.blur_kernel, self.blur_kernel), 0)
        _, binary = cv2.threshold(blur, self.threshold_value, 255, cv2.THRESH_BINARY)
        return cv2.Canny(binary, self.canny_min, self.canny_max)

    def region_of_interest(self, image):
        """Applique un masque trapézoïdal à l'image pour ne garder que la zone d'intérêt."""
        self._set_image_shape(image)
        polygons = np.array([
            (int(self.width * (1 - self.bottom_width) / 2), self.height),
            (int(self.width * (1 + self.bottom_width) / 2), self.height),
            (int(self.width * (1 + self.top_width) / 2), int(self.height * (1 - self.trapezoid_height))),
            (int(self.width * (1 - self.top_width) / 2), int(self.height * (1 - self.trapezoid_height)))
        ], dtype=np.int32)
        mask = np.zeros_like(image)
        cv2.fillPoly(mask, [polygons], 255)
        return cv2.bitwise_and(image, mask)

    def make_coordinates(self, line_parameters):
        """Calcule les coordonnées d'une ligne à partir de sa pente et son intercept."""
        slope, intercept = line_parameters
        y1 = self.height
        y2 = int(y1 * (1 / 4))
        x1 = int((y1 - intercept) / slope) if slope != 0 else int(self.width / 2)
        x2 = int((y2 - intercept) / slope) if slope != 0 else int(self.width / 2)
        return np.array([x1, y1, x2, y2])

    def average_slope_intercept(self, lines):
        """
        Moyenne les pentes/intercepts des lots de lignes pour obtenir 2 lignes gauche/droite.
        """
        left_fit, right_fit = [], []
        if lines is None:
            return []
        
        reference_x = self.width // 2
        for line in lines:
            x1, y1, x2, y2 = line.reshape(4)
            if x1 == x2:  # éviter division par zéro
                continue
            slope, intercept = np.polyfit((x1, x2), (y1, y2), 1)
            middle_x = (x1 + x2) / 2
            if slope < 0 and middle_x < reference_x:
                left_fit.append((slope, intercept))
            elif slope > 0 and middle_x > reference_x:
                right_fit.append((slope, intercept))
            elif slope < 0:
                left_fit.append((slope, intercept))
            elif slope > 0:
                right_fit.append((slope, intercept))

        if self.dual_camera:
            # Pour deux caméras, ne retourner qu'une ligne selon le côté
            if self.camera_side == 'left':
                left_line = self.make_coordinates(np.average(left_fit, axis=0)) if left_fit else None
                if left_line is not None:
                    self.left_line_history.append(left_line)
                    return np.array([left_line])
                elif self.left_line_history:
                    return np.array([np.mean(self.left_line_history, axis=0).astype(int)])
                else:
                    return []
            elif self.camera_side == 'right':
                right_line = self.make_coordinates(np.average(right_fit, axis=0)) if right_fit else None
                if right_line is not None:
                    self.right_line_history.append(right_line)
                    return np.array([right_line])
                elif self.right_line_history:
                    return np.array([np.mean(self.right_line_history, axis=0).astype(int)])
                else:
                    return []
            else:
                return []
        else:
            # Pour une seule caméra, retourner les deux lignes si elles existent
            result_lines = []
            if left_fit:
                left_line = self.make_coordinates(np.average(left_fit, axis=0))
                if left_line is not None:
                    self.left_line_history.append(left_line)
                    result_lines.append(left_line)
            elif self.left_line_history:
                # Utiliser l'historique si pas de détection actuelle
                result_lines.append(np.mean(self.left_line_history, axis=0).astype(int))
            if right_fit:
                right_line = self.make_coordinates(np.average(right_fit, axis=0))
                if right_line is not None:
                    self.right_line_history.append(right_line)
                    result_lines.append(right_line)
            elif self.right_line_history:
                # Utiliser l'historique si pas de détection actuelle
                result_lines.append(np.mean(self.right_line_history, axis=0).astype(int))
            return np.array(result_lines)

    def get_lines(self, frame):
        """Retourne les 2 lignes détectées (après moyenne) à partir d'une image."""
        self._set_image_shape(frame)
        canny_image = self.canny(frame)
        cropped_image = self.region_of_interest(canny_image)
        lines = cv2.HoughLinesP(
            cropped_image, 2, np.pi / 180, 50,
            minLineLength=self.min_line_length, maxLineGap=self.max_line_gap
        )
        averaged_lines = self.average_slope_intercept(lines) if lines is not None else []
        return averaged_lines

    def display(self, frame, lines, window_name="Lane Detection", resize=None):
        """Affiche le résultat visuel de la détection de lignes sur une frame."""
        line_image = np.zeros_like(frame)
        if lines is not None and len(lines) > 0:
            for line in lines:
                x1, y1, x2, y2 = map(int, line)
                # Couleur différente selon la position de la ligne
                color = (0, 0, 255)  # Rouge par défaut
                middle_x = (x1 + x2) / 2
                reference_x = self.width // 2
                if middle_x < reference_x:
                    color = (0, 255, 0)  # Vert pour ligne gauche
                else:
                    color = (255, 0, 0)  # Bleu pour ligne droite
                
                cv2.line(line_image, (x1, y1), (x2, y2), color, 10)
        combo_image = cv2.addWeighted(frame, 0.8, line_image, 1, 1)
        if resize:
            combo_image_resized = cv2.resize(combo_image, resize)
            cv2.imshow(window_name, combo_image_resized)
        else:
            cv2.imshow(window_name, combo_image)

    def run(self):
        """Boucle d'affichage continue pour visualiser la détection de lignes.
            Utile uniquement si ce fichier est exécuté directement."""
        if self.cap is not None:    # Si on utilise un fichier vidéo
            while self.cap.isOpened():
                _, frame = self.cap.read()
                lines = self.get_lines(frame)
                self.display(frame, lines)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            self.cap.release()
            cv2.destroyAllWindows()
        elif hasattr(self.video_source, "get_frame"):   # Si on utilise un flux vidéo
            try:
                while True:
                    frame = self.video_source.get_frame()
                    lines = self.get_lines(frame)
                    self.display(frame, lines)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
            finally:
                cv2.destroyAllWindows()

if __name__ == "__main__":
    # Permet d'ajuster les paramètres sur une image locale (pas besoin de caméra)
    import sys
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
        frame = cv2.imread(image_path)
        lane_detector = LaneDetection(frame, dual_camera=False, camera_side="left")
        adjuster = ParameterAdjuster(lane_detector)
        adjuster.adjust_all_parameters()
        # Sauvegarder les paramètres ajustés
        import json
        with open("../single_camera_config.json", "w") as f:
            json.dump(lane_detector.get_parameters(), f)
        print("Paramètres sauvegardés dans single_camera_config.json")
    else:
        video_source = os.path.join(os.path.dirname(__file__), "../media_tests/road_video.mp4")
        lane_detector = LaneDetection(video_source, dual_camera=False, camera_side="left")
        adjuster = ParameterAdjuster(lane_detector)
        adjuster.adjust_all_parameters()
        lane_detector.run()
