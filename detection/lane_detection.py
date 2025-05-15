import cv2
import os
import numpy as np
from .parameter_adjuster import ParameterAdjuster

class LaneDetection:
    def __init__(self, video_source, dual_camera=False, camera_side=None, parameters=None):
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
        y2 = int(y1 * (2 / 3))
        x1 = int((y1 - intercept) / slope)
        x2 = int((y2 - intercept) / slope)
        return np.array([x1, y1, x2, y2])

    def average_slope_intercept(self, lines):
        """Moyenne les pentes/intercepts des lots de lignes pour obtenir 2 lignes gauche/droite."""
        left_fit, right_fit = [], []
        if lines is None:
            return []
        for line in lines:
            x1, y1, x2, y2 = line.reshape(4)
            if x1 == x2:  # éviter division par zéro
                continue
            slope, intercept = np.polyfit((x1, x2), (y1, y2), 1)
            """ Potentiellement modifier la condition suivante, en effet elle risque de ne pas marcher
            dans un virage tendu."""
            if slope < 0:
                left_fit.append((slope, intercept))
            else:
                right_fit.append((slope, intercept))

        if self.dual_camera:
            # Pour deux caméras, ne retourner qu'une ligne selon le côté
            if self.camera_side == 'left':
                left_line = self.make_coordinates(np.average(left_fit, axis=0)) if left_fit else None
                return np.array([left_line]) if left_line is not None else []
            elif self.camera_side == 'right':
                right_line = self.make_coordinates(np.average(right_fit, axis=0)) if right_fit else None
                return np.array([right_line]) if right_line is not None else []
            else:
                return []
        else:
            # Pour une seule caméra, retourner les deux lignes si elles existent
            result_lines = []
            if left_fit:
                left_line = self.make_coordinates(np.average(left_fit, axis=0))
                if left_line is not None:
                    result_lines.append(left_line)
            if right_fit:
                right_line = self.make_coordinates(np.average(right_fit, axis=0))
                if right_line is not None:
                    result_lines.append(right_line)
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

    def display(self, frame, lines, window_name="Lane Detection", resize=(1200, 600)):
        """Affiche le résultat visuel de la détection de lignes sur une frame."""
        line_image = np.zeros_like(frame)
        if lines is not None and len(lines) > 0:
            for line in lines:
                if line is not None and len(line) == 4 and np.all(np.isfinite(line)):
                    x1, y1, x2, y2 = line.reshape(4)
                    cv2.line(line_image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 10)
        combo_image = cv2.addWeighted(frame, 0.8, line_image, 1, 1)
        combo_image_resized = cv2.resize(combo_image,resize)
        cv2.imshow(window_name, combo_image_resized)
    
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
    video_source = os.path.join(os.path.dirname(__file__), "../media_tests/road_video.mp4")
    lane_detector = LaneDetection(video_source, dual_camera=False, camera_side="left")
    adjuster = ParameterAdjuster(lane_detector)
    adjuster.adjust_all_parameters()
    lane_detector.run()
