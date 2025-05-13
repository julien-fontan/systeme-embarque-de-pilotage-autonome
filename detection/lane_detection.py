import cv2
import os
import numpy as np
from .parameter_adjuster import ParameterAdjuster

class LaneDetection:
    def __init__(self, video_source, dual_camera=False, camera_side=None, parameters=None, use_region_of_interest=True):
        """
        :param video_source: Video source for the camera(s) (can be a path, int, or CameraStream object).
        :param dual_camera: Boolean indicating if dual cameras are used.
        :param camera_side: 'left' or 'right' for dual cameras to specify the side of the camera.
        :param parameters: Dictionary of parameters for lane detection.
        :param use_region_of_interest: Boolean to use ROI or not.
        """
        # Si video_source a une méthode get_frame, c'est un objet CameraStream
        self.video_source = video_source
        self.use_region_of_interest = use_region_of_interest
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

    def canny(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (self.blur_kernel, self.blur_kernel), 0)
        _, binary = cv2.threshold(blur, self.threshold_value, 255, cv2.THRESH_BINARY)
        return cv2.Canny(binary, self.canny_min, self.canny_max)

    def region_of_interest(self, image):
        height, width = image.shape[:2]
        polygons = np.array([
            (int(width * (1 - self.bottom_width) / 2), height),
            (int(width * (1 + self.bottom_width) / 2), height),
            (int(width * (1 + self.top_width) / 2), int(height * (1 - self.trapezoid_height))),
            (int(width * (1 - self.top_width) / 2), int(height * (1 - self.trapezoid_height)))
        ], dtype=np.int32)
        mask = np.zeros_like(image)
        cv2.fillPoly(mask, [polygons], 255)
        return cv2.bitwise_and(image, mask)

    def display_lines(self, image, lines):
        line_image = np.zeros_like(image)
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line.reshape(4)
                cv2.line(line_image, (x1, y1), (x2, y2), (0, 0, 255), 10)
        return line_image

    def average_slope_intercept(self, image, lines):
        left_fit, right_fit = [], []
        for line in lines:
            x1, y1, x2, y2 = line.reshape(4)
            parameters = np.polyfit((x1, x2), (y1, y2), 1)
            slope, intercept = parameters
            if slope < 0:
                left_fit.append((slope, intercept))
            else:
                right_fit.append((slope, intercept))

        if self.dual_camera:
            # For dual cameras, classify lines based on the camera side
            if self.camera_side == 'left':
                left_line = self.make_coordinates(image, np.average(left_fit, axis=0)) if left_fit else None
                return np.array([left_line]) if left_line is not None else []
            elif self.camera_side == 'right':
                right_line = self.make_coordinates(image, np.average(right_fit, axis=0)) if right_fit else None
                return np.array([right_line]) if right_line is not None else []
        else:
            # For single camera, return both left and right lines
            left_line = self.make_coordinates(image, np.average(left_fit, axis=0)) if left_fit else None
            right_line = self.make_coordinates(image, np.average(right_fit, axis=0)) if right_fit else None
            return np.array([line for line in [left_line, right_line] if line is not None])

    def make_coordinates(self, image, line_parameters):
        slope, intercept = line_parameters
        y1 = image.shape[0]
        y2 = int(y1 * (2 / 3))
        x1 = int((y1 - intercept) / slope)
        x2 = int((y2 - intercept) / slope)
        return np.array([x1, y1, x2, y2])

    def process_frame(self, frame):
        canny_image = self.canny(frame)
        cropped_image = self.region_of_interest(canny_image) if self.use_region_of_interest else canny_image
        lines = cv2.HoughLinesP(cropped_image, 2, np.pi / 180, 50, minLineLength=self.min_line_length, maxLineGap=self.max_line_gap)
        averaged_lines = self.average_slope_intercept(frame, lines) if lines is not None else []
        line_image = self.display_lines(frame, averaged_lines)
        return cv2.addWeighted(frame, 0.8, line_image, 1, 1)

    def run(self):
        if self.cap is not None:
            while self.cap.isOpened():
                ret, frame = self.cap.read()
                if not ret:
                    break
                processed_frame = self.process_frame(frame)
                resized_processed_frame = cv2.resize(processed_frame, (1200, 600))
                cv2.imshow("Lane Detection", resized_processed_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            self.cap.release()
            cv2.destroyAllWindows()
        elif hasattr(self.video_source, "get_frame"):
            # Utilisation d'un objet CameraStream
            try:
                while True:
                    frame = self.video_source.get_frame()
                    processed_frame = self.process_frame(frame)
                    resized_processed_frame = cv2.resize(processed_frame, (1200, 600))
                    cv2.imshow("Lane Detection", resized_processed_frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
            finally:
                cv2.destroyAllWindows()

    def get_parameters(self):
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

if __name__ == "__main__":
    video_source = os.path.join(os.path.dirname(__file__), "../media_tests/road_video.mp4")
    lane_detector = LaneDetection(video_source, dual_camera=False, camera_side="left")
    adjuster = ParameterAdjuster(lane_detector)
    adjuster.adjust_all_parameters()
    lane_detector.run()
