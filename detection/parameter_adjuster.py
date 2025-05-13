import cv2
import numpy as np
import json
import os

class ParameterAdjuster:
    def __init__(self, lane_detector):
        self.lane_detector = lane_detector

        # Charger les paramètres de base si non définis
        if not hasattr(lane_detector, "parameters") or not lane_detector.get_parameters():
            config_path = os.path.join(os.path.dirname(__file__), "../single_camera_config.json")
            if os.path.exists(config_path):
                with open(config_path, "r") as f:
                    parameters = json.load(f)
                # Met à jour les attributs du lane_detector
                lane_detector.blur_kernel = parameters.get("blur_kernel", 7)
                lane_detector.threshold_value = parameters.get("threshold_value", 145)
                lane_detector.canny_min = parameters.get("canny_min", 50)
                lane_detector.canny_max = parameters.get("canny_max", 150)
                lane_detector.top_width = parameters.get("top_width", 0.4)
                lane_detector.bottom_width = parameters.get("bottom_width", 1.0)
                lane_detector.trapezoid_height = parameters.get("trapezoid_height", 0.5)
                lane_detector.min_line_length = parameters.get("min_line_length", 50)
                lane_detector.max_line_gap = parameters.get("max_line_gap", 4)

    def get_live_frame(self):
        if self.lane_detector.cap is not None:
            ret, frame = self.lane_detector.cap.read()
            return frame if ret else None
        elif hasattr(self.lane_detector.video_source, "get_frame"):
            return self.lane_detector.video_source.get_frame()
        else:
            return None

    def adjust_canny_parameters(self):
        def nothing(x):
            pass

        cv2.namedWindow("Canny")
        cv2.createTrackbar("Blur", "Canny", self.lane_detector.blur_kernel, 50, nothing)
        cv2.createTrackbar("Threshold", "Canny", self.lane_detector.threshold_value, 255, nothing)
        cv2.createTrackbar("Canny Min", "Canny", self.lane_detector.canny_min, 255, nothing)
        cv2.createTrackbar("Canny Max", "Canny", self.lane_detector.canny_max, 255, nothing)

        while True:
            frame = self.get_live_frame()
            if frame is None:
                print("Erreur : Impossible de lire une image de la vidéo.")
                break

            self.lane_detector.blur_kernel = max(1, cv2.getTrackbarPos("Blur", "Canny") | 1)
            self.lane_detector.threshold_value = cv2.getTrackbarPos("Threshold", "Canny")
            self.lane_detector.canny_min = cv2.getTrackbarPos("Canny Min", "Canny")
            self.lane_detector.canny_max = cv2.getTrackbarPos("Canny Max", "Canny")

            canny_image = self.lane_detector.canny(frame)
            resized_canny_image = cv2.resize(canny_image, (1200, 600))
            cv2.imshow("Canny", resized_canny_image)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cv2.destroyWindow("Canny")

    def adjust_trapezoid_parameters(self):
        def nothing(x):
            pass

        cv2.namedWindow("Adjust Trapezoid")
        cv2.createTrackbar("Top Width (%)", "Adjust Trapezoid", int(self.lane_detector.top_width * 100), 100, nothing)
        cv2.createTrackbar("Bottom Width (%)", "Adjust Trapezoid", int(self.lane_detector.bottom_width * 100), 100, nothing)
        cv2.createTrackbar("Height (%)", "Adjust Trapezoid", int(self.lane_detector.trapezoid_height * 100), 100, nothing)

        while True:
            frame = self.get_live_frame()
            if frame is None:
                print("Erreur : Impossible de lire une image de la vidéo.")
                break

            self.lane_detector.top_width = cv2.getTrackbarPos("Top Width (%)", "Adjust Trapezoid") / 100
            self.lane_detector.bottom_width = cv2.getTrackbarPos("Bottom Width (%)", "Adjust Trapezoid") / 100
            self.lane_detector.trapezoid_height = cv2.getTrackbarPos("Height (%)", "Adjust Trapezoid") / 100

            masked_image = self.lane_detector.region_of_interest(frame)
            resized_masked_image = cv2.resize(masked_image, (1200, 600))
            cv2.imshow("Adjust Trapezoid", resized_masked_image)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cv2.destroyWindow("Adjust Trapezoid")

    def adjust_hough_parameters(self):
        def nothing(x):
            pass

        cv2.namedWindow("Hough")
        cv2.createTrackbar("Min Length", "Hough", self.lane_detector.min_line_length, 500, nothing)
        cv2.createTrackbar("Max Gap", "Hough", self.lane_detector.max_line_gap, 50, nothing)

        while True:
            frame = self.get_live_frame()
            if frame is None:
                print("Erreur : Impossible de lire une image de la vidéo.")
                break

            self.lane_detector.min_line_length = cv2.getTrackbarPos("Min Length", "Hough")
            self.lane_detector.max_line_gap = cv2.getTrackbarPos("Max Gap", "Hough")

            processed_image = self.lane_detector.canny(frame)
            cropped_image = self.lane_detector.region_of_interest(processed_image)
            lines = cv2.HoughLinesP(cropped_image, 2, np.pi / 180, 50, minLineLength=self.lane_detector.min_line_length, maxLineGap=self.lane_detector.max_line_gap)
            averaged_lines = self.lane_detector.average_slope_intercept(frame, lines) if lines is not None else []
            line_image = self.lane_detector.display_lines(frame, averaged_lines)
            combo_image = cv2.addWeighted(frame, 0.8, line_image, 1, 1)

            resized_combo_image = cv2.resize(combo_image, (1200, 600))
            cv2.imshow("Hough", resized_combo_image)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cv2.destroyWindow("Hough")

    def adjust_all_parameters(self):
        # On ne lit plus la première image, on ajuste en live
        self.adjust_canny_parameters()
        self.adjust_trapezoid_parameters()
        self.adjust_hough_parameters()
