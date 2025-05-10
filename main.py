from camera.camera_stream import CameraStream
from detection.lane_detection import LaneDetection
from detection.parameter_adjuster import ParameterAdjuster
import cv2
import json

def main(dual_camera=True, show_visuals=True, adjust_parameters=False, use_region_of_interest=True):
    config_file = "dual_camera_config.json" if dual_camera else "single_camera_config.json"

    if adjust_parameters:
        lane_detector = LaneDetection(video_source=None, dual_camera=dual_camera)
        adjuster = ParameterAdjuster(lane_detector)
        adjuster.adjust_all_parameters()
        # Sauvegarder les paramètres ajustés dans un fichier
        with open(config_file, "w") as f:
            json.dump(lane_detector.get_parameters(), f)
        print(f"Configuration sauvegardée dans {config_file}")
        return

    # Charger les paramètres depuis le fichier
    try:
        with open(config_file, "r") as f:
            parameters = json.load(f)
    except FileNotFoundError:
        print(f"Fichier de configuration {config_file} introuvable. Utilisation des paramètres par défaut.")
        parameters = {}

    lane_detector = LaneDetection(video_source=None, dual_camera=dual_camera, parameters=parameters, use_region_of_interest=use_region_of_interest)
    camera1 = CameraStream(camera_id=0)
    camera2 = CameraStream(camera_id=1) if dual_camera else None

    try:
        while True:
            frame1 = camera1.get_frame()
            frame2 = camera2.get_frame() if dual_camera else None

            if dual_camera:
                processed_frame1 = lane_detector.process_frame(frame1)
                processed_frame2 = lane_detector.process_frame(frame2)
                if show_visuals:
                    combined_frame = cv2.hconcat([processed_frame1, processed_frame2])
                    cv2.imshow("Lane Detection - Dual Cameras", combined_frame)
            else:
                processed_frame = lane_detector.process_frame(frame1)
                if show_visuals:
                    cv2.imshow("Lane Detection - Single Camera", processed_frame)

            if show_visuals and cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        camera1.stop()
        if dual_camera:
            camera2.stop()
        if show_visuals:
            cv2.destroyAllWindows()

if __name__ == "__main__":
    main(dual_camera=True, show_visuals=True, adjust_parameters=False, use_region_of_interest=True)