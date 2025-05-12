from camera.camera_stream import CameraStream
from detection.lane_detection import LaneDetection
from detection.parameter_adjuster import ParameterAdjuster
import cv2
import json

def main(dual_camera=False, show_visuals=True, adjust_parameters=False, use_region_of_interest=True):
    config_file = "dual_camera_config.json" if dual_camera else "single_camera_config.json"
    
    if adjust_parameters:
        # Utiliser la caméra 0 pour ajuster les paramètres
        camera1 = CameraStream(camera_id=0)
        frame = camera1.get_frame()
        lane_detector = LaneDetection(video_source=0, dual_camera=dual_camera)
        adjuster = ParameterAdjuster(lane_detector)
        adjuster.adjust_all_parameters()
        # Sauvegarder les paramètres ajustés dans un fichier
        with open(config_file, "w") as f:
            json.dump(lane_detector.get_parameters(), f)
        print(f"Configuration sauvegardée dans {config_file}")
        camera1.stop()
        return

    # Charger les paramètres depuis le fichier
    try:
        with open(config_file, "r") as f:
            parameters = json.load(f)
    except FileNotFoundError:
        print(f"Fichier de configuration {config_file} introuvable. Utilisation des paramètres par défaut.")
        parameters = {}

    camera1 = CameraStream(camera_id=0)
    camera2 = CameraStream(camera_id=1) if dual_camera else None

    # Passe la source vidéo correcte à LaneDetection
    if dual_camera:
        lane_detector1 = LaneDetection(video_source=camera1, dual_camera=True, camera_side='left', parameters=parameters, use_region_of_interest=use_region_of_interest)
        lane_detector2 = LaneDetection(video_source=camera2, dual_camera=True, camera_side='right', parameters=parameters, use_region_of_interest=use_region_of_interest)
    else:
        lane_detector = LaneDetection(video_source=camera1, dual_camera=False, parameters=parameters, use_region_of_interest=use_region_of_interest)

    try:
        while True:
            frame1 = camera1.get_frame()
            frame2 = camera2.get_frame() if dual_camera and camera2 is not None else None

            if dual_camera:
                processed_frame1 = lane_detector1.process_frame(frame1)
                processed_frame2 = lane_detector2.process_frame(frame2)
                if show_visuals:
                    # Redimensionner pour concaténer proprement
                    h, w = processed_frame1.shape[:2]
                    processed_frame2 = cv2.resize(processed_frame2, (w, h))
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
        if dual_camera and camera2 is not None:
            camera2.stop()
        if show_visuals:
            cv2.destroyAllWindows()

if __name__ == "__main__":
    main(dual_camera=False, show_visuals=False, adjust_parameters=True, use_region_of_interest=True)


# rpicam-hello --list cameras
# libcamera-hello --list cameras (outdated)
# pour regarder sur l'écran connecté au raspberry, tapper dans le terminal : export DISPLAY=:0
