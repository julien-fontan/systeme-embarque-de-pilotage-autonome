from camera.camera_stream import CameraStream
from detection.lane_detection import LaneDetection
from detection.parameter_adjuster import ParameterAdjuster
from decision.lane_follower import LaneFollower
from motor.motor_controller import MotorController
import numpy as np
import cv2
import json

def main(dual_camera=False, show_visuals=False, adjust_parameters=False, use_vanishing_point=True):

    config_file = "dual_camera_config.json" if dual_camera else "single_camera_config.json"
    
    if adjust_parameters and show_visuals:
        # Utiliser la caméra 0 pour ajuster les paramètres
        camera1 = CameraStream(camera_id=0)
        lane_detector = LaneDetection(video_source=camera1, dual_camera=dual_camera)
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

    if dual_camera:
        lane_detector1 = LaneDetection(video_source=camera1, dual_camera=True, camera_side='left', 
                                      parameters=parameters)
        lane_detector2 = LaneDetection(video_source=camera2, dual_camera=True, camera_side='right', 
                                      parameters=parameters)
    else:
        lane_detector1 = LaneDetection(video_source=camera1, dual_camera=False, 
                                      parameters=parameters)

    motor_controller = MotorController()
    lane_follower = LaneFollower(dual_camera=dual_camera)

    try:
        while True:
            frame1 = camera1.get_frame()
            frame2 = camera2.get_frame() if dual_camera else None

            if dual_camera:
                lines1 = lane_detector1.get_lines(frame1)
                lines2 = lane_detector2.get_lines(frame2)
                if show_visuals:
                    lane_detector1.display(frame1, lines1, window_name="Lane Detection - Camera 1", resize=None)
                    lane_detector2.display(frame2, lines2, window_name="Lane Detection - Camera 2", resize=None)
                # Normalement, mettre ici la logique de commande moteur pour 2 caméras (on s'en fout)
            else:
                lines1 = lane_detector1.get_lines(frame1)
                if show_visuals:
                    lane_detector1.display(frame1, lines1, window_name="Lane Detection - Single Camera")

                # Commande proportionnelle du moteur
                offset = lane_follower.get_offset(lines1, frame1.shape)
                motor_controller.set_steering(offset)
                
                # Ancienne commande (à garder pour compatibilité)
                # action = lane_follower.decide_action(lines1, frame1.shape)
                # if action == "left":
                #     motor_controller.left()
                # elif action == "right":
                #     motor_controller.right()
                # elif action == "stop":
                #     motor_controller.stop()

            if show_visuals and cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        camera1.stop()
        if dual_camera:
            camera2.stop()
        if show_visuals:
            cv2.destroyAllWindows()

if __name__ == "__main__":
    main(dual_camera=False, show_visuals=False, adjust_parameters=False)
    """ Si la raspberry est connectée en SSH, utiliser show_visuals=False (donc adjust_parameters=False)
    Si vous voulez visualiser les images en temps réel (et potentiellement utiliser adjust_parameters),
    connectez la raspberry à un écran, éxécutez ce code sur un terminal directement sur la carte,
    et réglez show_visuals=True."""
