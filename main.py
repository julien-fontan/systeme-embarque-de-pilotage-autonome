from camera.camera_stream import CameraStream
from detection.lane_detection import LaneDetection
from motor.motor_control import MotorControl
import cv2

def main():
    # Initialisation
    camera = CameraStream()
    lane_detector = LaneDetection(video_source=None)  # Pas de vidéo source, on utilise le flux live
    motor = MotorControl(left_motor_pins=[17, 18], right_motor_pins=[22, 23])

    try:
        while True:
            frame = camera.get_frame()
            processed_frame = lane_detector.process_frame(frame)

            # Exemple de logique pour ajuster les moteurs
            # (à adapter selon les résultats de la détection)
            left_speed, right_speed = 0.5, 0.5  # Exemple de vitesse
            motor.move([left_speed], [right_speed])

            cv2.imshow("Lane Detection", processed_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        camera.stop()
        motor.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()