from picamera2 import Picamera2
import cv2

class CameraStream:
    def __init__(self, camera_id=0):
        self.camera = Picamera2(camera_id)
        self.camera.configure(self.camera.create_preview_configuration())
        self.camera.start()

    def get_frame(self):
        return self.camera.capture_array()

    def stop(self):
        self.camera.stop()