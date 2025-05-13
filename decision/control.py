class LaneFollower:
    def __init__(self, motor_controller):
        self.motor_controller = motor_controller

    def follow_lane(self, lines, frame_shape):
        # lines: tableau de lignes détectées (peut être 0, 1 ou 2)
        # frame_shape: (height, width, ...)
        frame_center = frame_shape[1] // 2
        if len(lines) == 2:
            # Calculer la position centrale de la route
            left_x2 = lines[0][2]
            right_x2 = lines[1][2]
            lane_center = (left_x2 + right_x2) // 2
            offset = lane_center - frame_center
            if offset < -30:
                self.motor_controller.left()
            elif offset > 30:
                self.motor_controller.right()
        elif len(lines) == 1:
            # Stratégie simple : suivre la ligne détectée
            x2 = lines[0][2]
            if x2 < frame_center:
                self.motor_controller.left()
            else:
                self.motor_controller.right()
        else:
            self.motor_controller.stop()

# ajouter une fonction pour interpréter les pentes des lignes détectées