class LaneFollower:
    def __init__(self, dual_camera=False):
        self.dual_camera = dual_camera
        self.last_offset = 0  # Pour garder la dernière direction connue

    def compute_offset(self, lines, frame_shape):
        frame_center = frame_shape[1] // 2
        if not self.dual_camera:
            if len(lines) == 2:
                left_x2 = lines[0][2]
                right_x2 = lines[1][2]
                lane_center = (left_x2 + right_x2) // 2
                offset = lane_center - frame_center
                self.last_offset = offset
                return offset
            elif len(lines) == 1:
                x2 = lines[0][2]
                offset = x2 - frame_center
                self.last_offset = offset
                return offset
            else:
                # Si aucune ligne détectée, garder la dernière valeur (ou 0)
                return self.last_offset
        else:
            # Dual camera: à adapter si besoin
            return 0

    def decide_action(self, lines, frame_shape):
        # Pour compatibilité, garder l'ancienne logique
        offset = self.compute_offset(lines, frame_shape)
        if offset < -30:
            return "left"
        elif offset > 30:
            return "right"
        else:
            return "stop"

    def get_offset(self, lines, frame_shape):
        # Nouvelle méthode pour obtenir l'offset directement
        return self.compute_offset(lines, frame_shape)