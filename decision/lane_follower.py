class LaneFollower:
    def __init__(self, dual_camera=False):
        self.dual_camera = dual_camera

    def decide_action(self, lines, frame_shape):
        # lines: tableau de lignes détectées (peut être 0, 1 ou 2)
        # frame_shape: (height, width, ...)
        frame_center = frame_shape[1] // 2

        if not self.dual_camera:

            """ PARTIE A POTENTIELLEMENT MODIFIER """
            if len(lines) == 2:
                # Calculer la position centrale de la route
                left_x2 = lines[0][2]
                right_x2 = lines[1][2]
                lane_center = (left_x2 + right_x2) // 2
                offset = lane_center - frame_center
                if offset < -30:
                    return "left"
                elif offset > 30:
                    return "right"
                else:
                    return "stop"
                
            elif len(lines) == 1:
                x2 = lines[0][2]
                if x2 < frame_center:
                    return "left"
                elif x2 > frame_center:
                    return "right"
                else:
                    return "stop"
                
            elif len(lines) == 0:
                return "stop"   # A MODIFIER, VOIR SI ON SOULEVE PAS UNE ERREUR ?
            
            """ FIN PARTIE A POTENTIELLEMENT MODIFIER """