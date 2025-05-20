import RPi.GPIO as GPIO

class MotorController:
    def __init__(self):
        # Pins pour le moteur unique
        self.EN = 18
        self.IN1 = 20
        self.IN2 = 16

        GPIO.setmode(GPIO.BCM)
        GPIO.setwarnings(False)
        GPIO.setup(self.EN, GPIO.OUT)
        GPIO.setup(self.IN1, GPIO.OUT)
        GPIO.setup(self.IN2, GPIO.OUT)

        # PWM pour la vitesse (100% par défaut)
        self.pwm = GPIO.PWM(self.EN, 100)
        self.pwm.start(100)

        self.stop()

    def left(self):
        # Tourner à gauche (sens 1)
        GPIO.output(self.IN1, GPIO.HIGH)
        GPIO.output(self.IN2, GPIO.LOW)

    def right(self):
        # Tourner à droite (sens 2)
        GPIO.output(self.IN1, GPIO.LOW)
        GPIO.output(self.IN2, GPIO.HIGH)

    def stop(self):
        # Arrêt du moteur
        GPIO.output(self.IN1, GPIO.LOW)
        GPIO.output(self.IN2, GPIO.LOW)

    def set_steering(self, offset, max_offset=200):
        """
        Commande proportionnelle du moteur selon l'offset.
        offset: décalage (positif = droite, négatif = gauche)
        max_offset: valeur max attendue pour l'offset (pour normaliser)
        """
        # Clamp offset
        offset = max(-max_offset, min(max_offset, offset))
        # Augmentation de la puissance minimale à 50% (au lieu de 30%)
        # et utilisation de 100% plus rapidement
        duty_cycle = min(60, 50 + int(50 * abs(offset) / max_offset))  # 50% mini, 100% max

        if offset < -50:
            # Gauche proportionnelle - CORRECTION : on tourne à droite quand on est à gauche de la route
            GPIO.output(self.IN1, GPIO.LOW)
            GPIO.output(self.IN2, GPIO.HIGH)
            self.pwm.ChangeDutyCycle(duty_cycle)
        elif offset > 50:
            # Droite proportionnelle - CORRECTION : on tourne à gauche quand on est à droite de la route
            GPIO.output(self.IN1, GPIO.HIGH)
            GPIO.output(self.IN2, GPIO.LOW)
            self.pwm.ChangeDutyCycle(duty_cycle)
        else:
            # Centré
            self.stop()

    def __del__(self):
        self.stop()
        GPIO.cleanup()

"""
REMPLIR CE FICHIER POUR LE CONTRÔLE DU MOTEUR
"""