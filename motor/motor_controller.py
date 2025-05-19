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

    def __del__(self):
        self.stop()
        GPIO.cleanup()

"""
REMPLIR CE FICHIER POUR LE CONTRÔLE DU MOTEUR
"""