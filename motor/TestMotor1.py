import RPi.GPIO as GPIO
import time

# Configuration des broches
AIN1 = 17
AIN2 = 27

GPIO.setmode(GPIO.BCM)
GPIO.setup(AIN1, GPIO.OUT)
GPIO.setup(AIN2, GPIO.OUT)

# PWM sur AIN1
pwm = GPIO.PWM(AIN1, 1000)  # 1000 Hzpip3 install RPi.GPIO

pwm.start(0)  # Démarre avec 0% de duty cycle

def tourner_avant(vitesse):
    pwm.ChangeDutyCycle(vitesse)  # vitesse entre 0 et 100
    GPIO.output(AIN2, GPIO.LOW)

def tourner_arriere(vitesse):
    GPIO.output(AIN1, GPIO.LOW)
    GPIO.output(AIN2, GPIO.HIGH)

def stop():
    GPIO.output(AIN1, GPIO.LOW)
    GPIO.output(AIN2, GPIO.LOW)

try:
    tourner_avant(70)
    time.sleep(2)
    stop()
    time.sleep(1)
    tourner_arriere(1)  # 100% vitesse car pas de PWM sur AIN2 ici
    time.sleep(2)
    stop()

finally:
    pwm.stop()
    GPIO.cleanup()
