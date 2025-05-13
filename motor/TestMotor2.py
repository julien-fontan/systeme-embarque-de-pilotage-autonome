import RPi.GPIO as GPIO
import time

# Configuration des GPIO
IN1 = 17  # GPIO17
IN2 = 27  # GPIO27
PWM_FREQ = 1000  # fréquence PWM en Hz

# Initialisation
GPIO.setmode(GPIO.BCM)
GPIO.setup(IN1, GPIO.OUT)
GPIO.setup(IN2, GPIO.OUT)

# PWM sur les deux broches
pwm1 = GPIO.PWM(IN1, PWM_FREQ)
pwm2 = GPIO.PWM(IN2, PWM_FREQ)
pwm1.start(0)
pwm2.start(0)

def stop():
    pwm1.ChangeDutyCycle(0)
    pwm2.ChangeDutyCycle(0)
    print("Moteur arrêté")

def forward(speed=100):
    pwm1.ChangeDutyCycle(speed)
    pwm2.ChangeDutyCycle(0)
    print(f"Moteur en avant à {speed}%")

def backward(speed=100):
    pwm1.ChangeDutyCycle(0)
    pwm2.ChangeDutyCycle(speed)
    print(f"Moteur en arrière à {speed}%")

def brake():
    pwm1.ChangeDutyCycle(100)
    pwm2.ChangeDutyCycle(100)
    print("Frein actif")

def test_all_commands():
    try:
        print("Test : Avant à 80%")
        forward(80)
        time.sleep(2)

        print("Arrêt")
        stop()
        time.sleep(1)

        print("Test : Arrière à 30%")
        backward(30)
        time.sleep(2)

        print("Freinage")
        brake()
        time.sleep(1)

        print("Arrêt final")
        stop()
    finally:
        pwm1.stop()
        pwm2.stop()
        GPIO.cleanup()

# Exemple d'utilisation
if __name__ == "_main_":
    test_all_commands()