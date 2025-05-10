import RPi.GPIO as GPIO

class MotorControl:
    def __init__(self, left_motor_pins, right_motor_pins):
        self.left_motor_pins = left_motor_pins
        self.right_motor_pins = right_motor_pins
        GPIO.setmode(GPIO.BCM)
        for pin in left_motor_pins + right_motor_pins:
            GPIO.setup(pin, GPIO.OUT)
            GPIO.output(pin, GPIO.LOW)

    def move(self, left_speed, right_speed):
        # Exemple simple : contrôle des moteurs avec PWM
        for pin, speed in zip(self.left_motor_pins, left_speed):
            GPIO.output(pin, GPIO.HIGH if speed > 0 else GPIO.LOW)
        for pin, speed in zip(self.right_motor_pins, right_speed):
            GPIO.output(pin, GPIO.HIGH if speed > 0 else GPIO.LOW)

    def stop(self):
        for pin in self.left_motor_pins + self.right_motor_pins:
            GPIO.output(pin, GPIO.LOW)
        GPIO.cleanup()