import RPi.GPIO as GPIO
from time import sleep

# Définition des pins
M1_En = 21
M1_In1 = 20
M1_In2 = 16

M2_En = 18
M2_In1 = 23
M2_In2 = 24

# Création d'une liste des pins pour chaque moteur
Pins = [[M1_En, M1_In1, M1_In2], [M2_En, M2_In1, M2_In2]]

# Configuration GPIO
GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)

# Setup des pins
for moteur in Pins:
    for pin in moteur:
        GPIO.setup(pin, GPIO.OUT)
        GPIO.output(pin, GPIO.LOW)

# PWM pour contrôler la vitesse
M1_Vitesse = GPIO.PWM(M1_En, 100)
M2_Vitesse = GPIO.PWM(M2_En, 100)
M1_Vitesse.start(100)
M2_Vitesse.start(100)

def sens1(moteurNum):
    """Fait tourner le moteur dans le sens 1 (horaire)"""
    GPIO.output(Pins[moteurNum - 1][1], GPIO.HIGH)
    GPIO.output(Pins[moteurNum - 1][2], GPIO.LOW)
    print(f"Moteur {moteurNum} tourne dans le sens 1 (horaire)")

def sens2(moteurNum):
    """Fait tourner le moteur dans le sens 2 (anti-horaire)"""
    GPIO.output(Pins[moteurNum - 1][1], GPIO.LOW)
    GPIO.output(Pins[moteurNum - 1][2], GPIO.HIGH)
    print(f"Moteur {moteurNum} tourne dans le sens 2 (anti-horaire)")

def arret(moteurNum):
    """Arrête un moteur spécifique"""
    GPIO.output(Pins[moteurNum - 1][1], GPIO.LOW)
    GPIO.output(Pins[moteurNum - 1][2], GPIO.LOW)
    print(f"Moteur {moteurNum} arrêté")

def arretComplet():
    """Arrête tous les moteurs"""
    for moteur in Pins:
        GPIO.output(moteur[1], GPIO.LOW)
        GPIO.output(moteur[2], GPIO.LOW)
    print("Tous les moteurs sont arrêtés")

def changer_vitesse(moteurNum, vitesse):
    """Change la vitesse d'un moteur (0-100%)"""
    if moteurNum == 1:
        M1_Vitesse.ChangeDutyCycle(vitesse)
    else:
        M2_Vitesse.ChangeDutyCycle(vitesse)
    print(f"Vitesse du moteur {moteurNum} réglée à {vitesse}%")

print("=== Test des moteurs ===")
print("Séquence de test: avant - arrêt - arrière - arrêt - alternance gauche/droite")

try:
    # Démarrage en douceur
    print("\nTest 1: Démarrage progressif du moteur 1")
    for vitesse in range(20, 101, 10):
        changer_vitesse(1, vitesse)
        sens1(1)
        sleep(0.5)
    arret(1)
    
    print("\nTest 2: Démarrage progressif du moteur 2")
    for vitesse in range(20, 101, 10):
        changer_vitesse(2, vitesse)
        sens1(2)
        sleep(0.5)
    arret(2)
    
    print("\nTest 3: Les deux moteurs ensemble (avant)")
    changer_vitesse(1, 80)
    changer_vitesse(2, 80)
    sens1(1)
    sens1(2)
    sleep(3)
    arretComplet()
    sleep(1)
    
    print("\nTest 4: Les deux moteurs ensemble (arrière)")
    changer_vitesse(1, 80)
    changer_vitesse(2, 80)
    sens2(1)
    sens2(2)
    sleep(3)
    arretComplet()
    sleep(1)
    
    print("\nTest 5: Rotation gauche/droite")
    for _ in range(3):
        # Tourner à gauche
        sens1(1)
        sens2(2)
        sleep(1)
        arretComplet()
        sleep(0.5)
        
        # Tourner à droite
        sens2(1)
        sens1(2)
        sleep(1)
        arretComplet()
        sleep(0.5)
    
    print("\nTest terminé!")
    
except KeyboardInterrupt:
    print("\nTest interrompu par l'utilisateur")
finally:
    arretComplet()
    M1_Vitesse.stop()
    M2_Vitesse.stop()
    GPIO.cleanup()
    print("GPIO nettoyé")
