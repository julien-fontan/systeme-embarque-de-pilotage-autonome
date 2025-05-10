import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline

# Définition des segments de droites
segments = [((0, 0), (1, 2)),
            ((1, 2), (3, 3)),
            ((3, 3), (4, 2)),
            ((4, 2), (6, 1))]

# Calcul des milieux des segments
milieux = [((p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2) for p1, p2 in segments]
milieux = np.array(milieux)  # Conversion en tableau NumPy

# Création de la spline cubique
x = milieux[:, 0]  # Coordonnées X des milieux
y = milieux[:, 1]  # Coordonnées Y des milieux

# Générer une interpolation spline
cs = CubicSpline(x, y, bc_type='natural')

# Génération des points interpolés
x_fine = np.linspace(x[0], x[-1], 100)
y_fine = cs(x_fine)

# Tracé des segments initiaux
plt.figure(figsize=(8, 6))
for p1, p2 in segments:
    plt.plot([p1[0], p2[0]], [p1[1], p2[1]], 'r--', label="Segments de droites")

# Tracé des points milieux
plt.scatter(x, y, color='blue', label="Milieux des segments")

# Tracé de la courbe interpolée
plt.plot(x_fine, y_fine, 'g-', linewidth=2, label="Courbe interpolée")

# Personnalisation du graphe
plt.title("Courbe passant par les milieux des segments")
plt.xlabel("X")
plt.ylabel("Y")
plt.legend()
plt.grid()
plt.show()


