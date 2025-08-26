import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

X = np.array([50, 60, 70, 80, 90, 100, 110, 120, 130, 140]).reshape(-1, 1)
y = np.array([150, 180, 200, 220, 250, 270, 300, 320, 350, 380])

modelo = LinearRegression()
modelo.fit(X, y)

X_pred = np.linspace(40, 150, 100).reshape(-1, 1)
y_pred = modelo.predict(X_pred)

plt.scatter(X, y, color='blue', label="Datos reales")
plt.plot(X_pred, y_pred, color='red', label="Regresión Lineal")
plt.xlabel("Tamaño de la Casa (m²)")
plt.ylabel("Precio ($1000s)")
plt.title("Regresión Lineal: Precio vs Tamaño de la Casa")
plt.legend()
plt.show()