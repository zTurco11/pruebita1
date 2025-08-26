import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, accuracy_score

iris = load_iris()
X = iris.data  
y = iris.target  


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

mlp = MLPClassifier(hidden_layer_sizes=(5, 3), max_iter=1000, random_state=42)

mlp.fit(X_train, y_train)

y_pred = mlp.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy del modelo: {accuracy * 100:.2f}%")

print("\nReporte de clasificación:")
print(classification_report(y_test, y_pred))

plt.figure(figsize=(8, 6))
plt.scatter(range(len(y_test)), y_test, label="Etiquetas reales", color='blue')
plt.scatter(range(len(y_pred)), y_pred, label="Predicciones", color='red', marker='x')
plt.title("Comparación de etiquetas reales vs predicciones")
plt.xlabel("Muestras")
plt.ylabel("Clase")
plt.legend()
plt.show()