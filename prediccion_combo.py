import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, mean_squared_error

X, y = make_blobs(n_samples=300, centers=3, cluster_std=1.0, random_state=42)

plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis')
plt.title('Datos Sintéticos Generados')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.show()

kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(X)
y_kmeans = kmeans.predict(X)

plt.scatter(X[:, 0], X[:, 1], c=y_kmeans, cmap='viridis')
plt.title('Clustering con K-Means')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.show()


y_reg = X[:, 0] * 2 + X[:, 1] * 3 + np.random.randn(300) * 0.5  

X_train, X_test, y_train, y_test = train_test_split(X, y_reg, test_size=0.3, random_state=42)

reg_model = LinearRegression()
reg_model.fit(X_train, y_train)

y_pred = reg_model.predict(X_test)
print(f"Mean Squared Error en Regresión Lineal: {mean_squared_error(y_test, y_pred)}")

plt.scatter(X_test[:, 0], y_test, color='blue', label='Datos reales')
plt.scatter(X_test[:, 0], y_pred, color='red', label='Predicciones')
plt.title('Regresión Lineal')
plt.xlabel('Feature 1')
plt.ylabel('Valor Predicho')
plt.legend()
plt.show()

X_train, X_test, y_train, y_test = train_test_split(X, y_kmeans, test_size=0.3, random_state=42)


mlp_model = MLPClassifier(hidden_layer_sizes=(5,), max_iter=1000, random_state=42)
mlp_model.fit(X_train, y_train)


y_pred_mlp = mlp_model.predict(X_test)
print("Reporte de clasificación MLP:\n", classification_report(y_test, y_pred_mlp))


plt.scatter(X_test[:, 0], X_test[:, 1], c=y_pred_mlp, cmap='viridis')
plt.title('Clasificación con MLP')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.show()