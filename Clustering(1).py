import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


np.random.seed(42)
cluster_1 = np.random.normal(loc=[2, 2], scale=0.5, size=(10, 2))
cluster_2 = np.random.normal(loc=[6, 6], scale=0.5, size=(10, 2))
cluster_3 = np.random.normal(loc=[10, 2], scale=0.5, size=(10, 2))

data = np.vstack((cluster_1, cluster_2, cluster_3))

kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
kmeans.fit(data)
labels = kmeans.labels_
centroids = kmeans.cluster_centers_

plt.figure(figsize=(8, 6))
plt.scatter(data[:, 0], data[:, 1], c=labels, cmap='viridis', edgecolors='k')
plt.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='x', s=200, label='Centroides')
plt.title("Clustering con K-Means")
plt.xlabel("Eje X")
plt.ylabel("Eje Y")
plt.legend()
plt.show()