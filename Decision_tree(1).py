from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt

X, y = load_iris(return_X_y=True)

tree = DecisionTreeClassifier(criterion='gini', max_depth=3)
tree.fit(X, y)

plt.figure(figsize=(10, 6))
plot_tree(tree, filled=True, feature_names=load_iris().feature_names, class_names=load_iris().target_names)
plt.title("Árbol de Decisión (Iris)")
plt.show()

