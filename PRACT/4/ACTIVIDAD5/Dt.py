import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# 1. Generar dataset
X, y = make_moons(n_samples=100, noise=0.35, random_state=42)

# Dividir en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 2. Crear y entrenar árboles con diferentes técnicas de pruning

# a) Árbol sin poda
tree_no_prune = DecisionTreeClassifier(random_state=42)
tree_no_prune.fit(X_train, y_train)

# b) Árbol con poda mínima de hojas (min_samples_leaf)
tree_min_leaf = DecisionTreeClassifier(random_state=42, min_samples_leaf=5)
tree_min_leaf.fit(X_train, y_train)

# c) Árbol con profundidad máxima limitada (max_depth)
tree_max_depth = DecisionTreeClassifier(random_state=42, max_depth=3)
tree_max_depth.fit(X_train, y_train)

# 3. Evaluación
models = {
    "No Pruning": tree_no_prune,
    "Min Samples Leaf=5": tree_min_leaf,
    "Max Depth=3": tree_max_depth
}

for name, model in models.items():
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"{name} - Accuracy: {acc:.2f}")
    print(classification_report(y_test, y_pred))

# 4. Visualización de un árbol (ejemplo: max_depth=3)
plt.figure(figsize=(12,8))
plot_tree(tree_no_prune, feature_names=['X1', 'X2'], class_names=['0','1'], filled=True, rounded=True)
plt.show()

# 5. Consulta por un caso
# Ejemplo de un punto nuevo
new_point = np.array([[0.5, 0.2]])
predicted_class = tree_no_prune.predict(new_point)
print(f"Predicción para el punto {new_point[0]}: {predicted_class[0]}")

# También podemos ver las reglas de decisión para entender la predicción
from sklearn.tree import export_text
rules = export_text(tree_min_leaf, feature_names=['X1', 'X2'])
print("Reglas del árbol:\n")
print(rules)



# Graficar dataset
plt.figure(figsize=(8,6))
plt.scatter(X[y==0][:,0], X[y==0][:,1], color='red', label='Clase 0', edgecolor='k', s=80)
plt.scatter(X[y==1][:,0], X[y==1][:,1], color='blue', label='Clase 1', edgecolor='k', s=80)
plt.xlabel("X1")
plt.ylabel("X2")
plt.title("Dataset Make Moons con Ruido")
plt.legend()
plt.grid(True)
plt.show()