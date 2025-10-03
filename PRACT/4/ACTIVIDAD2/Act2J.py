import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# ----------------------------
# 1. Generamos dataset sintético con ruido
# ----------------------------
X, y = make_moons(n_samples=100, noise=0.35, random_state=42)

# Dividir en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# ----------------------------
# 2. Función para entrenar y graficar frontera de decisión
# ----------------------------
def plot_knn(k):
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)

    # Graficar frontera
    h = .02
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.contourf(xx, yy, Z, alpha=0.3)
    plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, edgecolor="k", marker="o", label="train")
    plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, edgecolor="k", marker="s", label="test")
    plt.title(f"K={k}, Acc={acc:.2f}")
    plt.legend()
    plt.show()

    print(f"Resultados con K={k}")
    print("Accuracy:", acc)
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))


# ----------------------------
# 3. Probar distintos k
# ----------------------------
for k in [1, 3, 5, 15, 50]:
    plot_knn(k)
