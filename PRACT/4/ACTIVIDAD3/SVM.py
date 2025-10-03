import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# ----------------------------
# 1. Dataset sintético
# ----------------------------
X, y = make_moons(n_samples=300, noise=0.35, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# ----------------------------
# 2. Función para entrenar y graficar SVM
# ----------------------------
def plot_svm(kernel, C):
    model = SVC(kernel=kernel, C=C, gamma="scale")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)

    # Frontera de decisión
    h = 0.02
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.contourf(xx, yy, Z, alpha=0.3)
    plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, edgecolor="k", marker="o", label="train")
    plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, edgecolor="k", marker="s", label="test")
    plt.title(f"SVM kernel={kernel}, C={C}, Acc={acc:.2f}")
    plt.legend()
    plt.show()

    print(f"\nResultados con kernel={kernel}, C={C}")
    print("Accuracy:", acc)
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))


# ----------------------------
# 3. Probar distintos kernels y C
# ----------------------------
configs = [
    ("linear", 0.1),
    ("linear", 10),
    ("rbf", 0.1),
    ("rbf", 10),
    ("poly", 3),  # grado=3 por defecto
]

for kernel, C in configs:
    plot_svm(kernel, C)
