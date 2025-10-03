import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# ===============================
# 1. Dataset ruidoso
# ===============================
X, y = make_moons(n_samples=100, noise=0.35, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# ===============================
# 2. Entrenar modelos
# ===============================
dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train, y_train)

rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# ===============================
# 3. Visualización de fronteras
# ===============================
# Crear malla de puntos
x_min, x_max = X[:,0].min() - 1, X[:,0].max() + 1
y_min, y_max = X[:,1].min() - 1, X[:,1].max() + 1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300),
                     np.linspace(y_min, y_max, 300))

# Predicciones para malla
Z_dt = dt_model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
Z_rf = rf_model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

# ===============================
# 4. Graficar
# ===============================
fig, axes = plt.subplots(1, 2, figsize=(14,6))

# Decision Tree
axes[0].contourf(xx, yy, Z_dt, alpha=0.3, cmap=plt.cm.RdYlBu)
axes[0].scatter(X[:,0], X[:,1], c=y, cmap=plt.cm.RdYlBu, edgecolor='k')
axes[0].set_title("Decision Tree")
axes[0].set_xlabel("X1")
axes[0].set_ylabel("X2")

# Random Forest
axes[1].contourf(xx, yy, Z_rf, alpha=0.3, cmap=plt.cm.RdYlBu)
axes[1].scatter(X[:,0], X[:,1], c=y, cmap=plt.cm.RdYlBu, edgecolor='k')
axes[1].set_title("Random Forest (Bagging)")
axes[1].set_xlabel("X1")
axes[1].set_ylabel("X2")

plt.tight_layout()
plt.show()
