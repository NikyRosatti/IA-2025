import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score

# ===============================
# 1. Generar dataset ruidoso
# ===============================
X, y = make_moons(n_samples=1000, noise=0.35, random_state=42)

# Dividir en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# ===============================
# 2. Entrenar Decision Tree
# ===============================
dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train, y_train)
y_pred_dt = dt_model.predict(X_test)

# Matriz de confusión DT
cm_dt = confusion_matrix(y_test, y_pred_dt)
disp_dt = ConfusionMatrixDisplay(confusion_matrix=cm_dt)
print("Accuracy DT:", accuracy_score(y_test, y_pred_dt))
disp_dt.plot(cmap=plt.cm.Blues)
plt.title("Decision Tree - Confusion Matrix")
plt.show()

# ===============================
# 3. Entrenar Random Forest (Bagging)
# ===============================
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)

# Matriz de confusión RF
cm_rf = confusion_matrix(y_test, y_pred_rf)
disp_rf = ConfusionMatrixDisplay(confusion_matrix=cm_rf)
print("Accuracy RF:", accuracy_score(y_test, y_pred_rf))
disp_rf.plot(cmap=plt.cm.Greens)
plt.title("Random Forest - Confusion Matrix")
plt.show()

# ===============================
# 4. Opcional: visualizar árbol individual de DT
# ===============================
plt.figure(figsize=(12,8))
plot_tree(rf_model, feature_names=['X1','X2'], class_names=['0','1'], filled=True, rounded=True)
plt.show()
