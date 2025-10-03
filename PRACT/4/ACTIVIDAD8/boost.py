import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score

# ===============================
# 1. Dataset ruidoso
# ===============================
X, y = make_moons(n_samples=1000, noise=0.35, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# ===============================
# 2. Boosting con Decision Tree
# ===============================
dt_base = DecisionTreeClassifier(max_depth=1, random_state=42)  # árbol débil
ada_dt = AdaBoostClassifier(estimator=dt_base, n_estimators=50, random_state=42)
ada_dt.fit(X_train, y_train)
y_pred_ada_dt = ada_dt.predict(X_test)

cm_ada_dt = confusion_matrix(y_test, y_pred_ada_dt)
disp_ada_dt = ConfusionMatrixDisplay(cm_ada_dt)
print("Accuracy AdaBoost con DT:", accuracy_score(y_test, y_pred_ada_dt))
disp_ada_dt.plot(cmap=plt.cm.Blues)
plt.title("AdaBoost + Decision Tree")
plt.show()

# ===============================
# 3. Boosting con KNN
# ===============================
knn_base = KNeighborsClassifier(n_neighbors=3)
ada_knn =knn_base
ada_knn.fit(X_train, y_train)
y_pred_ada_knn = ada_knn.predict(X_test)

cm_ada_knn = confusion_matrix(y_test, y_pred_ada_knn)
disp_ada_knn = ConfusionMatrixDisplay(cm_ada_knn)
print("Accuracy AdaBoost con KNN:", accuracy_score(y_test, y_pred_ada_knn))
disp_ada_knn.plot(cmap=plt.cm.Oranges)
plt.title("AdaBoost + KNN")
plt.show()
