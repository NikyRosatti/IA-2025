import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import StackingClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay

# ===============================
# 1. Cargar dataset Iris
# ===============================
iris = load_iris()
X = iris.data
y = iris.target

# Dividir en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# ===============================
# 2. Definir clasificadores base
# ===============================
base_estimators = [
    ('dt', DecisionTreeClassifier(max_depth=3, random_state=42)),
    ('knn', KNeighborsClassifier(n_neighbors=5)),
    ('lr', LogisticRegression(max_iter=200, random_state=42))
]

# ===============================
# 3. Stacking con distintos meta-modelos
# ===============================

# a) Meta-modelo: Logistic Regression
stack_lr = StackingClassifier(estimators=base_estimators, final_estimator=LogisticRegression(), cv=5)
stack_lr.fit(X_train, y_train)
y_pred_lr = stack_lr.predict(X_test)
print("Accuracy Stacking (meta=LR):", accuracy_score(y_test, y_pred_lr))
cm_lr = confusion_matrix(y_test, y_pred_lr)
ConfusionMatrixDisplay(cm_lr, display_labels=iris.target_names).plot()
plt.title("StackingClassifier - Meta: LogisticRegression")
plt.show()

# b) Meta-modelo: Decision Tree
stack_dt = StackingClassifier(estimators=base_estimators, final_estimator=DecisionTreeClassifier(max_depth=3), cv=5)
stack_dt.fit(X_train, y_train)
y_pred_dt = stack_dt.predict(X_test)
print("Accuracy Stacking (meta=DT):", accuracy_score(y_test, y_pred_dt))
cm_dt = confusion_matrix(y_test, y_pred_dt)
ConfusionMatrixDisplay(cm_dt, display_labels=iris.target_names).plot()
plt.title("StackingClassifier - Meta: DecisionTree")
plt.show()

# c) Meta-modelo: KNN
stack_knn = StackingClassifier(estimators=base_estimators, final_estimator=KNeighborsClassifier(n_neighbors=3), cv=5)
stack_knn.fit(X_train, y_train)
y_pred_knn = stack_knn.predict(X_test)
print("Accuracy Stacking (meta=KNN):", accuracy_score(y_test, y_pred_knn))
cm_knn = confusion_matrix(y_test, y_pred_knn)
ConfusionMatrixDisplay(cm_knn, display_labels=iris.target_names).plot()
plt.title("StackingClassifier - Meta: KNN")
plt.show()
