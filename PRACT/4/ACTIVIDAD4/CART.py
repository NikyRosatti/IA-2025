import pandas as pd
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt

# Datos
data = {
    "Horas": ["Baja","Baja","Media","Alta","Media","Alta","Baja","Alta","Media","Alta"],
    "Asistencia": ["Baja","Alta","Alta","Alta","Baja","Alta","Baja","Baja","Alta","Alta"],
    "Tareas": ["No","No","No","Si","Si","No","Si","Si","Si","Si"],
    "Resultado": ["Reprobado","Reprobado","Aprobado","Promoción","Aprobado",
                  "Aprobado","Reprobado","Aprobado","Promoción","Promoción"]
}

df = pd.DataFrame(data)

# Codificación categórica
X = pd.get_dummies(df[["Horas","Asistencia","Tareas"]])
y = df["Resultado"]

# Entrenar árbol CART con criterio Gini
clf = DecisionTreeClassifier(criterion="gini", random_state=42)
clf.fit(X, y)

# Graficar árbol
plt.figure(figsize=(10,6))
plot_tree(clf, feature_names=X.columns, class_names=clf.classes_, filled=True)
plt.show()
