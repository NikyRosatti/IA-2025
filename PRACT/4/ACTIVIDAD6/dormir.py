import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# ===============================
# a) Cargar dataset y particionar
# ===============================

df = pd.read_csv("ScreenTime vs MentalWellness.csv")  # Ajusta el nombre de tu archivo CSV

# Quitar columna innecesaria (si existe)
if 'user_id' in df.columns:
    df = df.drop(columns=['user_id'])
    
df = df.drop(columns=["Unnamed: 15"])

# Definir variable objetivo y features
# Supongamos que vamos a predecir "sleep_quality_1_5" como categórica
# Convertimos en binaria: baja calidad <=2, alta >2
df['sleep_quality_label'] = (df['sleep_quality_1_5'] <= 2).astype(int)

X = df.drop(columns=['sleep_quality_1_5', 'sleep_quality_label'])
y = df['sleep_quality_label']

# Codificar variables categóricas
X = pd.get_dummies(X, drop_first=True)

# Dividir en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ===============================
# b) Previsualizar info de entrenamiento
# ===============================
print("Primeros 5 registros de X_train:")
print(X_train.head())

print("\nDescripción estadística de X_train:")
print(X_train.describe())

# Distribución de la clase objetivo
sns.countplot(x=y_train)
plt.title("Distribución de calidad de sueño (entrenamiento)")
plt.show()

# ===============================
# c) Entrenar y evaluar modelos
# ===============================

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42)
}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    print(f"\nModelo: {name}")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))

# ===============================
# d) Características comunes de personas con baja calidad de sueño
# ===============================

# Separar datasets
good_sleep = df[df['sleep_quality_label'] == 0]
bad_sleep  = df[df['sleep_quality_label'] == 1]

# Variables a comparar
variables = ['screen_time_hours', 'work_screen_hours', 'leisure_screen_hours', 'sleep_hours', 'stress_level_0_10']

# Crear histogramas comparativos
fig, axes = plt.subplots(len(variables), 1, figsize=(10, 5*len(variables)))
for i in range(0, len(variables), 2):
    fig, axes = plt.subplots(min(2, len(variables)-i), 1, figsize=(10,6))
    if min(2, len(variables)-i) == 1:
        axes = [axes]
    for j, var in enumerate(variables[i:i+2]):
        sns.histplot(good_sleep[var], color='green', label='Buena calidad', kde=True, ax=axes[j], alpha=0.6)
        sns.histplot(bad_sleep[var], color='red', label='Mala calidad', kde=True, ax=axes[j], alpha=0.6)
        axes[j].set_title(f'Comparación de {var}')
        axes[j].legend()
    plt.tight_layout()
    plt.show()
