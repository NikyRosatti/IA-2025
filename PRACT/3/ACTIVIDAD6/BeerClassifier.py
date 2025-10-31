import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsOneClassifier
from sklearn.metrics import classification_report, accuracy_score, ConfusionMatrixDisplay


# Generación del Dataset Sintético
print("Generando dataset sintético...")
np.random.seed(42)

# 100 por cada una de las 4 clases
n_samples_per_class = 100

# 400 muestras
n_samples = 400 

# Clase 0: Lager (menor IBU y RMS)
lager_ibu = np.random.normal(loc=15, scale=5, size=n_samples_per_class)
lager_rms = np.random.normal(loc=20, scale=5, size=n_samples_per_class)
lager_X = np.column_stack((lager_ibu, lager_rms))
lager_y = np.zeros(n_samples_per_class)

# Clase 1: Stout (Alto IBU y RMS)
stout_ibu = np.random.normal(loc=40, scale=6, size=n_samples_per_class)
stout_rms = np.random.normal(loc=60, scale=6, size=n_samples_per_class)
stout_X = np.column_stack((stout_ibu, stout_rms))
stout_y = np.ones(n_samples_per_class)

# Clase 2: IPA (amarga y no tan oscura Alto IBU, Bajo RMS)
ipa_ibu = np.random.normal(loc=50, scale=6, size=n_samples_per_class)
ipa_rms = np.random.normal(loc=25, scale=5, size=n_samples_per_class)
ipa_X = np.column_stack((ipa_ibu, ipa_rms))
ipa_y = np.full(n_samples_per_class, 2)

# Clase 3: Scottish (ligeramente oscura y no tan amarga Bajo IBU, Medio RMS)
scottish_ibu = np.random.normal(loc=20, scale=5, size=n_samples_per_class)
scottish_rms = np.random.normal(loc=40, scale=6, size=n_samples_per_class)
scottish_X = np.column_stack((scottish_ibu, scottish_rms))
scottish_y = np.full(n_samples_per_class, 3)

# Nombres de las clases para los gráficos
class_names = ['Lager', 'Stout', 'IPA', 'Scottish']

# Concatenamos dataset completo
X = np.vstack((lager_X, stout_X, ipa_X, scottish_X))
y = np.concatenate((lager_y, stout_y, ipa_y, scottish_y)).astype(int)

# Mezclamos aleatoriamente
indices = np.arange(n_samples)
np.random.shuffle(indices)
X = X[indices]
y = y[indices]

# Split 80% entrenamiento, 20% validacion
split = int(0.8 * n_samples)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

print(f"Dataset generado: {n_samples} muestras totales.")
print(f"Train: {X_train.shape[0]} muestras, Test: {X_test.shape[0]} muestras.")



# Funciones Auxiliares para Evaluación y Gráficos
def plot_decision_boundary(pipeline, X, y, class_names, title):
    """ 
    Dibuja la frontera de decisión de un pipeline de sklearn 
    """
    cmap_light = ListedColormap(['#AAAAFF', '#FFAAAA', '#AAFFAA', '#CCAACC'])
    cmap_bold = ListedColormap(['#0000FF', '#FF0000', '#00FF00', '#880088'])

    h = .02
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    # El pipeline escala la malla (xx, yy) automáticamente
    Z = pipeline.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    plt.figure(figsize=(10, 7))
    plt.contourf(xx, yy, Z, cmap=cmap_light, alpha=0.8)
    scatter = plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold, edgecolor='k', s=20)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.title(title)
    plt.xlabel("IBU (Amargor)")
    plt.ylabel("RMS (Oscuridad)")
    
    handles, _ = scatter.legend_elements()
    plt.legend(handles=handles, labels=class_names)
    plt.grid(True, linestyle='--', alpha=0.5)

def train_and_evaluate(pipeline, X_train, y_train, X_test, y_test, class_names, model_name):
    """ 
    Entrena, Evalúa y Grafica un pipeline
    """
    print(f"\n" + "="*50)
    print(f"Entrenando Modelo: {model_name}")
    print("="*50)
    
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.4f}")
    print("\nReporte de Clasificación (Test Set):")
    print(classification_report(y_test, y_pred, target_names=class_names))
    
    fig, ax = plt.subplots(figsize=(8, 6))
    ConfusionMatrixDisplay.from_estimator(
        pipeline,
        X_test,
        y_test,
        display_labels=class_names,
        cmap=plt.cm.Blues,
        ax=ax
    )
    ax.set_title(f"Matriz de Confusión - {model_name}")
    
    print(f"Graficando Frontera de Decisión para {model_name}...")
    plot_decision_boundary(
        pipeline,
        X_train, # Graficamos sobre el train set
        y_train,
        class_names,
        title=f"Frontera de Decisión - {model_name} (Train Set)"
    )

# Clasificación OVR (One-vs-Rest)
pipeline_ovr = Pipeline([
    ('scaler', StandardScaler()),
    ('model', LogisticRegression(multi_class='ovr', solver='liblinear', random_state=42))
])
train_and_evaluate(pipeline_ovr, X_train, y_train, X_test, y_test, class_names, "OVR (One-vs-Rest)")

# Clasificación OVO (One-vs-One)
base_model_ovo = LogisticRegression(solver='liblinear', random_state=42)
pipeline_ovo = Pipeline([
    ('scaler', StandardScaler()),
    ('model', OneVsOneClassifier(base_model_ovo))
])
train_and_evaluate(pipeline_ovo, X_train, y_train, X_test, y_test, class_names, "OVO (One-vs-One)")

# d) Clasificación Softmax (Multinomial)
pipeline_softmax = Pipeline([
    ('scaler', StandardScaler()),
    ('model', LogisticRegression(multi_class='multinomial', solver='lbfgs', random_state=42, C=1.0))
])
train_and_evaluate(pipeline_softmax, X_train, y_train, X_test, y_test, class_names, "Softmax (Multinomial, C=1.0)")


# Evaluación con Hiperparámetros (Softmax)
# C es el inverso de la fuerza de regularización.
# C=0.1 = Regularización Fuerte (modelo más simple)
# C=10.0 = Regularización Débil (modelo más complejo)

# Modelo con Regularización Fuerte C=0.1
pipeline_softmax_c_low = Pipeline([
    ('scaler', StandardScaler()),
    ('model', LogisticRegression(multi_class='multinomial', solver='lbfgs', random_state=42, C=0.1))
])
train_and_evaluate(pipeline_softmax_c_low, X_train, y_train, X_test, y_test, class_names, "Softmax (Multinomial, C=0.1)")

# Modelo con Regularización Débil C=10.0
pipeline_softmax_c_high = Pipeline([
    ('scaler', StandardScaler()),
    ('model', LogisticRegression(multi_class='multinomial', solver='lbfgs', random_state=42, C=10.0))
])
train_and_evaluate(pipeline_softmax_c_high, X_train, y_train, X_test, y_test, class_names, "Softmax (Multinomial, C=10.0)")

# Mostrar todos los gráficos al final
print("\nMostrando todos los gráficos generados...")
plt.show()