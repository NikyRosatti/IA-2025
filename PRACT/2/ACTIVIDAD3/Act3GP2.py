import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

# 1. Generar dataset grande (200 puntos)
np.random.seed(42)
n_samples = 200
X = np.linspace(0, 10, n_samples).reshape(-1, 1)
y = (X**2).ravel() + np.random.randn(n_samples) * 8   # x^2 + ruido

# 2. Separar en entrenamiento (70%) y validación (30%)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=42)

# 3. Probar distintos grados de polinomialidad
degrees = [1, 2, 5, 10]
train_errors, val_errors = [], []

plt.figure(figsize=(12, 8))

for i, degree in enumerate(degrees, 1):
    # Generar características polinomiales
    poly = PolynomialFeatures(degree=degree)
    X_train_poly = poly.fit_transform(X_train)
    X_val_poly = poly.transform(X_val)

    # Ajustar modelo
    model = LinearRegression()
    model.fit(X_train_poly, y_train)

    # Predicciones
    y_train_pred = model.predict(X_train_poly)
    y_val_pred = model.predict(X_val_poly)

    # Calcular errores
    train_mse = mean_squared_error(y_train, y_train_pred)
    val_mse = mean_squared_error(y_val, y_val_pred)
    train_errors.append(train_mse)
    val_errors.append(val_mse)

    print(f"Grado {degree}: Train MSE={train_mse:.2f}, Validation MSE={val_mse:.2f}")

    # 4. Graficar cada modelo
    X_plot = np.linspace(0, 10, 200).reshape(-1, 1)
    X_plot_poly = poly.transform(X_plot)
    y_plot = model.predict(X_plot_poly)

    plt.subplot(2, 2, i)
    plt.scatter(X_train, y_train, color="blue", s=10, label="Train")
    plt.scatter(X_val, y_val, color="orange", s=10, label="Validation")
    plt.plot(X_plot, y_plot, color="red", linewidth=2, label=f"Modelo grado {degree}")
    plt.title(f"Grado {degree}")
    plt.legend()

plt.tight_layout()
plt.show()

# 5. Curvas de error (bias-variance tradeoff)
plt.figure(figsize=(8, 5))
plt.plot(degrees, train_errors, marker="o", label="Train MSE")
plt.plot(degrees, val_errors, marker="o", label="Validation MSE")
plt.xlabel("Grado polinomial")
plt.ylabel("MSE")
plt.title("Errores de entrenamiento vs validación")
plt.legend()
plt.show()
