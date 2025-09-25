import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error

# --------------------------
# 1) Generar dataset cuadrático
# --------------------------
np.random.seed(42)
n_samples = 20
X = np.linspace(0, 10, n_samples).reshape(-1, 1)
y = (X**2).ravel() + np.random.randn(n_samples) * 5

# --------------------------
# 2) Regresión lineal simple
# --------------------------
lin_reg = LinearRegression()
lin_reg.fit(X, y)
y_pred_lin = lin_reg.predict(X)
mse_lin = mean_squared_error(y, y_pred_lin)
print(f"MSE - Regresión Lineal: {mse_lin:.4f}")

# --------------------------
# 3) Transformación polinomial
# --------------------------
degree = 2
poly = PolynomialFeatures(degree=degree)
X_poly = poly.fit_transform(X)
print(f"\nDataset transformado (grado {degree}):\n", X_poly[:5])  # primeros 5 ejemplos

# --------------------------
# 4) Ajuste de regresión polinomial
# --------------------------
poly_reg = LinearRegression()
poly_reg.fit(X_poly, y)
y_pred_poly = poly_reg.predict(X_poly)
mse_poly = mean_squared_error(y, y_pred_poly)
print(f"MSE - Regresión Polinomial grado {degree}: {mse_poly:.4f}")

# --------------------------
# 5) Graficar comparación
# --------------------------
plt.scatter(X, y, color="blue", label="Datos (x^2 + ruido)")
plt.plot(X, y_pred_lin, color="red", label=f"Lineal MSE={mse_lin:.2f}")
plt.plot(X, y_pred_poly, color="green", linestyle="--", label=f"Polinomial grado {degree} MSE={mse_poly:.2f}")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.title("Regresión Lineal vs Polinomial")
plt.show()

# --------------------------
# 6) Dataset transformado para más de 2 grados
# --------------------------
for d in [2, 3, 4]:
    poly = PolynomialFeatures(degree=d)
    X_poly = poly.fit_transform(X)
    print(f"\nPrimeros 5 ejemplos transformados - grado {d}:\n", X_poly[:5])
