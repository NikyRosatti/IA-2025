import numpy as np
import matplotlib.pyplot as plt

# ---------------------------
# Parámetros
# ---------------------------
np.random.seed(42)   # reproducibilidad
n_samples = 100
true_w = 3.0
true_b = 2.0

# ---------------------------
# 1) Generar datos sintéticos
# ---------------------------
x = np.linspace(0, 10, n_samples)
e = np.random.normal(loc=0.0, scale=1.0, size=n_samples)
y = true_w * x + true_b + e

# convertir a columnas
x_col = x.reshape(-1, 1)
y_col = y.reshape(-1, 1)
m = n_samples

# Matriz con bias
X_b = np.c_[np.ones((m, 1)), x_col]  # (m x 2)

# ---------------------------
# 2) Ecuación Normal
# ---------------------------
# theta_normal = (X^T X)^{-1} X^T y  -- usar pseudo-inversa
theta_normal = np.linalg.pinv(X_b.T @ X_b) @ X_b.T @ y_col
b_norm, w_norm = theta_normal[0,0], theta_normal[1,0]
print("Ecuación Normal: y = {:.4f} x + {:.4f}".format(w_norm, b_norm))

# ---------------------------
# 3) Descenso por Gradiente (registrando pérdida)
# ---------------------------
def compute_mse(X_b, y, theta):
    m = len(y)
    preds = X_b @ theta
    return (1/(2*m)) * np.sum((preds - y)**2)

def gradient_descent(X_b, y, eta=0.01, n_iter=1000, theta_init=None):
    m, n = X_b.shape
    if theta_init is None:
        theta = np.zeros((n, 1))
    else:
        theta = theta_init.copy()
    losses = []
    for it in range(n_iter):
        gradients = (1/m) * X_b.T @ (X_b @ theta - y)   # gradiente MSE
        theta = theta - eta * gradients
        loss = compute_mse(X_b, y, theta)
        losses.append(loss)
    return theta, losses

eta = 0.05
n_iter = 20
theta_init = np.zeros((2,1))
theta_gd, losses = gradient_descent(X_b, y_col, eta=eta, n_iter=n_iter, theta_init=theta_init)
b_gd, w_gd = theta_gd[0,0], theta_gd[1,0]
print("Gradiente:       y = {:.4f} x + {:.4f}".format(w_gd, b_gd))

# ---------------------------
# 4) Comparación numérica
# ---------------------------
mse_normal = compute_mse(X_b, y_col, theta_normal)
mse_gd = compute_mse(X_b, y_col, theta_gd)
print(f"MSE (Ecuación Normal) = {mse_normal:.6f}")
print(f"MSE (Gradiente Desc.) = {mse_gd:.6f}")
print("Diferencias en theta (GD - Normal):", (theta_gd - theta_normal).ravel())

# ---------------------------
# 5) Gráficas
# ---------------------------

# a) Datos y rectas
plt.figure(figsize=(8,5))
plt.scatter(x, y, color="blue", label="Datos (sintéticos)")
x_line = np.array([x.min(), x.max()])
plt.plot(x_line, w_norm*x_line + b_norm, color="red", label=f"Ecuación Normal: y={w_norm:.2f}x+{b_norm:.2f}")
plt.plot(x_line, w_gd*x_line + b_gd, color="green", linestyle="--", label=f"Gradiente: y={w_gd:.2f}x+{b_gd:.2f}")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.title("Ajuste por Ecuación Normal vs Gradiente Descendente")
plt.grid(True)
plt.show()

# b) Convergencia del error
plt.figure(figsize=(8,4))
plt.plot(range(1, n_iter+1), losses)
plt.xlabel("Iteración")
plt.ylabel("Pérdida (MSE/2)")
plt.title("Convergencia del error (Gradiente Descendente)")
plt.grid(True)
plt.show()

# Opcional: Histogramas de residuos
resid_normal = (X_b @ theta_normal - y_col).ravel()
resid_gd = (X_b @ theta_gd - y_col).ravel()

plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
plt.hist(resid_normal, bins=20, alpha=0.7)
plt.title("Residuos - Ecuación Normal")
plt.subplot(1,2,2)
plt.hist(resid_gd, bins=20, alpha=0.7)
plt.title("Residuos - Gradiente Descendente")
plt.tight_layout()
plt.show()
