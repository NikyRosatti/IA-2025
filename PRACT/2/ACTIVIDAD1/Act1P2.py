import numpy as np
import matplotlib.pyplot as plt

def gradient_descent(X_b, y, eta=0.01, n_iter=1000):
    m = len(y)
    theta = np.zeros((2, 1))  # [b, w]
    for iteration in range(n_iter):
        gradients = (1/m) * X_b.T @ (X_b @ theta - y)
        theta = theta - eta * gradients
    return theta

# =========================
# Datos originales
# =========================
x = np.array([1, 2, 3, 4, 5])
y = np.array([2, 3, 5, 7, 8]).reshape(-1, 1)
m = len(x)

# Matriz con bias
X_b = np.c_[np.ones((m, 1)), x.reshape(-1, 1)]

# Entrenar con descenso por gradiente
theta = gradient_descent(X_b, y, eta=0.01, n_iter=1000)
b, w = theta[0, 0], theta[1, 0]

print(f"Sin outlier: y = {w:.2f}x + {b:.2f}")

# Predicciones
y_pred = X_b @ theta

# =========================
# Datos con un outlier
# =========================
x_out = np.array([1, 2, 3, 4, 5, 10])
y_out = np.array([2, 3, 5, 7, 8, 30]).reshape(-1, 1)
m_out = len(x_out)

X_b_out = np.c_[np.ones((m_out, 1)), x_out.reshape(-1, 1)]

# Entrenar con descenso por gradiente
theta_out = gradient_descent(X_b_out, y_out, eta=0.01, n_iter=1000)
b_out, w_out = theta_out[0, 0], theta_out[1, 0]

print(f"Con outlier: y = {w_out:.2f}x + {b_out:.2f}")

# Predicciones con outlier
y_pred_out = X_b_out @ theta_out

# =========================
# Graficar
# =========================
plt.scatter(x, y, color="blue", label="Datos originales")
plt.plot(x, y_pred, color="red", label=f"Sin outlier: y={w:.2f}x+{b:.2f}")

plt.scatter(10, 30, color="purple", marker="x", s=100, label="Outlier")
plt.plot(x_out, y_pred_out, color="green", linestyle="--", label=f"Con outlier: y={w_out:.2f}x+{b_out:.2f}")

plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.show()
