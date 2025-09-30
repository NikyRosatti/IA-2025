import numpy as np
import matplotlib.pyplot as plt

# ----------------------------
# Dataset sintético (tu código)
# ----------------------------
np.random.seed(42)
n_samples = 200

lager_ibu = np.random.normal(loc=28, scale=4, size=n_samples//2)
lager_rms = np.random.normal(loc=24, scale=6, size=n_samples//2)
lager = np.column_stack((lager_ibu, lager_rms))
lager_y = np.zeros(n_samples//2)

stout_ibu = np.random.normal(loc=46, scale=5, size=n_samples//2)
stout_rms = np.random.normal(loc=58, scale=6, size=n_samples//2)
stout = np.column_stack((stout_ibu, stout_rms))
stout_y = np.ones(n_samples//2)

X = np.vstack((lager, stout))
y = np.concatenate((lager_y, stout_y))

indices = np.arange(n_samples)
np.random.shuffle(indices)
X = X[indices]
y = y[indices]

split = int(0.8 * n_samples)
X_train, X_val = X[:split], X[split:]
y_train, y_val = y[:split], y[split:]

# ----------------------------
# Funciones auxiliares
# ----------------------------
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def modelo(X, Y, learning_rate, iterations):
    X = X.T
    n = X.shape[0]
    m = X.shape[1]
    W = np.zeros((n,1))
    B = 0.0
    Y = Y.reshape(1, m)

    for i in range(iterations):
        Z = np.dot(W.T, X) + B
        A = sigmoid(Z)
        costo = -(1/m)*np.sum(Y*np.log(A+1e-8) + (1-Y)*np.log(1-A+1e-8))

        # gradientes CORREGIDOS (1/m en vez de 1/n)
        dW = (1/m) * np.dot(A - Y, X.T)
        dB = (1/m) * np.sum(A - Y)

        W = W - learning_rate * dW.T
        B = B - learning_rate * dB

        if i % (iterations//10) == 0:
            print(f"Costo luego de iteración {i}: {costo:.4f}")

    return W, B

def predict(X, W, B):
    Z = np.dot(W.T, X.T) + B
    A = sigmoid(Z)
    return (A >= 0.5).astype(int).flatten()

# ----------------------------
# Entrenamiento
# ----------------------------
W, B = modelo(X_train, y_train, learning_rate=0.01, iterations=1000)

# ----------------------------
# Validación
# ----------------------------
y_pred = predict(X_val, W, B)
accuracy = np.mean(y_pred == y_val)
print("\nAccuracy en validación:", accuracy)

# ----------------------------
# Visualización frontera
# ----------------------------
# Creamos una malla de puntos
x_min, x_max = X[:,0].min() - 2, X[:,0].max() + 2
y_min, y_max = X[:,1].min() - 2, X[:,1].max() + 2
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                     np.linspace(y_min, y_max, 200))

# Predicciones en la malla
Z = predict(np.c_[xx.ravel(), yy.ravel()], W, B)
Z = Z.reshape(xx.shape)

plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.RdBu)
plt.scatter(X_train[y_train==0][:,0], X_train[y_train==0][:,1], c="blue", label="Lager (train)")
plt.scatter(X_train[y_train==1][:,0], X_train[y_train==1][:,1], c="red", label="Stout (train)")
plt.scatter(X_val[y_val==0][:,0], X_val[y_val==0][:,1], c="cyan", marker="x", label="Lager (val)")
plt.scatter(X_val[y_val==1][:,0], X_val[y_val==1][:,1], c="orange", marker="x", label="Stout (val)")
plt.xlabel("IBU")
plt.ylabel("RMS")
plt.legend()
plt.title("Frontera de decisión Regresión Logística")
plt.show()

X_prueba = np.array([
    [15, 20], # Lager
    [12, 15],
    [28, 39],
    [21, 30],
    [45, 20], # Stout
    [40, 61],
    [42, 70]
])

y_prueba = predict(X_prueba, W, B)
print("Predicciones:", y_prueba)
