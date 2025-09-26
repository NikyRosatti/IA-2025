import numpy as np
import matplotlib.pyplot as plt

def perceptron_train(X, y, eta=0.1, epochs=20):
    w = np.zeros(X.shape[1])
    b = 0
    
    def step(z):
        return 1 if z >= 0 else 0
    
    for _ in range(epochs):
        for xi, target in zip(X, y):
            z = np.dot(w, xi) + b
            y_pred = step(z)
            error = target - y_pred
            w += eta * error * xi
            b += eta * error
    return w, b, step

def plot_decision_boundary(X, y, w, b):
    plt.figure(figsize=(6,6))
    # Puntos
    plt.scatter(X[y==0, 0], X[y==0, 1], color="red", label="Clase 0")
    plt.scatter(X[y==1, 0], X[y==1, 1], color="blue", label="Clase 1")
    
    # Recta de decisión
    if w[1] != 0:
        x_vals = np.linspace(min(X[:,0])-1, max(X[:,0])+1, 100)
        y_vals = -(w[0]*x_vals + b)/w[1]
        plt.plot(x_vals, y_vals, 'k--', label="Frontera")
    
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.legend()
    plt.grid(True)
    plt.title("Perceptrón - Clasificación")
    plt.show()

# Dataset del enunciado
X = np.array([[0,0],
              [0,1],
              [1,0],
              [2,2],
              [2,3],
              [3,2],
              [3,3]])
y = np.array([0,0,0,1,1,1,0])

# Entrenamiento
w, b, step = perceptron_train(X, y)

print("Pesos finales:", w)
print("Bias final:", b)

# Graficar frontera
plot_decision_boundary(X, y, w, b)
for p in X:
    print(f"{p} -> {step(np.dot(w, p) + b)}")
# Probar con puntos nuevos
test_points = np.array([[1,2], [2,0], [3,3]])
print("\nPruebas con puntos nuevos:")
for p in test_points:
    print(f"{p} -> {step(np.dot(w, p) + b)}")
