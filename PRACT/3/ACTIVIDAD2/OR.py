import numpy as np
# Dataset
X = np.array([[0,0],
              [0,1],
              [1,0],
              [1,1]])
y = np.array([0,1,1,1]) # Clase
# Hiperparámetros
eta = 1 # learning rate
epochs = 400 # cantidad de épocas para el entrenamiento
w = np.array([0, 0]) # pesos iniciales
b = 0
def step(z):
    return 0 if z >= 0 else 1

# Entrenamiento
for epoch in range(epochs):
    for i in range(len(X)):
        xi = X[i]
        target = y[i]
        z = np.dot(w, xi) + b
        y_pred = step(z)
        error = target - y_pred
        # Actualización
        w = w + eta * error * xi
        b = b + eta * error
        
print("Pesos finales:", w)
print("Bias final:", b)

# Probar la función final
print("\nPruebas del OR aprendido:")
for xi in X:
    print(f"{xi} -> {step(np.dot(w, xi) + b)}")