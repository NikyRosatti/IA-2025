import numpy as np
import matplotlib.pyplot as plt
# Dataset
X = np.array([[0,0],
              [0,1],
              [1,0],
              [2,2],
              [2,3],
              [3,2]])
y = np.array([0,0,0,1,1,1]) # Clase
# Hiperparámetros
eta = 0.8 # learning rate
epochs = 15 # cantidad de épocas para el entrenamiento
w = np.array([0, 0]) # pesos iniciales
b = 1
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
    
plt.figure(figsize=(6,6))
# Puntos de clase 0
plt.scatter(X[y==0, 0], X[y==0, 1], color='red', label='Clase 0')
# Puntos de clase 1
plt.scatter(X[y==1, 0], X[y==1, 1], color='blue', label='Clase 1')

# Recta de decisión: w1*x1 + w2*x2 + b = 0 -> x2 = -(w1*x1 + b)/w2
if w[1] != 0:
    x_vals = np.array([min(X[:,0])-1, max(X[:,0])+1])
    y_vals = -(w[0]*x_vals + b)/w[1]
    plt.plot(x_vals, y_vals, 'k--', label='Recta de decisión')

plt.xlabel('x1')
plt.ylabel('x2')
plt.legend()
plt.grid(True)
plt.title('Perceptrón - Función OR')
plt.show()