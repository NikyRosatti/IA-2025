# IA-04B ML-Sup-Clasificacion-NoParametrica

## Introducción
- Técnicas **no paramétricas**: no suponen forma funcional fija de los datos.
- Complejidad del modelo crece con la cantidad de datos.

## Técnicas Principales
1. **k-Nearest Neighbors (k-NN)**
   - Clasificación por mayoría de vecinos cercanos.
   - Hiperparámetros: k (número de vecinos), métrica de distancia.
   - Distancias: Euclídea (p=2), Manhattan (p=1), Chebyshev (p>2).
   - Requiere **escalado de datos** (normalización/estandarización).
   - Ventajas: simple, flexible.
   - Desventajas: costoso en datasets grandes, sensible al ruido.

2. **Árboles de decisión**
   - Dividen el espacio en regiones homogéneas.
   - No lineales, interpretables.

3. **Support Vector Machines (SVM)**
   - Separan clases maximizando el **margen**.
   - Utilizan **vectores soporte** para definir el hiperplano.
   - Funciona con kernel trick: transforma datos a espacio de mayor dimensión.
   - Kernels comunes: lineal, polinómico, RBF (gaussiano), sigmoide.
   - **Soft margin**: permite errores controlados para mejorar generalización.

## Implementaciones en Python
- k-NN con `scikit-learn`: `KNeighborsClassifier`
- SVM con `scikit-learn`: `SVC`, soporte para distintos kernels.

