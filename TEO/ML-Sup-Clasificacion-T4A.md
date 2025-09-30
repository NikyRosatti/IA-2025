# IA-04A ML-Sup-Clasificacion

## Introducción
- Tema: Aprendizaje Supervisado - Clasificación
- El algoritmo aprende a partir de datos etiquetados para predecir nuevas entradas.
- Se diferencia en dos tipos principales de problemas:
  - **Clasificación**: salida discreta y finita (ej. spam / no spam).
  - **Regresión**: salida numérica (ej. precio de casas).

## Tipos de Clasificación
- **Binaria**: dos clases (ej. spam/no spam).
- **Multi-clase**: más de dos clases (ej. tipos de frutas).
- **Multi-etiqueta**: un ejemplo puede tener múltiples etiquetas (ej. características de una foto).

## Métricas de Evaluación
- **Matriz de Confusión**: TP, TN, FP, FN.
- **Accuracy**: (TP+TN)/(TP+TN+FP+FN), puede ser engañoso en clases desbalanceadas.
- **Precision**: TP / (TP+FP), mide confiabilidad de los positivos.
- **Recall**: TP / (TP+FN), mide detección de positivos.
- **F1-Score**: media armónica de precision y recall.

## Clasificación Lineal
- Se busca una **frontera de decisión**: w0 + w1x1 + w2x2 = 0.
- Regla de aprendizaje del perceptrón:  
  wi ← wi + α(y − hw(x)) * xi

## Ejemplo Perceptrón
- Algoritmo propuesto por Rosenblatt (1958).
- Ajusta pesos iterativamente según error y tasa de aprendizaje.

## Regresión Logística
- Usa la **función sigmoide**: σ(z) = 1 / (1 + e^(-z))
- Predice probabilidades en vez de clases duras.
- Función de costo: **Log Loss**
- Entrenamiento con **descenso del gradiente**.

## Ejemplo en Python
- Clasificación de cervezas (Lager vs Stout) usando color (SRM) y amargor (IBU).
- Implementación de regresión logística con descenso del gradiente.

