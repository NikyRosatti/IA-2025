# Ensemble Learning 📊✨

**Ensemble Learning** combina múltiples modelos para mejorar la **precisión, estabilidad y robustez** de las predicciones. En lugar de depender de un solo modelo, usamos la fuerza de varios modelos trabajando juntos.  

---

## Voting 🗳️

**Voting** combina diferentes modelos y decide la predicción final basándose en la **votación de sus resultados**.  

### Tipos de Voting

- **Hard Voting (votación dura)**  
  - Cada modelo emite su predicción como **clase discreta**.  
  - La clase que obtiene la **mayoría de votos** es la predicción final.  
  - Ejemplo:
    ```
    Modelo A: 0
    Modelo B: 1
    Modelo C: 1
    → Resultado final: 1
    ```

- **Soft Voting (votación suave)**  
  - Cada modelo predice **probabilidades** para cada clase.  
  - Se suman o promedian las probabilidades.  
  - La clase con la **mayor probabilidad promedio** se elige como resultado final.  

- **Weighted Voting** 🏋️‍♂️  
  - Se pueden asignar pesos a los modelos para que algunos tengan más influencia en la votación.

---

## Bagging 🎲

**Bagging (Bootstrap Aggregating)** crea múltiples versiones de un modelo usando **subconjuntos de datos diferentes**, entrenando un modelo en cada subconjunto y combinando sus predicciones mediante voting.  

- Útil cuando los datos son **ruidosos o pequeños**.  
- Ejemplo: **Random Forest**
  - Conjunto de **Decision Trees**.  
  - Cada árbol se entrena con una **muestra aleatoria de los datos**.  
  - La predicción final se obtiene por **votación**.  
  - Introduce **aleatoriedad** para mejorar la generalización.

---

## Boosting ⚡

**Boosting** crea modelos secuenciales, donde cada nuevo modelo aprende **de los errores de los anteriores**.  

- **Enfoque:** dar más peso a los datos que los modelos anteriores clasificaron incorrectamente.  
- **Utilidad:** mejora la **precisión**, ideal para problemas donde los errores son costosos.  
- Ejemplos: **AdaBoost, Gradient Boosting, XGBoost**.

---

## Stacking 💊

**Stacking** (apilamiento de modelos) combina varios modelos base entrenados de manera independiente y utiliza un **metamodelo** para aprender de sus predicciones.  

- A diferencia de Boosting, **no ajusta los modelos secuencialmente**, sino que todos los modelos base se entrenan primero y sus predicciones sirven como **entrada para un nuevo modelo (metamodelo)**.  
- El **metamodelo** aprende a combinar las salidas de los modelos base para obtener una predicción más precisa.  
- **Utilidad:**
    - Mezclar modelos distintos (por ejemplo, árboles de decisión + regresión logística + SVM).  
    - Capturar patrones complejos que un solo modelo no podría detectar.  
    - Mejorar precisión y generalización en problemas complejos de clasificación o regresión.

> Usa crossover para entrenar.

---

## Resumen 🎨

| Método       | Cómo funciona                                         | Uso principal                                  |
|-------------|------------------------------------------------------|-----------------------------------------------|
| Hard Voting  | Predicciones discretas, mayoría de votos            | Clasificación robusta                          |
| Soft Voting  | Promedio de probabilidades                            | Cuando los modelos predicen probabilidades confiables |
| Bagging      | Modelos independientes sobre muestras aleatorias    | Reducir varianza y overfitting                |
| Boosting     | Modelos secuenciales corrigiendo errores anteriores | Reducir bias, aumentar precisión              |
| Stacking     | Modelos base + metamodelo que aprende de sus salidas| Mezclar modelos y capturar patrones complejos |
