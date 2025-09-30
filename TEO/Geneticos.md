
# 🧬 Algoritmos Genéticos

Los **algoritmos genéticos (AGs)** son una familia de algoritmos de búsqueda inspirados en los **principios de la evolución** en la naturaleza.
Imitan el proceso de **selección natural y reproducción**, generando **soluciones de alta calidad** para problemas de búsqueda, optimización y aprendizaje.

> ⚡ **Idea central:** una población de soluciones evoluciona iterativamente hacia mejores configuraciones.

---

## 🌱 Principios Evolutivos

* 🔄 **Variación:** los individuos presentan rasgos distintos.
* 🧾 **Herencia:** los rasgos se transmiten a la descendencia.
* 🏆 **Selección:** los más aptos sobreviven y se reproducen.

---

Son Algoritmos de **busqueda informada**, busqueda de **configuraciones exitosas** a partir de ciertas **configuraciones iniciales**, mediante la aplicacion de **reglas predifinidas de reconfiguracion o avance**

En los algoritmos genéticos, una **población de soluciones candidatas evoluciona de forma iterativa**: las mejores soluciones tienen más probabilidades de ser seleccionadas y transmitir sus características a la siguiente generación, logrando así **mejorar progresivamente la calidad de las soluciones al problema.**

---

## 🧩 Componentes principales

* **Genotipo:** colección de genes en cromosomas (los valores se llaman *alelos*).
* **Población:** conjunto de soluciones candidatas.
* **Fitness:** función de aptitud que mide qué tan buena es una solución.

---

## 🔄 Ciclo de Evolución

```text
Inicializar población  →  Evaluar fitness  →  Selección
        ↓                         ↑
  Crossover + Mutación  →  Nueva generación
```

---

## 🎲 Operadores Genéticos

### 🔎 Selección
>Se seleccionan aquellos con mayor puntuacion para combinarse y producir la siguiente generacion.

* 🎡 **Ruleta**: probabilidad proporcional al fitness.
* 🎯 **Torneo**: se eligen varios y gana el mejor.
* 📊 **Escalado de aptitud**: ajuste lineal.
* ➗ **SUS**: muestreo universal estocástico.

### 🧬 Crossover

* 1️⃣ Punto único
* 2️⃣ Dos puntos
* 🔀 Uniforme

### ⚡ Mutación
>Una **mutacion produce un cambio imprevisto** en algun individuo de la poblacion\
La **mutacion ayuda a mantener la diversidad en la poblacion**, y en muchos casos a recuperar informacion perdida en la evolucion de la poblacion.

* 🔄 Inversión de gen
* 🔀 Intercambio
* ↩️ Inversión de secuencia
* 🎲 Reorganización

## Eliminacion 
Es esencial mantener acotado el tamaño de las poblaciones 
Por esta razon, durante la evolucion se deben eliminar individuos de la poblacion. Los individuos elegidos para eliminar son aquellos "menos aptos" (de acuerdo a la funcioin de fitness)

### 🏅 Elitismo

> 🔒 Garantiza que los **mejores individuos** pasen a la siguiente generación.

---

>Las características clave que distinguen a los algoritmos genéticos son: 
 
* Mantener una población de soluciones.

* Usar una representación genética de las soluciones. 
  
* Emplear una función de aptitud para evaluar los resultados. 
 
* Presentar un comportamiento probabilístico

---

## ✅ Ventajas vs ❌ Limitaciones

| ✅ Ventajas                         | ❌ Limitaciones                    |
| ---------------------------------- | --------------------------------- |
| Optimización global                | Requiere definir bien el problema |
| Maneja representaciones complejas  | Ajuste de hiperparámetros         |
| Resiliente al ruido                | Costoso computacionalmente        |
| Compatible con paralelismo         | Riesgo de convergencia prematura  |
| Adecuado para aprendizaje continuo | No garantiza el óptimo            |

---

## 🧮 Multi-Objetivo (MOGA)

* Optimiza varias funciones en conflicto.
* Basado en **dominancia de Pareto**.
* 🎯 **NSGA-II:** mantiene diversidad con *crowding distance*.

📌 Ejemplo: minimizar **costo** y **contaminación** → frente de Pareto con distintas soluciones no dominadas.

---

👉 De esta forma no solo queda ordenado, sino también **amigable para leer y estudiar**.

¿Querés que te arme una **plantilla lista en .md** con este formato vistoso para que la uses directamente?
